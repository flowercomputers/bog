//! The maintenance thread: when work is due, and how commits wait for it.

use super::*;

/// Coordination between commits and the maintenance thread.
#[derive(Default)]
pub(super) struct Maintenance {
    pub(super) state: Mutex<MaintenanceState>,
    /// The thread waits on this for work.
    pub(super) wake: Condvar,
    /// Signalled whenever the thread installs a result or goes idle;
    /// throttled commits and [`Store::quiesce`] wait on it.
    pub(super) progress: Condvar,
}

#[derive(Default)]
pub(super) struct MaintenanceState {
    /// Work was asked for since the thread last started a pass.
    pub(super) requested: bool,
    /// The thread is in a pass.
    pub(super) busy: bool,
    pub(super) shutdown: bool,
    /// The thread panicked and exited.
    pub(super) dead: bool,
    /// How the last pass failed, if it did. Cleared by a clean pass.
    pub(super) error: Option<Error>,
}

impl Maintenance {
    pub(super) fn lock(&self) -> MutexGuard<'_, MaintenanceState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Store {
    /// Wait until the maintenance thread has finished a full pass begun
    /// after this call: layers are folded and tables compacted as far as
    /// the policy wants. Returns how that pass failed, if it did.
    pub fn quiesce(&self) -> Result<()> {
        self.assert_no_tx_here("quiesce");
        let m = &self.maintenance;
        let mut st = m.lock();
        self.request_locked(&mut st);
        while (st.requested || st.busy) && !st.dead && !st.shutdown {
            st = m.progress.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
        if st.dead {
            return Err(Error::MaintenancePanicked);
        }
        st.error.clone().map_or(Ok(()), Err)
    }
}

impl Shared {
    /// Whether there may be a checkpoint to write, layers to fold or
    /// tables to compact.
    pub(super) fn maintenance_due(&self) -> bool {
        let v = self.current.load();
        v.layers.len() - v.live_from() >= 2
            || v.tables.len() > self.opts.max_tables
            || self.over_limits(1)
    }

    /// Wake the maintenance thread, starting it if this is the first
    /// request. If it cannot be started, the request is recorded as failed.
    pub(crate) fn request_maintenance(&self) {
        self.request_locked(&mut self.maintenance.lock());
    }

    pub(super) fn request_locked(&self, st: &mut MaintenanceState) {
        st.requested = true;
        self.maintenance.wake.notify_one();
        if st.shutdown || st.dead {
            return;
        }
        let mut worker = self.worker.lock().unwrap_or_else(PoisonError::into_inner);
        if worker.is_some() {
            return;
        }
        let me = self.me.upgrade().expect("a store is alive while used");
        match thread::Builder::new()
            .name("bog-maintenance".into())
            .spawn(move || me.maintenance_loop())
        {
            Ok(handle) => *worker = Some(handle),
            Err(e) => {
                st.requested = false;
                st.error = Some(Error::io("starting the maintenance thread", e));
                self.maintenance.progress.notify_all();
            }
        }
    }

    /// Tell anyone waiting on maintenance that something changed.
    pub(super) fn progress(&self) {
        let _st = self.maintenance.lock();
        self.maintenance.progress.notify_all();
    }

    /// Wait while there are more than `max_layers` layers or twice
    /// `max_tables` tables, or the WAL or memtable is at twice its limit,
    /// so reads and memory stay bounded when commits outrun the
    /// maintenance thread. Fails if a pass run since it started waiting
    /// failed, rather than waiting on one that keeps failing.
    pub(super) fn throttle(&self) -> Result<()> {
        let over = || {
            let v = self.current.load();
            v.layers.len() > self.opts.max_layers
                || v.tables.len() > 2 * self.opts.max_tables
                || self.over_limits(2)
        };
        if !over() {
            return Ok(());
        }
        let m = &self.maintenance;
        let mut st = m.lock();
        let mut asked = false;
        // checked under the lock the thread signals progress under, so a
        // result installed after this check wakes the wait below
        while over() {
            if st.dead {
                return Err(Error::MaintenancePanicked);
            }
            if st.shutdown {
                break;
            }
            // idle after being asked: the pass that answered has finished
            if asked
                && !st.requested
                && !st.busy
                && let Some(e) = &st.error
            {
                return Err(e.clone());
            }
            asked = true;
            self.request_locked(&mut st);
            st = m.progress.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
        Ok(())
    }

    /// The maintenance thread: wait for a request, then run a pass, until
    /// the store is dropped. A failed pass is recorded and retried at the
    /// next request; a panic is recorded and ends the thread.
    pub(super) fn maintenance_loop(&self) {
        let m = &self.maintenance;
        loop {
            {
                let mut st = m.lock();
                while !st.requested && !st.shutdown {
                    st = m.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
                }
                if st.shutdown {
                    return;
                }
                st.requested = false;
                st.busy = true;
            }
            let res = panic::catch_unwind(AssertUnwindSafe(|| self.maintain()));
            let mut st = m.lock();
            st.busy = false;
            match res {
                Ok(res) => st.error = res.err(),
                Err(_) => {
                    st.dead = true;
                    st.error = Some(Error::MaintenancePanicked);
                }
            }
            m.progress.notify_all();
            if st.dead {
                return;
            }
        }
    }

    /// One maintenance pass: checkpoint if the WAL or memtable is over its
    /// limit, fold layers until the plan is satisfied, then compact tables
    /// until there are few enough.
    pub(super) fn maintain(&self) -> Result<()> {
        // a failed checkpoint should not stop folds and compaction, which
        // keep reads fast while it is retried
        let checkpointed = if self.over_limits(1) {
            self.checkpoint_now().map(|_| ())
        } else {
            Ok(())
        };
        while !self.shutdown.load(Relaxed) && self.fold_once() {
            std::hint::spin_loop()
        }
        let compacted = self.compact_while_over().map(|_| ());
        checkpointed.and(compacted)
    }
}
