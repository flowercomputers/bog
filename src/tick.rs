//! Ticks: writes queued from any thread, then run together as one
//! transaction when the application says so.
//!
//! A [`Ticker`] wraps a schema's [`Db`]. [`submit`](Ticker::submit)
//! queues a unit of writes; [`tick`](Ticker::tick) runs the
//! [systems](Ticker::system), then every queued unit in the order
//! submitted, and commits once. So a tick is one graph settlement, one
//! log record and one commit to subscribers and
//! [`on_changes`](crate::schema::DbBuilder::on_changes) hooks, however
//! many units it ran.
//!
//! Each unit runs as an `attempt`: one that fails, or that a
//! constraint refuses, is undone alone and the tick carries on without
//! it. [`submit_then`](Ticker::submit_then) reports each unit's outcome
//! once the tick has committed.
//!
//! The ticker owns no clock: the application calls `tick` on whatever
//! schedule it keeps, and ticks are numbered from one. A schema that
//! wants the number stored, replicated or visible to views writes it to
//! a [`Cell`](crate::schema::Cell) from a system.

use std::sync::{Mutex, PoisonError};

use crate::{
    error::Result,
    schema::{Db, Declare, Schema},
};

type Unit<S> = Box<dyn for<'a, 'b> FnOnce(&'a mut <S as Schema>::Write<'b>) -> Result<()> + Send>;
type Done = Box<dyn FnOnce(Result<()>) + Send>;
type System<S> =
    Box<dyn for<'a, 'b> FnMut(&'a mut <S as Schema>::Write<'b>, u64) -> Result<()> + Send>;

/// A [`Db`] written one tick at a time. See the [module docs](self).
pub struct Ticker<S: Schema> {
    db: Db<S>,
    inbox: Mutex<Vec<(Unit<S>, Option<Done>)>>,
    /// The systems and the last tick run; held for the whole of a tick,
    /// so ticks do not overlap.
    state: Mutex<State<S>>,
}

struct State<S: Schema> {
    systems: Vec<System<S>>,
    tick: u64,
}

impl<S: Declare> Ticker<S> {
    /// Tick `db`, numbering the first tick `after + 1`: pass zero for a
    /// new database, or the last tick a reopened one stored.
    pub fn new(db: Db<S>, after: u64) -> Self {
        Ticker {
            db,
            inbox: Mutex::new(Vec::new()),
            state: Mutex::new(State {
                systems: Vec::new(),
                tick: after,
            }),
        }
    }

    /// Run `system` at the start of every tick, inside its transaction,
    /// with the tick's number: timers, simulation steps, the clock.
    /// Systems run in the order added. One that fails fails the tick.
    pub fn system(
        &self,
        system: impl for<'a, 'b> FnMut(&'a mut S::Write<'b>, u64) -> Result<()> + Send + 'static,
    ) {
        self.lock().systems.push(Box::new(system));
    }

    /// Queue `unit` for the next tick.
    pub fn submit(
        &self,
        unit: impl for<'a, 'b> FnOnce(&'a mut S::Write<'b>) -> Result<()> + Send + 'static,
    ) {
        self.push(Box::new(unit), None);
    }

    /// Queue `unit` for the next tick, and call `done` with its outcome
    /// once that tick has committed, on the thread that ticked. If the
    /// tick itself fails, every unit's outcome is the tick's error.
    pub fn submit_then(
        &self,
        unit: impl for<'a, 'b> FnOnce(&'a mut S::Write<'b>) -> Result<()> + Send + 'static,
        done: impl FnOnce(Result<()>) + Send + 'static,
    ) {
        self.push(Box::new(unit), Some(Box::new(done)));
    }

    fn push(&self, unit: Unit<S>, done: Option<Done>) {
        self.inbox
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((unit, done));
    }

    /// How many units are queued.
    pub fn queued(&self) -> usize {
        self.inbox
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Run a tick: the systems, then every queued unit, in one
    /// transaction. Returns the tick's number. Units submitted while it
    /// runs wait for the next.
    pub fn tick(&self) -> Result<u64> {
        let mut state = self.lock();
        let units = std::mem::take(&mut *self.inbox.lock().unwrap_or_else(PoisonError::into_inner));
        state.tick += 1;
        let tick = state.tick;
        let mut outcomes = Vec::with_capacity(units.len());
        let committed = self.db.write(|tx| {
            for system in &mut state.systems {
                system(tx, tick)?;
            }
            for (unit, done) in units {
                let outcome = S::attempt(tx, unit);
                outcomes.push((done, outcome));
            }
            Ok(())
        });
        for (done, outcome) in outcomes {
            if let Some(done) = done {
                done(match &committed {
                    Ok(()) => outcome,
                    Err(e) => Err(e.clone()),
                });
            }
        }
        committed.map(|()| tick)
    }

    /// The last tick run.
    pub fn last_tick(&self) -> u64 {
        self.lock().tick
    }

    /// The database, to read, subscribe to, or write between ticks.
    pub fn db(&self) -> &Db<S> {
        &self.db
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State<S>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
