use std::ops::AddAssign;

use crate::Bag;

pub fn consolidate<T: Ord>(batch: &mut Bag<T>) {
    batch.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    consolidate_sorted(batch);
}

/// The merge half of [`consolidate`], for batches already sorted by key:
/// sum equal-keyed diffs, drop zeroes, in place, no allocation.
#[inline]
pub fn consolidate_sorted<T: PartialOrd>(batch: &mut Bag<T>) {
    debug_assert!(batch.is_sorted_by(|a, b| a.0 <= b.0));
    let mut write = 0usize;
    for read in 0..batch.len() {
        if write > 0 && batch[write - 1].0 == batch[read].0 {
            let d = batch[read].1;
            (&mut batch[write - 1].1).add_assign(d);
        } else {
            // previous group is sealed
            // if it cancelled, reuse its slot
            if write > 0 && batch[write - 1].1 == 0 {
                write -= 1;
            }
            batch.swap(write, read);
            write += 1;
        }
    }
    if write > 0 && batch[write - 1].1 == 0 {
        write -= 1;
    }
    batch.truncate(write);
}
