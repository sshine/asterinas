// SPDX-License-Identifier: MPL-2.0

//! Dirty tracking for values that are written back to a device.

use core::ops::{Deref, DerefMut};

use crate::prelude::*;

/// A value with dirty tracking.
///
/// Taking a mutable reference marks the value dirty, so a filesystem can
/// decide what to write back without comparing against the on-disk copy.
/// Dropping a value that is still dirty loses an update, hence the warning.
pub(crate) struct Dirty<T: Debug> {
    value: T,
    dirty: bool,
}

impl<T: Debug> Dirty<T> {
    /// Creates a new `Dirty` value without setting the dirty flag.
    pub(crate) fn new(val: T) -> Dirty<T> {
        Dirty {
            value: val,
            dirty: false,
        }
    }

    /// Creates a new `Dirty` value with the dirty flag set.
    pub(crate) fn _new_dirty(val: T) -> Dirty<T> {
        Dirty {
            value: val,
            dirty: true,
        }
    }

    /// Returns whether the value is dirty.
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clears the dirty flag.
    pub(crate) fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

impl<T: Debug> Deref for Dirty<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T: Debug> DerefMut for Dirty<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.dirty = true;
        &mut self.value
    }
}

impl<T: Debug> Drop for Dirty<T> {
    fn drop(&mut self) {
        if self.is_dirty() {
            warn!("dropped while dirty: {:?}", self.value);
        }
    }
}

impl<T: Debug> Debug for Dirty<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tag = if self.dirty { "Dirty" } else { "Clean" };
        write!(f, "[{}] {:?}", tag, self.value)
    }
}
