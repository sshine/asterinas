// SPDX-License-Identifier: MPL-2.0

//! Classification of logical block ranges as mapped runs, unwritten runs, or
//! holes.

use super::extent::{ExtentTree, Mapping};
use crate::fs::fs_impls::ext4::prelude::*;

/// What backs the current logical interval.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum IoRange {
    /// A contiguous mapped device-block range.
    Mapped(Range<Ext4Bid>),
    /// Blocks that are allocated but were never written.
    Unwritten(Range<Iblock>),
    /// Blocks with no mapping at all.
    Hole(Range<Iblock>),
}

/// Iterator over the runs that make up a logical block range.
///
/// Each item preserves the classification that block-backed I/O needs:
/// contiguous device mappings stay grouped, and ranges that read as zeros stay
/// explicit rather than being served from the device.
pub(super) struct IoRangeIter<'a> {
    range: Range<Iblock>,
    tree: &'a ExtentTree,
    device: &'a dyn BlockDevice,
}

impl<'a> IoRangeIter<'a> {
    pub(super) fn new(
        range: Range<Iblock>,
        tree: &'a ExtentTree,
        device: &'a dyn BlockDevice,
    ) -> Self {
        Self {
            range,
            tree,
            device,
        }
    }

    /// Returns the next run, or `None` once the range is exhausted.
    pub(super) fn next(&mut self) -> Result<Option<IoRange>> {
        if self.range.is_empty() {
            return Ok(None);
        }

        let start = self.range.start;
        let max_blocks = self.range.len() as u32;
        let mapping = self.tree.lookup(self.device, start, max_blocks)?;
        debug_assert!(mapping.len() > 0);
        self.range.start += mapping.len();

        Ok(Some(match mapping {
            Mapping::Mapped { start, len } => IoRange::Mapped(start..start + len as u64),
            Mapping::Unwritten { len } => IoRange::Unwritten(start..start + len),
            Mapping::Hole { len } => IoRange::Hole(start..start + len),
        }))
    }
}
