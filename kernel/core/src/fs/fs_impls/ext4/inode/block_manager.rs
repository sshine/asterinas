// SPDX-License-Identifier: MPL-2.0

//! The page-cache backend of one ext4 inode.
//!
//! `InodeBlockManager` joins the inode's logical view of its file to the block
//! device, translating through the extent tree. Ranges that are unwritten or
//! absent are served as zeros without touching the device.
//!
//! Writing is not implemented, so every write submission reports `EROFS`. The
//! page cache never asks for one on a read-only mount; the arm exists so that
//! a mistake elsewhere fails rather than corrupting the volume.

use core::sync::atomic::{AtomicUsize, Ordering};

use aster_block::bio::BioCompleteFn;

use super::{
    extent::{ExtentTree, Mapping},
    io_range::{IoRange, IoRangeIter},
};
use crate::{
    fs::fs_impls::ext4::{fs::Ext4, prelude::*},
    vm::page_cache::PageRun,
};

/// Bridges one inode's logical file view and the physical block device.
#[derive(Debug)]
pub(super) struct InodeBlockManager {
    extents: ExtentTree,
    /// Cached `npages` bound for `PageCache`.
    npages: AtomicUsize,
    /// Filesystem handle, for the block device and for node reads.
    fs: Weak<Ext4>,
}

impl InodeBlockManager {
    pub(super) fn new(extents: ExtentTree, fs: Weak<Ext4>, npages: usize) -> Self {
        Self {
            extents,
            npages: AtomicUsize::new(npages),
            fs,
        }
    }

    /// Returns a strong reference to the owning filesystem.
    pub(super) fn fs(&self) -> Result<Arc<Ext4>> {
        self.fs
            .upgrade()
            .ok_or_else(|| Error::with_message(Errno::EIO, "filesystem already dropped"))
    }

    /// Creates an iterator over the runs backing `block_range`.
    pub(super) fn iter_io_ranges<'a>(
        &'a self,
        fs: &'a Ext4,
        block_range: Range<Iblock>,
    ) -> IoRangeIter<'a> {
        IoRangeIter::new(block_range, &self.extents, fs.block_device())
    }

    fn check_bounds(&self, end_idx: usize) -> Result<()> {
        if end_idx > self.npages.load(Ordering::Acquire) {
            return_errno_with_message!(Errno::EINVAL, "I/O beyond the end of the file");
        }

        Ok(())
    }
}

impl BlockAsPageCacheBackend for InodeBlockManager {
    fn submit_read_bio(
        &self,
        idx: usize,
        bio_segment: BioSegment,
        complete_fn: BioCompleteFn,
        io_batch: &mut IoBatch,
    ) -> Result<()> {
        self.check_bounds(idx + 1)?;

        let fs = self.fs()?;
        match self.extents.lookup(fs.block_device(), idx as Iblock, 1)? {
            Mapping::Mapped { start, .. } => {
                fs.read_blocks_async(start, bio_segment, Some(complete_fn), io_batch)
            }
            // Nothing on the device backs these blocks, so the page is zeroed
            // without any I/O at all.
            Mapping::Unwritten { .. } | Mapping::Hole { .. } => {
                complete_fn(BioStatus::Zeros);
                Ok(())
            }
        }
    }

    fn submit_read_page_run(&self, mut pages: PageRun<'_>, io_batch: &mut IoBatch) -> Result<()> {
        let start_idx = pages.start_idx();
        let end_idx = start_idx + pages.len();
        self.check_bounds(end_idx)?;

        let fs = self.fs()?;
        let mut io_ranges = self.iter_io_ranges(&fs, start_idx as Iblock..end_idx as Iblock);

        while let Some(io_range) = io_ranges.next()? {
            match io_range {
                IoRange::Mapped(device_range) => {
                    let nblocks = (device_range.end - device_range.start) as usize;
                    pages.submit_contiguous_read_pages(
                        fs.block_device(),
                        Bid::new(device_range.start),
                        nblocks,
                        io_batch,
                    )?;
                }
                IoRange::Unwritten(range) | IoRange::Hole(range) => {
                    for locked_page in pages.by_ref().take(range.len()) {
                        locked_page
                            .fill_zeros(0, PAGE_SIZE)
                            .expect("a cache page is exactly one page long");
                        locked_page.set_up_to_date();
                    }
                }
            }
        }

        Ok(())
    }

    fn submit_write_bio(
        &self,
        _idx: usize,
        _bio_segment: BioSegment,
        _complete_fn: BioCompleteFn,
        _io_batch: &mut IoBatch,
    ) -> Result<()> {
        return_errno_with_message!(Errno::EROFS, "ext4 writing is not implemented yet")
    }
}
