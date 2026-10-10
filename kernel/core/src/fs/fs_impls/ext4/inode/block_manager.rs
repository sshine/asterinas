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

#[cfg(ktest)]
mod test {
    use ostd::prelude::ktest;

    use super::*;
    use crate::fs::{
        fs_impls::ext4::{
            fs,
            test_utils::{
                Ext4VolumeBuilder, SPARSE_INO, TWO_EXTENT_FILL, TWO_EXTENT_INO, assert_errno,
            },
        },
        vfs::file_system::{FileSystem, FsFlags},
    };

    /// Mounts a fixture volume and keeps it alive alongside the filesystem.
    fn mount() -> (Arc<dyn FileSystem>, Arc<Ext4>) {
        let volume = Ext4VolumeBuilder::new().build();
        let fs = fs::open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs
            .downcast_ref::<Ext4>()
            .expect("should be an ext4 volume")
            .root_inode()
            .expect("root should be readable")
            .fs()
            .expect("the filesystem should outlive its root");
        (fs, ext4)
    }

    /// Submits one page read and returns the status it completed with, along
    /// with the bytes it left in the segment.
    fn read_one_page(
        manager: &InodeBlockManager,
        idx: usize,
    ) -> Result<(BioStatus, [u8; BLOCK_SIZE])> {
        let bio_segment = BioSegment::alloc(1, BioDirection::FromDevice);
        let status: Arc<SpinLock<Option<BioStatus>>> = Arc::new(SpinLock::new(None));
        let recorder = status.clone();
        let complete_fn: BioCompleteFn =
            Box::new(move |completed| *recorder.lock() = Some(completed));

        let mut io_batch = IoBatch::new();
        manager.submit_read_bio(idx, bio_segment.clone(), complete_fn, &mut io_batch)?;
        io_batch.wait_all()?;

        let mut bytes = [0u8; BLOCK_SIZE];
        bio_segment.read_bytes(0, &mut bytes)?;
        let status = status.lock().expect("the read should have completed");
        Ok((status, bytes))
    }

    /// The page cache reads a run of pages where it can and one page where it
    /// cannot, so the single-page path needs its own test rather than riding
    /// on the run one.
    #[ktest]
    fn one_page_is_read_from_the_block_its_extent_names() {
        let (_fs, ext4) = mount();
        let inode = ext4.read_inode(TWO_EXTENT_INO).expect("should read");
        let manager = inode
            .payload
            .block_manager()
            .expect("a regular file has a block manager");

        let (status, bytes) = read_one_page(manager, 1).expect("should read");

        assert_eq!(status, BioStatus::Complete);
        assert!(bytes.iter().all(|&byte| byte == TWO_EXTENT_FILL[1]));
    }

    /// A page with nothing behind it is completed as zeros without any I/O,
    /// which is what keeps a sparse file from reading the device.
    #[ktest]
    fn a_page_over_a_hole_completes_as_zeros() {
        let (_fs, ext4) = mount();
        let inode = ext4.read_inode(SPARSE_INO).expect("should read");
        let manager = inode
            .payload
            .block_manager()
            .expect("a regular file has a block manager");

        let (status, _) = read_one_page(manager, 1).expect("should read");

        assert_eq!(status, BioStatus::Zeros);
    }

    #[ktest]
    fn reading_a_page_past_the_end_of_the_file_is_refused() {
        let (_fs, ext4) = mount();
        let inode = ext4.read_inode(TWO_EXTENT_INO).expect("should read");
        let manager = inode
            .payload
            .block_manager()
            .expect("a regular file has a block manager");

        assert_errno(read_one_page(manager, 2), Errno::EINVAL);
    }

    /// Nothing should ever ask a read-only mount to write a page back, so the
    /// refusal is a guard rather than a path. It still has to work.
    #[ktest]
    fn writing_a_page_back_reports_erofs() {
        let (_fs, ext4) = mount();
        let inode = ext4.read_inode(TWO_EXTENT_INO).expect("should read");
        let manager = inode
            .payload
            .block_manager()
            .expect("a regular file has a block manager");

        let bio_segment = BioSegment::alloc(1, BioDirection::ToDevice);
        let complete_fn: BioCompleteFn = Box::new(|_| {});
        let mut io_batch = IoBatch::new();

        assert_errno(
            manager.submit_write_bio(0, bio_segment, complete_fn, &mut io_batch),
            Errno::EROFS,
        );
    }
}
