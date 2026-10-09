// SPDX-License-Identifier: MPL-2.0

//! Test-only fixtures and helpers for ext4 kernel tests.
//!
//! Volumes are built in memory field by field rather than loaded from an image
//! produced by `mkfs.ext4`. That keeps the repository free of binary blobs, and
//! more importantly it lets a test state the one field it cares about, which is
//! what makes the rejection paths testable at all.

use core::fmt;

use aster_block::{
    BlockDeviceMeta, SECTOR_SIZE,
    bio::{BioEnqueueError, BioStatus, BioType, SubmittedBio},
};
use device_id::{DeviceId, MajorId, MinorId};
use ostd::mm::{FrameAllocOptions, HasSize, Segment, io::util::HasVmReaderWriter};

use super::{
    features::{FeatureCompat, FeatureIncompat, FeatureRoCompat},
    prelude::*,
    super_block::{MAGIC_NUM, RawSuperBlock, SUPER_BLOCK_OFFSET},
};

/// Asserts that `result` is the given error.
pub(super) fn assert_errno<T: Debug>(result: Result<T>, expected: Errno) {
    match result {
        Ok(value) => panic!("expected {:?}, got Ok({:?})", expected, value),
        Err(err) => assert_eq!(err.error(), expected),
    }
}

/// A block device backed by memory, for tests that need no image on disk.
pub(super) struct Ext4MemoryDisk {
    segment: Segment<()>,
    minor: u32,
}

impl Ext4MemoryDisk {
    /// Creates a zeroed device of `nblocks` blocks.
    ///
    /// Each device gets a distinct ID because `Ext4Type` keys its instance
    /// cache on it, and two devices sharing one ID would be served the same
    /// filesystem across unrelated tests.
    pub(super) fn new(nblocks: usize) -> Self {
        use core::sync::atomic::{AtomicU32, Ordering};

        static NEXT_MINOR: AtomicU32 = AtomicU32::new(0);

        let npages = (nblocks * BLOCK_SIZE).div_ceil(PAGE_SIZE);
        let segment = FrameAllocOptions::new()
            .zeroed(true)
            .alloc_segment(npages)
            .unwrap();

        Self {
            segment,
            minor: NEXT_MINOR.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Creates a device holding a volume that `SuperBlock::try_from` accepts.
    pub(super) fn with_valid_super_block() -> Self {
        let disk = Self::new(64);
        disk.write_super_block(&make_valid_raw_super_block());
        disk
    }

    pub(super) fn write_super_block(&self, raw: &RawSuperBlock) {
        self.segment.write_val(SUPER_BLOCK_OFFSET, raw).unwrap();
    }
}

impl Debug for Ext4MemoryDisk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ext4MemoryDisk")
            .field("bytes", &self.segment.size())
            .finish()
    }
}

impl BlockDevice for Ext4MemoryDisk {
    fn enqueue(&self, bio: SubmittedBio) -> core::result::Result<(), BioEnqueueError> {
        if bio.type_() == BioType::Flush {
            bio.complete(BioStatus::Complete);
            return Ok(());
        }

        let mut cur_device_ofs = bio.sid_range().start.to_raw() as usize * SECTOR_SIZE;

        for seg in bio.segments() {
            let io_size = match bio.type_() {
                BioType::Read => seg
                    .dma_slice()
                    .writer()
                    .unwrap()
                    .write(self.segment.reader().skip(cur_device_ofs)),
                BioType::Write => self
                    .segment
                    .writer()
                    .skip(cur_device_ofs)
                    .write(&mut seg.dma_slice().reader().unwrap()),
                _ => {
                    bio.complete(BioStatus::NotSupported);
                    return Ok(());
                }
            };
            cur_device_ofs += io_size;
        }

        bio.complete(BioStatus::Complete);
        Ok(())
    }

    fn metadata(&self) -> BlockDeviceMeta {
        BlockDeviceMeta {
            max_nr_segments_per_bio: usize::MAX,
            nr_sectors: self.segment.size() / SECTOR_SIZE,
        }
    }

    fn name(&self) -> &str {
        "ext4-memory-disk"
    }

    fn id(&self) -> DeviceId {
        DeviceId::new(MajorId::new(1), MinorId::new(self.minor))
    }
}

/// Builds a superblock with the values `mkfs.ext4 -b 4096 -I 256` produces.
pub(super) fn make_valid_raw_super_block() -> RawSuperBlock {
    let mut raw = RawSuperBlock::new_zeroed();

    raw.magic = MAGIC_NUM;
    raw.log_block_size = 2;
    raw.log_cluster_size = 2;
    raw.first_data_block = 0;
    raw.blocks_per_group = 32768;
    raw.clusters_per_group = 32768;
    raw.inodes_per_group = 8192;
    raw.inodes_count = raw.inodes_per_group;
    raw.blocks_count_lo = 262144;
    raw.rev_level = 1;
    raw.first_ino = 11;
    raw.inode_size = 256;
    raw.desc_size = 64;
    raw.want_extra_isize = 32;
    raw.feature_compat = FeatureCompat::EXT_ATTR.bits()
        | FeatureCompat::RESIZE_INODE.bits()
        | FeatureCompat::DIR_INDEX.bits();
    raw.feature_incompat = FeatureIncompat::FILETYPE.bits()
        | FeatureIncompat::EXTENTS.bits()
        | FeatureIncompat::SIZE_64BIT.bits()
        | FeatureIncompat::FLEX_BG.bits();
    raw.feature_ro_compat = FeatureRoCompat::SPARSE_SUPER.bits()
        | FeatureRoCompat::LARGE_FILE.bits()
        | FeatureRoCompat::HUGE_FILE.bits()
        | FeatureRoCompat::DIR_NLINK.bits()
        | FeatureRoCompat::EXTRA_ISIZE.bits();

    raw
}
