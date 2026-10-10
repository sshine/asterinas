// SPDX-License-Identifier: MPL-2.0

//! Test-only fixtures and helpers for ext4 kernel tests.
//!
//! Volumes are built in memory field by field rather than loaded from an image
//! produced by `mkfs.ext4`. That keeps the repository free of binary blobs,
//! and more importantly it lets a test state the one field it cares about,
//! which is what makes the rejection paths testable at all.
//!
//! [`Ext4VolumeBuilder`] lays out a complete two-group volume whose contents
//! are described below. The geometry is far smaller than anything `mkfs.ext4`
//! would produce, because the driver reads its geometry from the superblock
//! and a realistic one would need a 4 GiB device to hold two groups.
//!
//! ```text
//! block    0  superblock
//!          1  group descriptors
//!          2  block bitmap, group 0
//!          3  inode bitmap, group 0
//!       4- 6  inode table, group 0
//!       6-    data
//!        128  block bitmap, group 1
//!        129  inode bitmap, group 1
//!    130-132  inode table, group 1
//!    132-     data
//! ```

use core::fmt;

use aster_block::{
    BlockDeviceMeta,
    bio::{BioEnqueueError, BioType, SubmittedBio},
};
use device_id::{DeviceId, MajorId, MinorId};
use ostd::mm::{FrameAllocOptions, HasSize, Segment};

use super::{
    features::FeatureIncompat,
    inode::RawInode,
    prelude::*,
    super_block::{MAGIC_NUM, RawSuperBlock, SUPER_BLOCK_OFFSET, SuperBlock},
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

    /// Writes one block, for a test that needs a node on a device without a
    /// volume around it.
    pub(super) fn write_block(&self, bid: u64, data: &[u8]) {
        self.segment
            .write_bytes(bid as usize * BLOCK_SIZE, data)
            .unwrap();
    }

    fn from_image(image: &[u8]) -> Self {
        let disk = Self::new(image.len().div_ceil(BLOCK_SIZE));
        disk.segment.write_bytes(0, image).unwrap();
        disk
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
    raw.feature_compat = REDUCED_COMPAT;
    raw.feature_incompat = REDUCED_INCOMPAT;
    raw.feature_ro_compat = REDUCED_RO_COMPAT;

    raw
}

/// The feature words of a volume made with everything this driver cannot
/// maintain suppressed.
const REDUCED_COMPAT: u32 = 0x38;
const REDUCED_INCOMPAT: u32 = 0x2c2;
const REDUCED_RO_COMPAT: u32 = 0x6b;

/// The feature words `mkfs.ext4` writes with no options given.
const STOCK_COMPAT: u32 = 0x103c;
const STOCK_INCOMPAT: u32 = 0x22c2;
const STOCK_RO_COMPAT: u32 = 0x46b;

const NR_GROUPS: usize = 2;
const BLOCKS_PER_GROUP: u32 = 128;
const INODES_PER_GROUP: u32 = 32;
const TOTAL_BLOCKS: u32 = BLOCKS_PER_GROUP * NR_GROUPS as u32;

/// Inode numbers of the files every built volume contains.
pub(super) const HELLO_INO: Ext4Ino = 11;
pub(super) const SYMLINK_INO: Ext4Ino = 12;
pub(super) const SPARSE_INO: Ext4Ino = 13;
pub(super) const TWO_EXTENT_INO: Ext4Ino = 14;
pub(super) const DEEP_INO: Ext4Ino = 15;
pub(super) const BAD_EXTENT_INO: Ext4Ino = 16;
pub(super) const DEVICE_INO: Ext4Ino = 17;
pub(super) const SECOND_GROUP_INO: Ext4Ino = INODES_PER_GROUP + 1;

pub(super) const HELLO_CONTENT: &[u8] = b"hello, ext4\n";
pub(super) const SYMLINK_TARGET: &str = "hello";
/// The byte each data block of the two-extent and deep files is filled with,
/// indexed by logical block.
pub(super) const TWO_EXTENT_FILL: [u8; 2] = [0xa1, 0xa2];
pub(super) const DEEP_FILL: [u8; 2] = [0xd1, 0xd2];
pub(super) const SPARSE_FILL: u8 = 0x51;
/// Logical block at which the sparse file resumes after its hole.
pub(super) const SPARSE_TAIL_IBLOCK: usize = 4;
pub(super) const SPARSE_BLOCKS: usize = 5;

/// The entries the root directory of a built volume holds, in order, each with
/// the `file_type` byte that names its kind.
pub(super) const ROOT_ENTRIES: [(&str, Ext4Ino, u8); 10] = [
    (".", 2, 2),
    ("..", 2, 2),
    ("hello", HELLO_INO, 1),
    ("link", SYMLINK_INO, 7),
    ("sparse", SPARSE_INO, 1),
    ("two", TWO_EXTENT_INO, 1),
    ("deep", DEEP_INO, 1),
    ("bad", BAD_EXTENT_INO, 1),
    ("null", DEVICE_INO, 3),
    ("second", SECOND_GROUP_INO, 1),
];

/// A volume laid out in memory, together with what a test needs to know about
/// where things were put.
pub(super) struct Ext4Volume {
    pub disk: Arc<dyn BlockDevice>,
    pub super_block: SuperBlock,
    inode_table_bids: [u64; NR_GROUPS],
}

impl Ext4Volume {
    pub(super) fn inode_table_bid(&self, group_idx: usize) -> u64 {
        self.inode_table_bids[group_idx]
    }
}

/// Lays out a complete ext4 volume in memory.
pub(super) struct Ext4VolumeBuilder {
    desc_size: usize,
    inode_size: usize,
    stock_features: bool,
    corrupt_inode_table: Option<(usize, u64)>,
}

impl Ext4VolumeBuilder {
    pub(super) fn new() -> Self {
        Self {
            desc_size: 64,
            inode_size: 256,
            stock_features: false,
            corrupt_inode_table: None,
        }
    }

    /// Uses 32-byte group descriptors, as a volume without `64BIT` does.
    pub(super) fn desc_size(mut self, desc_size: usize) -> Self {
        self.desc_size = desc_size;
        self
    }

    /// Declares the features `mkfs.ext4` sets when given no options.
    pub(super) fn stock_features(mut self) -> Self {
        self.stock_features = true;
        self
    }

    /// Points one group's inode table outside the volume.
    pub(super) fn corrupt_inode_table_bid(mut self, group_idx: usize, bid: u64) -> Self {
        self.corrupt_inode_table = Some((group_idx, bid));
        self
    }

    pub(super) fn build(self) -> Ext4Volume {
        let inode_table_blocks =
            (INODES_PER_GROUP as usize * self.inode_size).div_ceil(BLOCK_SIZE) as u64;
        let inode_table_bids = [4, BLOCKS_PER_GROUP as u64 + 2];
        let mut image = Image::new();
        let mut blocks = BlockAllocator::new(inode_table_bids, inode_table_blocks);

        self.write_super_block(&mut image);
        self.write_group_descs(&mut image, &inode_table_bids, inode_table_blocks);

        let files = self.write_files(&mut image, &mut blocks);
        for (ino, raw) in files {
            self.write_inode(&mut image, &inode_table_bids, ino, &raw);
        }

        let raw_super_block = image.read_val::<RawSuperBlock>(SUPER_BLOCK_OFFSET);
        Ext4Volume {
            disk: Arc::new(Ext4MemoryDisk::from_image(&image.bytes)) as Arc<dyn BlockDevice>,
            super_block: SuperBlock::try_from(raw_super_block)
                .expect("the built superblock should be valid"),
            inode_table_bids,
        }
    }

    fn write_super_block(&self, image: &mut Image) {
        let mut raw = make_valid_raw_super_block();
        raw.blocks_per_group = BLOCKS_PER_GROUP;
        raw.clusters_per_group = BLOCKS_PER_GROUP;
        raw.inodes_per_group = INODES_PER_GROUP;
        raw.inodes_count = INODES_PER_GROUP * NR_GROUPS as u32;
        raw.blocks_count_lo = TOTAL_BLOCKS;
        raw.inode_size = self.inode_size as u16;
        raw.desc_size = self.desc_size as u16;
        if self.stock_features {
            raw.feature_compat = STOCK_COMPAT;
            raw.feature_incompat = STOCK_INCOMPAT;
            raw.feature_ro_compat = STOCK_RO_COMPAT;
        }
        if self.desc_size == 32 {
            // The descriptor size field only means anything with `64BIT`, and
            // without it the short form is the only one defined.
            raw.feature_incompat &= !FeatureIncompat::SIZE_64BIT.bits();
            raw.desc_size = 0;
        }

        image.write_val(SUPER_BLOCK_OFFSET, &raw);
    }

    fn write_group_descs(
        &self,
        image: &mut Image,
        inode_table_bids: &[u64; NR_GROUPS],
        inode_table_blocks: u64,
    ) {
        let table_offset = BLOCK_SIZE;
        for (group_idx, &inode_table_bid) in inode_table_bids.iter().enumerate() {
            let group_first = group_idx as u64 * BLOCKS_PER_GROUP as u64;
            let mut raw = RawGroupDescImage::new_zeroed();
            raw.block_bitmap_lo = group_first as u32 + if group_idx == 0 { 2 } else { 0 };
            raw.inode_bitmap_lo = raw.block_bitmap_lo + 1;
            raw.inode_table_lo = inode_table_bid as u32;
            raw.free_blocks_count_lo = (BLOCKS_PER_GROUP as u64 - inode_table_blocks - 4) as u16;
            raw.free_inodes_count_lo = (INODES_PER_GROUP - 16) as u16;

            if let Some((corrupt_idx, bid)) = self.corrupt_inode_table
                && corrupt_idx == group_idx
            {
                raw.inode_table_lo = bid as u32;
                raw.inode_table_hi = (bid >> 32) as u32;
            }

            // A volume without `64BIT` stores only the first half, and writing
            // the whole structure would run into the next descriptor.
            image.write(
                table_offset + group_idx * self.desc_size,
                &raw.as_bytes()[..self.desc_size],
            );
        }
    }

    fn write_inode(
        &self,
        image: &mut Image,
        inode_table_bids: &[u64; NR_GROUPS],
        ino: Ext4Ino,
        raw: &RawInode,
    ) {
        let group_idx = ((ino - 1) / INODES_PER_GROUP) as usize;
        let index_in_group = ((ino - 1) % INODES_PER_GROUP) as usize;
        let offset =
            inode_table_bids[group_idx] as usize * BLOCK_SIZE + index_in_group * self.inode_size;
        image.write_val(offset, raw);
    }

    /// Writes every file's data and returns the inodes that describe them.
    fn write_files(
        &self,
        image: &mut Image,
        blocks: &mut BlockAllocator,
    ) -> Vec<(Ext4Ino, RawInode)> {
        let mut inodes = Vec::new();

        let root_bid = blocks.alloc();
        image.write(root_bid as usize * BLOCK_SIZE, &dir_block(&ROOT_ENTRIES));
        inodes.push((
            2,
            dir_inode(BLOCK_SIZE as u64, 1, extent_root(&[(0, root_bid, 1)])),
        ));

        let hello_bid = blocks.alloc();
        image.write(hello_bid as usize * BLOCK_SIZE, HELLO_CONTENT);
        inodes.push((
            HELLO_INO,
            file_inode(
                HELLO_CONTENT.len() as u64,
                1,
                extent_root(&[(0, hello_bid, 1)]),
            ),
        ));

        inodes.push((SYMLINK_INO, symlink_inode(SYMLINK_TARGET)));

        // A sparse file: one block, a three-block hole, then one more block.
        let sparse_head = blocks.alloc();
        let sparse_tail = blocks.alloc();
        image.fill(sparse_head, SPARSE_FILL);
        image.fill(sparse_tail, SPARSE_FILL);
        inodes.push((
            SPARSE_INO,
            file_inode(
                (SPARSE_BLOCKS * BLOCK_SIZE) as u64,
                2,
                extent_root(&[
                    (0, sparse_head, 1),
                    (SPARSE_TAIL_IBLOCK as u32, sparse_tail, 1),
                ]),
            ),
        ));

        // Two logically adjacent blocks that are not adjacent on the device,
        // which is what forces a second extent.
        let two_first = blocks.alloc();
        blocks.alloc();
        let two_second = blocks.alloc();
        image.fill(two_first, TWO_EXTENT_FILL[0]);
        image.fill(two_second, TWO_EXTENT_FILL[1]);
        inodes.push((
            TWO_EXTENT_INO,
            file_inode(
                2 * BLOCK_SIZE as u64,
                2,
                extent_root(&[(0, two_first, 1), (1, two_second, 1)]),
            ),
        ));

        // A tree one level deep: the inode holds an index node whose only
        // child is a leaf block.
        let deep_data = blocks.alloc_run(2);
        image.fill(deep_data, DEEP_FILL[0]);
        image.fill(deep_data + 1, DEEP_FILL[1]);
        let leaf_bid = blocks.alloc();
        image.write(
            leaf_bid as usize * BLOCK_SIZE,
            &extent_leaf_block(&[(0, deep_data, 2)]),
        );
        inodes.push((
            DEEP_INO,
            file_inode(
                2 * BLOCK_SIZE as u64,
                2,
                extent_index_root(&[(0, leaf_bid)]),
            ),
        ));

        let mut bad_root = extent_root(&[(0, hello_bid, 1)]);
        bad_root[0] = 0;
        inodes.push((BAD_EXTENT_INO, file_inode(BLOCK_SIZE as u64, 1, bad_root)));

        inodes.push((DEVICE_INO, device_inode(1, 3)));

        let second_bid = blocks.alloc_in_group(1);
        image.fill(second_bid, SECOND_GROUP_INO as u8);
        inodes.push((
            SECOND_GROUP_INO,
            file_inode(BLOCK_SIZE as u64, 1, extent_root(&[(0, second_bid, 1)])),
        ));

        inodes
    }
}

/// Hands out data blocks, stepping over the metadata of each group.
struct BlockAllocator {
    next: u64,
    reserved: [Range<u64>; NR_GROUPS],
}

impl BlockAllocator {
    fn new(inode_table_bids: [u64; NR_GROUPS], inode_table_blocks: u64) -> Self {
        let reserved = [
            0..inode_table_bids[0] + inode_table_blocks,
            BLOCKS_PER_GROUP as u64..inode_table_bids[1] + inode_table_blocks,
        ];
        Self {
            next: reserved[0].end,
            reserved,
        }
    }

    fn alloc(&mut self) -> u64 {
        self.alloc_run(1)
    }

    /// Hands out `count` consecutive blocks, which is what lets one extent
    /// cover more than one logical block.
    fn alloc_run(&mut self, count: u64) -> u64 {
        for reserved in &self.reserved {
            if reserved.contains(&self.next) || reserved.contains(&(self.next + count - 1)) {
                self.next = reserved.end;
            }
        }
        assert!(
            self.next + count <= TOTAL_BLOCKS as u64,
            "the fixture ran out of blocks"
        );

        let bid = self.next;
        self.next += count;
        bid
    }

    /// Hands out the first data block of `group_idx`, so that a test can reach
    /// a group other than the first.
    fn alloc_in_group(&mut self, group_idx: usize) -> u64 {
        self.next = self.next.max(self.reserved[group_idx].end);
        self.alloc()
    }
}

/// The whole volume as bytes, written field by field before it reaches a disk.
struct Image {
    bytes: Vec<u8>,
}

impl Image {
    fn new() -> Self {
        Self {
            bytes: vec![0u8; TOTAL_BLOCKS as usize * BLOCK_SIZE],
        }
    }

    fn write(&mut self, offset: usize, data: &[u8]) {
        self.bytes[offset..offset + data.len()].copy_from_slice(data);
    }

    fn write_val<T: Pod>(&mut self, offset: usize, value: &T) {
        self.write(offset, value.as_bytes());
    }

    fn read_val<T: Pod>(&self, offset: usize) -> T {
        T::from_bytes(&self.bytes[offset..offset + size_of::<T>()])
    }

    fn fill(&mut self, bid: u64, byte: u8) {
        let offset = bid as usize * BLOCK_SIZE;
        self.bytes[offset..offset + BLOCK_SIZE].fill(byte);
    }
}

const EXTENT_MAGIC: u16 = 0xf30a;
const EXTENT_ROOT_LEN: usize = 60;
const EXTENT_ENTRY_LEN: usize = 12;

/// Builds an extent node header.
fn extent_header(entries: u16, max: u16, depth: u16) -> [u8; EXTENT_ENTRY_LEN] {
    let mut bytes = [0u8; EXTENT_ENTRY_LEN];
    bytes[0..2].copy_from_slice(&EXTENT_MAGIC.to_le_bytes());
    bytes[2..4].copy_from_slice(&entries.to_le_bytes());
    bytes[4..6].copy_from_slice(&max.to_le_bytes());
    bytes[6..8].copy_from_slice(&depth.to_le_bytes());
    bytes
}

/// Builds one leaf entry mapping `len` logical blocks at `block` onto `start`.
fn extent_entry(block: u32, start: u64, len: u16) -> [u8; EXTENT_ENTRY_LEN] {
    let mut bytes = [0u8; EXTENT_ENTRY_LEN];
    bytes[0..4].copy_from_slice(&block.to_le_bytes());
    bytes[4..6].copy_from_slice(&len.to_le_bytes());
    bytes[6..8].copy_from_slice(&((start >> 32) as u16).to_le_bytes());
    bytes[8..12].copy_from_slice(&(start as u32).to_le_bytes());
    bytes
}

/// Builds one index entry pointing at the node in block `leaf`.
fn extent_index_entry(block: u32, leaf: u64) -> [u8; EXTENT_ENTRY_LEN] {
    let mut bytes = [0u8; EXTENT_ENTRY_LEN];
    bytes[0..4].copy_from_slice(&block.to_le_bytes());
    bytes[4..8].copy_from_slice(&(leaf as u32).to_le_bytes());
    bytes[8..10].copy_from_slice(&((leaf >> 32) as u16).to_le_bytes());
    bytes
}

/// Builds a leaf node small enough to live in an inode.
pub(super) fn extent_root(extents: &[(u32, u64, u16)]) -> [u8; EXTENT_ROOT_LEN] {
    let max = ((EXTENT_ROOT_LEN - EXTENT_ENTRY_LEN) / EXTENT_ENTRY_LEN) as u16;
    assert!(extents.len() <= max as usize);

    let mut root = [0u8; EXTENT_ROOT_LEN];
    root[..EXTENT_ENTRY_LEN].copy_from_slice(&extent_header(extents.len() as u16, max, 0));
    for (entry_idx, &(block, start, len)) in extents.iter().enumerate() {
        let offset = EXTENT_ENTRY_LEN * (entry_idx + 1);
        root[offset..offset + EXTENT_ENTRY_LEN].copy_from_slice(&extent_entry(block, start, len));
    }
    root
}

/// Builds an index node that lives in an inode and points at leaf blocks.
pub(super) fn extent_index_root(children: &[(u32, u64)]) -> [u8; EXTENT_ROOT_LEN] {
    let max = ((EXTENT_ROOT_LEN - EXTENT_ENTRY_LEN) / EXTENT_ENTRY_LEN) as u16;
    assert!(children.len() <= max as usize);

    let mut root = [0u8; EXTENT_ROOT_LEN];
    root[..EXTENT_ENTRY_LEN].copy_from_slice(&extent_header(children.len() as u16, max, 1));
    for (entry_idx, &(block, leaf)) in children.iter().enumerate() {
        let offset = EXTENT_ENTRY_LEN * (entry_idx + 1);
        root[offset..offset + EXTENT_ENTRY_LEN].copy_from_slice(&extent_index_entry(block, leaf));
    }
    root
}

/// Builds a leaf node that occupies a whole block.
pub(super) fn extent_leaf_block(extents: &[(u32, u64, u16)]) -> Vec<u8> {
    let max = ((BLOCK_SIZE - EXTENT_ENTRY_LEN) / EXTENT_ENTRY_LEN) as u16;
    let mut block = vec![0u8; BLOCK_SIZE];
    block[..EXTENT_ENTRY_LEN].copy_from_slice(&extent_header(extents.len() as u16, max, 0));
    for (entry_idx, &(logical, start, len)) in extents.iter().enumerate() {
        let offset = EXTENT_ENTRY_LEN * (entry_idx + 1);
        block[offset..offset + EXTENT_ENTRY_LEN]
            .copy_from_slice(&extent_entry(logical, start, len));
    }
    block
}

/// Builds one directory block holding the given entries, with the last record
/// padded out to the end of the block.
fn dir_block(entries: &[(&str, Ext4Ino, u8)]) -> Vec<u8> {
    let mut block = vec![0u8; BLOCK_SIZE];
    let mut offset = 0;

    for (entry_idx, &(name, ino, file_type)) in entries.iter().enumerate() {
        let rec_len = if entry_idx == entries.len() - 1 {
            BLOCK_SIZE - offset
        } else {
            (name.len() + 8).next_multiple_of(4)
        };

        block[offset..offset + 4].copy_from_slice(&ino.to_le_bytes());
        block[offset + 4..offset + 6].copy_from_slice(&(rec_len as u16).to_le_bytes());
        block[offset + 6] = name.len() as u8;
        block[offset + 7] = file_type;
        block[offset + 8..offset + 8 + name.len()].copy_from_slice(name.as_bytes());
        offset += rec_len;
    }

    block
}

const EXTENTS_FL: u32 = 1 << 19;

fn base_inode(mode: u16, size: u64, nblocks: u32) -> RawInode {
    let mut raw = RawInode::new_zeroed();
    raw.mode = mode;
    raw.link_count = 1;
    raw.size_lo = size as u32;
    raw.size_high = (size >> 32) as u32;
    raw.blocks_lo = nblocks * (BLOCK_SIZE / SECTOR_SIZE) as u32;
    raw.extra_isize = 32;
    raw.atime = 1_700_000_000;
    raw.ctime = 1_700_000_001;
    raw.mtime = 1_700_000_002;
    raw.crtime = 1_700_000_003;
    raw
}

fn file_inode(size: u64, nblocks: u32, block: [u8; EXTENT_ROOT_LEN]) -> RawInode {
    let mut raw = base_inode(0o100644, size, nblocks);
    raw.flags = EXTENTS_FL;
    raw.block = block;
    raw
}

fn dir_inode(size: u64, nblocks: u32, block: [u8; EXTENT_ROOT_LEN]) -> RawInode {
    let mut raw = base_inode(0o040755, size, nblocks);
    raw.link_count = 2;
    raw.flags = EXTENTS_FL;
    raw.block = block;
    raw
}

fn symlink_inode(target: &str) -> RawInode {
    let mut raw = base_inode(0o120777, target.len() as u64, 0);
    raw.block[..target.len()].copy_from_slice(target.as_bytes());
    raw
}

fn device_inode(major: u32, minor: u32) -> RawInode {
    let mut raw = base_inode(0o020666, 0, 0);
    raw.block[..4].copy_from_slice(&(((major & 0xff) << 8) | (minor & 0xff)).to_le_bytes());
    raw
}

/// The on-disk group descriptor, as the fixture writes it.
///
/// `block_group::RawGroupDesc` is the reader's view and is private to that
/// module; writing through a separate structure is what makes a test fail when
/// the two disagree.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
struct RawGroupDescImage {
    block_bitmap_lo: u32,
    inode_bitmap_lo: u32,
    inode_table_lo: u32,
    free_blocks_count_lo: u16,
    free_inodes_count_lo: u16,
    used_dirs_count_lo: u16,
    flags: u16,
    exclude_bitmap_lo: u32,
    block_bitmap_csum_lo: u16,
    inode_bitmap_csum_lo: u16,
    itable_unused_lo: u16,
    checksum: u16,
    block_bitmap_hi: u32,
    inode_bitmap_hi: u32,
    inode_table_hi: u32,
    free_blocks_count_hi: u16,
    free_inodes_count_hi: u16,
    used_dirs_count_hi: u16,
    itable_unused_hi: u16,
    exclude_bitmap_hi: u32,
    block_bitmap_csum_hi: u16,
    inode_bitmap_csum_hi: u16,
    reserved: u32,
}

/// Builds a volume and returns the device holding it, for tests outside this
/// module that need an ext4 volume but not its contents.
pub(in crate::fs) fn volume_disk() -> Arc<dyn BlockDevice> {
    Ext4VolumeBuilder::new().build().disk
}
