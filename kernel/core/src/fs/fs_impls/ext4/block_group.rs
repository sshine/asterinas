// SPDX-License-Identifier: MPL-2.0

//! Block groups: the descriptor table and the allocation bitmaps.
//!
//! An ext4 volume is divided into equally sized block groups, and one
//! descriptor per group records where that group's bitmaps and inode table
//! live and how much of it is free. The whole descriptor table is read at
//! mount, because every inode lookup needs the inode table address of the
//! group holding it.
//!
//! # Descriptor size
//!
//! A descriptor is 32 bytes on a volume without the `64BIT` feature and 64
//! bytes with it, where the added half carries the upper 32 bits of each block
//! address. The superblock states which, so both are read through one 64-byte
//! structure whose upper half stays zero for the short form.
//!
//! # flex_bg
//!
//! With `FLEX_BG` the metadata of several consecutive groups is packed into
//! one of them, so a descriptor's bitmaps and inode table routinely lie
//! outside the group they describe. Addresses are therefore validated against
//! the volume, never against the group.
//!
//! # Bitmaps
//!
//! Each group has one block of block bitmap and one of inode bitmap, which is
//! what bounds a group at 32768 blocks for a 4 KiB block size. They are read
//! on first use, so a read-only mount never touches them, and held in memory
//! until the group is synced.
//!
//! An uninitialized bitmap, which `uninit_bg` and `metadata_csum` allow, holds
//! arbitrary bytes rather than allocation state. Neither feature can be
//! present on a volume this driver mounts read-write, so the case cannot
//! arise; [`load_all`] checks for it anyway, because reading garbage as a
//! bitmap would hand out blocks that are already in use.

use ostd::const_assert;

use super::{features::FeatureRoCompat, fs::Ext4, prelude::*, super_block::SuperBlock};
use crate::fs::utils::IdBitmap;

/// One block group's descriptor, with the 32- and 64-bit halves joined.
#[derive(Clone, Copy, Debug)]
pub(super) struct GroupDesc {
    pub block_bitmap_bid: Ext4Bid,
    pub inode_bitmap_bid: Ext4Bid,
    pub inode_table_bid: Ext4Bid,
    pub free_blocks_count: u32,
    pub free_inodes_count: u32,
    pub used_dirs_count: u32,
    /// Number of inodes at the end of this group's table that were never
    /// written, and whose contents are therefore arbitrary.
    pub itable_unused: u32,
    pub flags: u16,
}

/// `EXT4_BG_INODE_UNINIT`: the inode bitmap was never written.
const BG_INODE_UNINIT: u16 = 0x1;
/// `EXT4_BG_BLOCK_UNINIT`: the block bitmap was never written.
const BG_BLOCK_UNINIT: u16 = 0x2;

impl GroupDesc {
    fn from_raw(raw: &RawGroupDesc) -> Self {
        Self {
            block_bitmap_bid: join(raw.block_bitmap_lo, raw.block_bitmap_hi),
            inode_bitmap_bid: join(raw.inode_bitmap_lo, raw.inode_bitmap_hi),
            inode_table_bid: join(raw.inode_table_lo, raw.inode_table_hi),
            free_blocks_count: join16(raw.free_blocks_count_lo, raw.free_blocks_count_hi),
            free_inodes_count: join16(raw.free_inodes_count_lo, raw.free_inodes_count_hi),
            used_dirs_count: join16(raw.used_dirs_count_lo, raw.used_dirs_count_hi),
            itable_unused: join16(raw.itable_unused_lo, raw.itable_unused_hi),
            flags: raw.flags,
        }
    }

    /// Overwrites the fields this driver owns in an on-disk descriptor.
    ///
    /// As with an inode record, the descriptor read from the device is patched
    /// rather than rebuilt, so that the checksum fields this driver does not
    /// maintain keep whatever they held.
    fn write_into(&self, raw: &mut RawGroupDesc) {
        raw.free_blocks_count_lo = self.free_blocks_count as u16;
        raw.free_blocks_count_hi = (self.free_blocks_count >> 16) as u16;
        raw.free_inodes_count_lo = self.free_inodes_count as u16;
        raw.free_inodes_count_hi = (self.free_inodes_count >> 16) as u16;
        raw.used_dirs_count_lo = self.used_dirs_count as u16;
        raw.used_dirs_count_hi = (self.used_dirs_count >> 16) as u16;
        raw.itable_unused_lo = self.itable_unused as u16;
        raw.itable_unused_hi = (self.itable_unused >> 16) as u16;
        raw.flags = self.flags;
    }

    /// Returns whether the inode at `index_in_group` has ever been written.
    ///
    /// The tail of an inode table that was never initialized holds arbitrary
    /// bytes, which can decode as a plausible inode. The count of such inodes
    /// is only maintained when the volume checksums its group descriptors.
    pub(super) fn is_inode_initialized(&self, index_in_group: u32, sb: &SuperBlock) -> bool {
        let tracked = FeatureRoCompat::GDT_CSUM.bits() | FeatureRoCompat::METADATA_CSUM.bits();
        if (sb.feature_ro_compat & tracked) == 0 {
            return true;
        }

        index_in_group < sb.inodes_per_group.saturating_sub(self.itable_unused)
    }
}

fn join(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | lo as u64
}

fn join16(lo: u16, hi: u16) -> u32 {
    ((hi as u32) << 16) | lo as u32
}

/// One block group, with its descriptor and the bitmaps it allocates from.
#[derive(Debug)]
pub(super) struct BlockGroup {
    group_idx: usize,
    /// First block of this group, filesystem-wide.
    first_block: Ext4Bid,
    /// Blocks in this group, which is short of a full group in the last one.
    nr_blocks: u32,
    /// Inodes in this group, likewise.
    nr_inodes: u32,
    state: RwMutex<GroupState>,
}

#[derive(Debug)]
struct GroupState {
    desc: Dirty<GroupDesc>,
    /// Read on first use, so a read-only mount never loads one.
    block_bitmap: Option<Dirty<IdBitmap>>,
    inode_bitmap: Option<Dirty<IdBitmap>>,
}

impl BlockGroup {
    pub(super) fn desc(&self) -> GroupDesc {
        *self.state.read().desc
    }

    pub(super) fn first_block(&self) -> Ext4Bid {
        self.first_block
    }

    pub(super) fn nr_blocks(&self) -> u32 {
        self.nr_blocks
    }

    /// Allocates up to `count` consecutive blocks from this group.
    ///
    /// Returns `None` when the group has no run of free blocks at all, which
    /// is the caller's signal to try the next one. A shorter run than asked
    /// for is a success, because a file is free to use several extents.
    pub(super) fn alloc_blocks(&self, fs: &Ext4, count: u32) -> Result<Option<Range<Ext4Bid>>> {
        let mut state = self.state.write();
        if state.desc.free_blocks_count == 0 {
            return Ok(None);
        }

        self.load_block_bitmap(fs, &mut state)?;
        let bitmap = state
            .block_bitmap
            .as_mut()
            .expect("the block bitmap was just loaded");

        // Ask for less and less until the bitmap can satisfy the request, so
        // that a fragmented group still yields what it has.
        let mut wanted = count.min(self.nr_blocks).min(u16::MAX as u32) as u16;
        let allocated = loop {
            if wanted == 0 {
                return Ok(None);
            }
            if let Some(range) = bitmap.alloc_consecutive(wanted) {
                break range;
            }
            wanted /= 2;
        };

        let allocated_count = allocated.end - allocated.start;
        state.desc.free_blocks_count -= allocated_count as u32;

        let start = self.first_block + allocated.start as u64;
        Ok(Some(start..start + allocated_count as u64))
    }

    /// Frees a range of blocks that lies within this group.
    pub(super) fn free_blocks(&self, fs: &Ext4, range: Range<Ext4Bid>) -> Result<()> {
        debug_assert!(range.start >= self.first_block);
        debug_assert!(range.end <= self.first_block + self.nr_blocks as u64);

        let mut state = self.state.write();
        self.load_block_bitmap(fs, &mut state)?;
        let bitmap = state
            .block_bitmap
            .as_mut()
            .expect("the block bitmap was just loaded");

        let start = (range.start - self.first_block) as u16;
        let end = (range.end - self.first_block) as u16;
        bitmap.free_consecutive(start..end);
        state.desc.free_blocks_count += (end - start) as u32;
        Ok(())
    }

    /// Allocates one inode from this group, returning its index within it.
    pub(super) fn alloc_inode(&self, fs: &Ext4, is_dir: bool) -> Result<Option<u32>> {
        let mut state = self.state.write();
        if state.desc.free_inodes_count == 0 {
            return Ok(None);
        }

        self.load_inode_bitmap(fs, &mut state)?;
        let index = state
            .inode_bitmap
            .as_mut()
            .expect("the inode bitmap was just loaded")
            .alloc();
        let Some(index) = index else {
            return Ok(None);
        };

        state.desc.free_inodes_count -= 1;
        if is_dir {
            state.desc.used_dirs_count += 1;
        }
        Ok(Some(index as u32))
    }

    /// Frees one inode by its index within this group.
    pub(super) fn free_inode(&self, fs: &Ext4, index: u32, is_dir: bool) -> Result<()> {
        let mut state = self.state.write();
        self.load_inode_bitmap(fs, &mut state)?;
        let bitmap = state
            .inode_bitmap
            .as_mut()
            .expect("the inode bitmap was just loaded");
        if !bitmap.is_allocated(index as u16) {
            return_errno_with_message!(Errno::EUCLEAN, "freeing an inode that is already free");
        }

        bitmap.free(index as u16);
        state.desc.free_inodes_count += 1;
        if is_dir {
            state.desc.used_dirs_count = state.desc.used_dirs_count.saturating_sub(1);
        }
        Ok(())
    }

    /// Returns whether the block at `bid` is marked allocated.
    #[cfg(ktest)]
    pub(super) fn is_block_allocated(&self, fs: &Ext4, bid: Ext4Bid) -> Result<bool> {
        let mut state = self.state.write();
        self.load_block_bitmap(fs, &mut state)?;
        Ok(state
            .block_bitmap
            .as_ref()
            .expect("the block bitmap was just loaded")
            .is_allocated((bid - self.first_block) as u16))
    }

    /// Writes back this group's bitmaps and descriptor, if anything changed.
    pub(super) fn sync(&self, fs: &Ext4) -> Result<()> {
        let mut state = self.state.write();
        // The addresses are read before the bitmaps are borrowed, because both
        // live in the same guard.
        let (block_bitmap_bid, inode_bitmap_bid) =
            (state.desc.block_bitmap_bid, state.desc.inode_bitmap_bid);

        if let Some(bitmap) = state.block_bitmap.as_mut()
            && bitmap.is_dirty()
        {
            fs.write_metadata(Bid::new(block_bitmap_bid).to_offset(), bitmap.as_bytes())?;
            bitmap.clear_dirty();
        }
        if let Some(bitmap) = state.inode_bitmap.as_mut()
            && bitmap.is_dirty()
        {
            fs.write_metadata(Bid::new(inode_bitmap_bid).to_offset(), bitmap.as_bytes())?;
            bitmap.clear_dirty();
        }

        if state.desc.is_dirty() {
            fs.write_back_group_desc(self.group_idx, &state.desc)?;
            state.desc.clear_dirty();
        }
        Ok(())
    }

    fn load_block_bitmap(&self, fs: &Ext4, state: &mut GroupState) -> Result<()> {
        if state.block_bitmap.is_some() {
            return Ok(());
        }

        let bitmap = read_bitmap(fs, state.desc.block_bitmap_bid, self.nr_blocks)?;
        state.block_bitmap = Some(Dirty::new(bitmap));
        Ok(())
    }

    fn load_inode_bitmap(&self, fs: &Ext4, state: &mut GroupState) -> Result<()> {
        if state.inode_bitmap.is_some() {
            return Ok(());
        }

        let bitmap = read_bitmap(fs, state.desc.inode_bitmap_bid, self.nr_inodes)?;
        state.inode_bitmap = Some(Dirty::new(bitmap));
        Ok(())
    }
}

/// Reads one bitmap block and keeps its first `len` bits.
fn read_bitmap(fs: &Ext4, bid: Ext4Bid, len: u32) -> Result<IdBitmap> {
    let mut buf = vec![0u8; BLOCK_SIZE];
    fs.block_device()
        .read_bytes(Bid::new(bid).to_offset(), &mut buf)
        .map_err(|_| Error::with_message(Errno::EIO, "failed to read an allocation bitmap"))?;

    debug_assert!(len <= IdBitmap::capacity() as u32);
    Ok(IdBitmap::from_buf(buf.into_boxed_slice(), len as u16))
}

/// Reads and validates the descriptors of every block group on the volume.
pub(super) fn load_all(device: &dyn BlockDevice, sb: &SuperBlock) -> Result<Vec<BlockGroup>> {
    let nr_groups = sb.nr_block_groups();
    let mut table = vec![0u8; nr_groups * sb.desc_size];
    device
        .read_bytes(Bid::new(sb.group_descriptors_bid()).to_offset(), &mut table)
        .map_err(|_| Error::with_message(Errno::EIO, "failed to read group descriptor table"))?;

    let inode_table_blocks = sb.inode_table_blocks();
    let mut groups = Vec::with_capacity(nr_groups);
    for group_idx in 0..nr_groups {
        let on_disk = &table[group_idx * sb.desc_size..][..sb.desc_size];

        // The short form is the long form with its upper half absent, so it is
        // read into a zeroed buffer of the full size.
        let mut bytes = [0u8; size_of::<RawGroupDesc>()];
        let copied = sb.desc_size.min(bytes.len());
        bytes[..copied].copy_from_slice(&on_disk[..copied]);

        let desc = GroupDesc::from_raw(&RawGroupDesc::from_bytes(&bytes));
        let within_volume =
            |start: Ext4Bid, len: u64| start != 0 && start.saturating_add(len) <= sb.blocks_count;
        if !within_volume(desc.inode_table_bid, inode_table_blocks)
            || !within_volume(desc.block_bitmap_bid, 1)
            || !within_volume(desc.inode_bitmap_bid, 1)
        {
            return_errno_with_message!(Errno::EUCLEAN, "group metadata lies outside the volume");
        }

        let first_block =
            sb.first_data_block as u64 + group_idx as u64 * sb.blocks_per_group as u64;
        let nr_blocks = (sb.blocks_count - first_block).min(sb.blocks_per_group as u64) as u32;
        let nr_inodes =
            (sb.inodes_count - group_idx as u32 * sb.inodes_per_group).min(sb.inodes_per_group);

        if desc.free_blocks_count > nr_blocks || desc.free_inodes_count > nr_inodes {
            return_errno_with_message!(Errno::EUCLEAN, "group claims more free than it holds");
        }

        groups.push(BlockGroup {
            group_idx,
            first_block,
            nr_blocks,
            nr_inodes,
            state: RwMutex::new(GroupState {
                desc: Dirty::new(desc),
                block_bitmap: None,
                inode_bitmap: None,
            }),
        });
    }

    Ok(groups)
}

/// Returns whether any group's bitmaps were never written.
///
/// Such a group's bitmap block holds arbitrary bytes, and allocating from it
/// would hand out blocks that are already in use. Only `uninit_bg` and
/// `metadata_csum` produce one, and both force a read-only mount, so this is
/// a check on that reasoning rather than a case that should occur.
pub(super) fn has_uninitialized_bitmaps(groups: &[BlockGroup]) -> bool {
    groups
        .iter()
        .any(|group| (group.desc().flags & (BG_BLOCK_UNINIT | BG_INODE_UNINIT)) != 0)
}

/// Writes a descriptor's own fields into the table, keeping the rest.
pub(super) fn patch_raw_desc(table_bytes: &mut [u8], desc: &GroupDesc) {
    let mut raw = RawGroupDesc::from_first_bytes(table_bytes);
    desc.write_into(&mut raw);
    let len = table_bytes.len().min(size_of::<RawGroupDesc>());
    table_bytes[..len].copy_from_slice(&raw.as_bytes()[..len]);
}

/// The on-disk block group descriptor.
///
/// Must be exactly 64 bytes. A volume without the `64BIT` feature stores only
/// the first 32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct RawGroupDesc {
    pub block_bitmap_lo: u32,
    pub inode_bitmap_lo: u32,
    pub inode_table_lo: u32,
    pub free_blocks_count_lo: u16,
    pub free_inodes_count_lo: u16,
    pub used_dirs_count_lo: u16,
    pub flags: u16,
    pub exclude_bitmap_lo: u32,
    pub block_bitmap_csum_lo: u16,
    pub inode_bitmap_csum_lo: u16,
    pub itable_unused_lo: u16,
    pub checksum: u16,
    pub block_bitmap_hi: u32,
    pub inode_bitmap_hi: u32,
    pub inode_table_hi: u32,
    pub free_blocks_count_hi: u16,
    pub free_inodes_count_hi: u16,
    pub used_dirs_count_hi: u16,
    pub itable_unused_hi: u16,
    pub exclude_bitmap_hi: u32,
    pub block_bitmap_csum_hi: u16,
    pub inode_bitmap_csum_hi: u16,
    pub reserved: u32,
}

const_assert!(size_of::<RawGroupDesc>() == 64);
#[cfg(ktest)]
mod test {
    use core::mem::offset_of;

    use ostd::prelude::*;

    use super::*;
    use crate::fs::fs_impls::ext4::test_utils::{Ext4VolumeBuilder, assert_errno};

    /// Offsets taken from a volume made by `mkfs.ext4`. The upper half only
    /// exists on a 64-bit volume, so a field misplaced in it goes unnoticed
    /// until such a volume is mounted.
    #[ktest]
    fn raw_group_desc_field_offsets_match_the_on_disk_layout() {
        assert_eq!(offset_of!(RawGroupDesc, inode_table_lo), 0x8);
        assert_eq!(offset_of!(RawGroupDesc, flags), 0x12);
        assert_eq!(offset_of!(RawGroupDesc, itable_unused_lo), 0x1c);
        assert_eq!(offset_of!(RawGroupDesc, checksum), 0x1e);
        assert_eq!(offset_of!(RawGroupDesc, inode_table_hi), 0x28);
        assert_eq!(offset_of!(RawGroupDesc, itable_unused_hi), 0x32);
    }

    #[ktest]
    fn descriptors_of_every_group_are_loaded() {
        let volume = Ext4VolumeBuilder::new().build();

        let groups = load_all(volume.disk.as_ref(), &volume.super_block).expect("should load");

        assert_eq!(groups.len(), volume.super_block.nr_block_groups());
        for (group_idx, group) in groups.iter().enumerate() {
            assert_eq!(
                group.desc().inode_table_bid,
                volume.inode_table_bid(group_idx)
            );
        }
    }

    /// A 32-byte descriptor leaves the upper half of every address unread, so
    /// the same table must decode to the same addresses either way.
    #[ktest]
    fn short_descriptors_decode_like_long_ones() {
        let long = Ext4VolumeBuilder::new().build();
        let short = Ext4VolumeBuilder::new().desc_size(32).build();

        let long_groups = load_all(long.disk.as_ref(), &long.super_block).expect("should load");
        let short_groups = load_all(short.disk.as_ref(), &short.super_block).expect("should load");

        assert_eq!(long_groups.len(), short_groups.len());
        for (long_group, short_group) in long_groups.iter().zip(short_groups.iter()) {
            assert_eq!(
                long_group.desc().inode_table_bid,
                short_group.desc().inode_table_bid
            );
        }
    }

    /// A block handed out by a write must be marked used, or the next write
    /// is handed the same one.
    #[ktest]
    fn an_allocated_block_is_marked_used_in_the_bitmap() {
        let volume = Ext4VolumeBuilder::new().build();
        let fs = crate::fs::fs_impls::ext4::fs::open(
            volume.disk.clone(),
            crate::fs::vfs::file_system::FsFlags::empty(),
            None,
        )
        .expect("should mount");
        let ext4 = fs
            .downcast_ref::<Ext4>()
            .expect("should be an ext4 volume")
            .root_inode()
            .expect("root should be readable")
            .fs()
            .expect("the filesystem should outlive its root");

        let run = ext4.alloc_blocks(0, 1).expect("should allocate");
        let group = ext4.group_of(run.start).expect("the block is in a group");

        assert!(
            group
                .is_block_allocated(&ext4, run.start)
                .expect("should read the bitmap")
        );
    }

    #[ktest]
    fn an_inode_table_outside_the_volume_is_rejected() {
        let volume = Ext4VolumeBuilder::new()
            .corrupt_inode_table_bid(0, u32::MAX as u64)
            .build();

        assert_errno(
            load_all(volume.disk.as_ref(), &volume.super_block),
            Errno::EUCLEAN,
        );
    }
}
