// SPDX-License-Identifier: MPL-2.0

//! The block group descriptor table.
//!
//! An ext4 volume is divided into equally sized block groups, and one
//! descriptor per group records where that group's bitmaps and inode table
//! live. The whole table is read once at mount and kept in memory, because
//! every inode lookup needs the inode table address of the group holding it.
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

use ostd::const_assert;

use super::{features::FeatureRoCompat, prelude::*, super_block::SuperBlock};

/// One block group's descriptor, with the 32- and 64-bit halves joined.
///
/// Only what the read path consults is decoded. The free counts and the
/// bitmap contents are the allocator's business and arrive with it.
#[derive(Clone, Copy, Debug)]
pub(super) struct GroupDesc {
    pub block_bitmap_bid: Ext4Bid,
    pub inode_bitmap_bid: Ext4Bid,
    pub inode_table_bid: Ext4Bid,
    /// Number of inodes at the end of this group's table that were never
    /// written, and whose contents are therefore arbitrary.
    pub itable_unused: u32,
}

impl GroupDesc {
    fn from_raw(raw: &RawGroupDesc) -> Self {
        Self {
            block_bitmap_bid: join(raw.block_bitmap_lo, raw.block_bitmap_hi),
            inode_bitmap_bid: join(raw.inode_bitmap_lo, raw.inode_bitmap_hi),
            inode_table_bid: join(raw.inode_table_lo, raw.inode_table_hi),
            itable_unused: join16(raw.itable_unused_lo, raw.itable_unused_hi),
        }
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

/// Reads and validates the descriptors of every block group on the volume.
pub(super) fn load_all(device: &dyn BlockDevice, sb: &SuperBlock) -> Result<Vec<GroupDesc>> {
    let nr_groups = sb.nr_block_groups();
    let mut table = vec![0u8; nr_groups * sb.desc_size];
    device
        .read_bytes(Bid::new(sb.group_descriptors_bid()).to_offset(), &mut table)
        .map_err(|_| Error::with_message(Errno::EIO, "failed to read group descriptor table"))?;

    let inode_table_blocks = sb.inode_table_blocks();
    let mut descs = Vec::with_capacity(nr_groups);
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

        descs.push(desc);
    }

    Ok(descs)
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

        let descs = load_all(volume.disk.as_ref(), &volume.super_block).expect("should load");

        assert_eq!(descs.len(), volume.super_block.nr_block_groups());
        for (group_idx, desc) in descs.iter().enumerate() {
            assert_eq!(desc.inode_table_bid, volume.inode_table_bid(group_idx));
        }
    }

    /// A 32-byte descriptor leaves the upper half of every address unread, so
    /// the same table must decode to the same addresses either way.
    #[ktest]
    fn short_descriptors_decode_like_long_ones() {
        let long = Ext4VolumeBuilder::new().build();
        let short = Ext4VolumeBuilder::new().desc_size(32).build();

        let long_descs = load_all(long.disk.as_ref(), &long.super_block).expect("should load");
        let short_descs = load_all(short.disk.as_ref(), &short.super_block).expect("should load");

        assert_eq!(long_descs.len(), short_descs.len());
        for (long_desc, short_desc) in long_descs.iter().zip(short_descs.iter()) {
            assert_eq!(long_desc.inode_table_bid, short_desc.inode_table_bid);
        }
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
