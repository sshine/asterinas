// SPDX-License-Identifier: MPL-2.0

//! Ext4 superblock: on-disk layout, in-memory representation, and mount validation.
//!
//! The superblock describes the layout and global state of an ext4 volume and
//! sits at a fixed offset of 1024 bytes from the start of the device.
//!
//! # Types
//!
//! - `RawSuperBlock` — the `#[repr(C)]` on-disk layout.
//! - `SuperBlock` — a validated, in-memory representation.
//!
//! # Telling ext2 and ext4 apart
//!
//! Both formats share the magic number `0xEF53`; the feature words are what
//! separate them. An ext4 volume made by `mkfs.ext4` carries at least the
//! `EXTENTS` incompatible feature, which ext2 does not implement and rejects.
//! Conversely this driver requires `EXTENTS`, so an ext2 volume offered to it
//! is rejected in turn and the root filesystem candidate list can try both in
//! either order.

use ostd::const_assert;

use super::{features, features::FeatureIncompat, prelude::*};

/// The ext2/ext3/ext4 magic number.
pub(super) const MAGIC_NUM: u16 = 0xef53;

/// The main superblock is located at byte 1024 from the beginning of the device.
pub(super) const SUPER_BLOCK_OFFSET: usize = 1024;

const SUPER_BLOCK_SIZE: usize = 1024;

/// The `log_block_size` shift that yields a 4096-byte block.
const LOG_BLOCK_SIZE_4K: u32 = 2;

/// Validated, Rust-typed in-memory representation of the ext4 superblock.
///
/// Feature words stay raw rather than becoming bitflags because the mount
/// policy turns on bits this driver does not know, and a truncating conversion
/// would discard exactly those.
#[derive(Clone, Copy, Debug)]
pub(super) struct SuperBlock {
    /// Size of an on-disk inode record in bytes.
    pub inode_size: usize,
    /// Size of an on-disk block group descriptor in bytes, 32 or 64.
    pub desc_size: usize,
    pub blocks_count: u64,
    pub free_blocks_count: u64,
    pub reserved_blocks_count: u64,
    pub inodes_count: u32,
    pub free_inodes_count: u32,
    pub first_data_block: u32,
    pub blocks_per_group: u32,
    pub inodes_per_group: u32,
    pub feature_compat: u32,
    pub feature_incompat: u32,
    pub feature_ro_compat: u32,
}

impl SuperBlock {
    /// Returns the number of block groups on the volume.
    pub(super) fn nr_block_groups(&self) -> usize {
        let data_blocks = self.blocks_count - self.first_data_block as u64;
        data_blocks.div_ceil(self.blocks_per_group as u64) as usize
    }

    /// Returns the block holding the first group descriptor.
    ///
    /// The descriptor table follows the block that holds the superblock, which
    /// for 4096-byte blocks is block 0.
    pub(super) fn group_descriptors_bid(&self) -> u64 {
        self.first_data_block as u64 + 1
    }

    /// Returns the number of blocks one group's inode table occupies.
    pub(super) fn inode_table_blocks(&self) -> u64 {
        (self.inodes_per_group as u64 * self.inode_size as u64) / BLOCK_SIZE as u64
    }

    /// Splits an inode number into its group index and its index within that
    /// group's inode table.
    pub(super) fn locate_inode(&self, ino: Ext4Ino) -> Result<(usize, u32)> {
        if ino == 0 || ino > self.inodes_count {
            return_errno_with_message!(Errno::ENOENT, "inode number out of range");
        }

        let group_idx = ((ino - 1) / self.inodes_per_group) as usize;
        let index_in_group = (ino - 1) % self.inodes_per_group;
        Ok((group_idx, index_in_group))
    }
}

impl TryFrom<RawSuperBlock> for SuperBlock {
    type Error = Error;

    fn try_from(sb: RawSuperBlock) -> Result<Self> {
        if sb.magic != MAGIC_NUM {
            return_errno_with_message!(Errno::EINVAL, "bad ext4 magic number");
        }

        // Only 4096-byte blocks are supported, for two reasons that have to
        // hold together: `aster_block::BLOCK_SIZE` equals the page size and the
        // page cache keeps one page per block, and `IdBitmap` indexes with a
        // `u16`, which a larger block size would overflow because it puts
        // 65536 blocks in a group.
        if sb.log_block_size != LOG_BLOCK_SIZE_4K {
            return_errno_with_message!(Errno::EINVAL, "unsupported block size");
        }

        features::check_incompat(sb.feature_incompat)?;

        // Without extents the volume is an ext2 or ext3 one, which the ext2
        // driver handles.
        if (sb.feature_incompat & FeatureIncompat::EXTENTS.bits()) == 0 {
            return_errno_with_message!(Errno::EINVAL, "no extent support, not an ext4 volume");
        }

        // A 64-bit volume states its descriptor size; a 32-bit one leaves the
        // field zero and uses the historical 32 bytes.
        let desc_size = if (sb.feature_incompat & FeatureIncompat::SIZE_64BIT.bits()) != 0 {
            let desc_size = sb.desc_size as usize;
            if desc_size < 64 || !desc_size.is_power_of_two() || desc_size > BLOCK_SIZE {
                return_errno_with_message!(Errno::EINVAL, "invalid group descriptor size");
            }
            desc_size
        } else {
            32
        };

        let inode_size = sb.inode_size as usize;
        if inode_size < 128 || !inode_size.is_power_of_two() || inode_size > BLOCK_SIZE {
            return_errno_with_message!(Errno::EINVAL, "invalid inode size");
        }

        if sb.blocks_per_group == 0 || sb.inodes_per_group == 0 {
            return_errno_with_message!(Errno::EINVAL, "invalid group sizes");
        }

        // A group's allocation bitmap is one block, so a group can hold no more
        // blocks or inodes than that block has bits.
        let max_per_group = (BLOCK_SIZE * 8) as u32;
        if sb.blocks_per_group > max_per_group || sb.inodes_per_group > max_per_group {
            return_errno_with_message!(Errno::EINVAL, "group is larger than its bitmap");
        }

        // One inode must never straddle a block boundary, which holds for every
        // volume `mkfs.ext4` produces and which the inode reader relies on.
        if !(sb.inodes_per_group as usize * inode_size).is_multiple_of(BLOCK_SIZE) {
            return_errno_with_message!(
                Errno::EINVAL,
                "inode table is not a whole number of blocks"
            );
        }

        let blocks_count = ((sb.blocks_count_hi as u64) << 32) | sb.blocks_count_lo as u64;
        if blocks_count <= sb.first_data_block as u64 {
            return_errno_with_message!(Errno::EINVAL, "invalid blocks count");
        }

        if sb.inodes_count == 0 {
            return_errno_with_message!(Errno::EINVAL, "volume has no inodes");
        }

        Ok(Self {
            inode_size,
            desc_size,
            blocks_count,
            free_blocks_count: ((sb.free_blocks_count_hi as u64) << 32)
                | sb.free_blocks_count_lo as u64,
            reserved_blocks_count: ((sb.reserved_blocks_count_hi as u64) << 32)
                | sb.reserved_blocks_count_lo as u64,
            inodes_count: sb.inodes_count,
            free_inodes_count: sb.free_inodes_count,
            first_data_block: sb.first_data_block,
            blocks_per_group: sb.blocks_per_group,
            inodes_per_group: sb.inodes_per_group,
            feature_compat: sb.feature_compat,
            feature_incompat: sb.feature_incompat,
            feature_ro_compat: sb.feature_ro_compat,
        })
    }
}

const_assert!(size_of::<RawSuperBlock>() == SUPER_BLOCK_SIZE);

/// The on-disk superblock structure.
///
/// Must be exactly 1024 bytes. Convert to `SuperBlock` via `TryFrom` for the
/// validated in-memory representation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct RawSuperBlock {
    pub inodes_count: u32,
    pub blocks_count_lo: u32,
    pub reserved_blocks_count_lo: u32,
    pub free_blocks_count_lo: u32,
    pub free_inodes_count: u32,
    pub first_data_block: u32,
    /// The number to left-shift 1024 by to obtain the block size.
    pub log_block_size: u32,
    pub log_cluster_size: u32,
    pub blocks_per_group: u32,
    pub clusters_per_group: u32,
    pub inodes_per_group: u32,
    pub mtime: u32,
    pub wtime: u32,
    pub mnt_count: u16,
    pub max_mnt_count: u16,
    pub magic: u16,
    pub state: u16,
    pub errors: u16,
    pub min_rev_level: u16,
    pub last_check_time: u32,
    pub check_interval: u32,
    pub creator_os: u32,
    pub rev_level: u32,
    pub def_resuid: u16,
    pub def_resgid: u16,
    pub first_ino: u32,
    pub inode_size: u16,
    pub block_group_nr: u16,
    pub feature_compat: u32,
    pub feature_incompat: u32,
    pub feature_ro_compat: u32,
    pub uuid: [u8; 16],
    pub volume_name: [u8; 16],
    pub last_mounted: [u8; 64],
    pub algorithm_usage_bitmap: u32,
    pub prealloc_blocks: u8,
    pub prealloc_dir_blocks: u8,
    pub reserved_gdt_blocks: u16,
    pub journal_uuid: [u8; 16],
    pub journal_inum: u32,
    pub journal_dev: u32,
    pub last_orphan: u32,
    pub hash_seed: [u32; 4],
    pub def_hash_version: u8,
    pub jnl_backup_type: u8,
    /// Size of a block group descriptor, meaningful only with `SIZE_64BIT`.
    pub desc_size: u16,
    pub default_mount_opts: u32,
    pub first_meta_bg: u32,
    pub mkfs_time: u32,
    pub jnl_blocks: [u32; 17],
    pub blocks_count_hi: u32,
    pub reserved_blocks_count_hi: u32,
    pub free_blocks_count_hi: u32,
    pub min_extra_isize: u16,
    pub want_extra_isize: u16,
    pub flags: u32,
    pub raid_stride: u16,
    pub mmp_interval: u16,
    pub mmp_block: u64,
    pub raid_stripe_width: u32,
    pub log_groups_per_flex: u8,
    pub checksum_type: u8,
    pub reserved_pad: u16,
    pub kbytes_written: u64,
    pub snapshot_inum: u32,
    pub snapshot_id: u32,
    pub snapshot_r_blocks_count: u64,
    pub snapshot_list: u32,
    pub error_count: u32,
    pub first_error_time: u32,
    pub first_error_ino: u32,
    pub first_error_block: u64,
    pub first_error_func: [u8; 32],
    pub first_error_line: u32,
    pub last_error_time: u32,
    pub last_error_ino: u32,
    pub last_error_line: u32,
    pub last_error_block: u64,
    pub last_error_func: [u8; 32],
    pub mount_opts: [u8; 64],
    pub usr_quota_inum: u32,
    pub grp_quota_inum: u32,
    pub overhead_blocks: u32,
    pub backup_bgs: [u32; 2],
    pub encrypt_algos: [u8; 4],
    pub encrypt_pw_salt: [u8; 16],
    pub lpf_ino: u32,
    pub prj_quota_inum: u32,
    pub checksum_seed: u32,
    pub wtime_hi: u8,
    pub mtime_hi: u8,
    pub mkfs_time_hi: u8,
    pub last_check_hi: u8,
    pub first_error_time_hi: u8,
    pub last_error_time_hi: u8,
    pub first_error_errcode: u8,
    pub last_error_errcode: u8,
    pub encoding: u16,
    pub encoding_flags: u16,
    pub orphan_file_inum: u32,
    pub reserved: [u32; 94],
    pub checksum: u32,
}

#[cfg(ktest)]
mod test {
    use core::mem::offset_of;

    use ostd::prelude::*;

    use super::*;
    use crate::fs::fs_impls::ext4::test_utils::{assert_errno, make_valid_raw_super_block};

    /// Offsets taken from a volume made by `mkfs.ext4` and read back with
    /// `dumpe2fs`. A field inserted at the wrong place keeps the total size
    /// correct while silently shifting everything after it.
    #[ktest]
    fn raw_super_block_field_offsets_match_the_on_disk_layout() {
        assert_eq!(offset_of!(RawSuperBlock, log_block_size), 0x18);
        assert_eq!(offset_of!(RawSuperBlock, blocks_per_group), 0x20);
        assert_eq!(offset_of!(RawSuperBlock, inodes_per_group), 0x28);
        assert_eq!(offset_of!(RawSuperBlock, magic), 0x38);
        assert_eq!(offset_of!(RawSuperBlock, rev_level), 0x4c);
        assert_eq!(offset_of!(RawSuperBlock, first_ino), 0x54);
        assert_eq!(offset_of!(RawSuperBlock, inode_size), 0x58);
        assert_eq!(offset_of!(RawSuperBlock, feature_compat), 0x5c);
        assert_eq!(offset_of!(RawSuperBlock, feature_incompat), 0x60);
        assert_eq!(offset_of!(RawSuperBlock, feature_ro_compat), 0x64);
        assert_eq!(offset_of!(RawSuperBlock, desc_size), 0xfe);
        assert_eq!(offset_of!(RawSuperBlock, blocks_count_hi), 0x150);
        assert_eq!(offset_of!(RawSuperBlock, want_extra_isize), 0x15e);
        assert_eq!(offset_of!(RawSuperBlock, checksum), 0x3fc);
    }

    #[ktest]
    fn a_valid_super_block_parses() {
        let sb = SuperBlock::try_from(make_valid_raw_super_block()).expect("should parse");

        assert_eq!(sb.inode_size, 256);
        assert_eq!(sb.desc_size, 64);
        assert_eq!(sb.blocks_per_group, 32768);
        assert_ne!(sb.feature_incompat & FeatureIncompat::EXTENTS.bits(), 0);
    }

    /// A volume whose last group is short still has that group counted, which
    /// a truncating division would drop along with every inode in it.
    #[ktest]
    fn a_partly_filled_last_group_still_counts() {
        let mut raw = make_valid_raw_super_block();
        raw.blocks_per_group = 128;
        raw.blocks_count_lo = 128 * 3 + 1;

        let sb = SuperBlock::try_from(raw).expect("should parse");

        assert_eq!(sb.nr_block_groups(), 4);
    }

    #[ktest]
    fn inode_numbers_split_into_group_and_index() {
        let mut raw = make_valid_raw_super_block();
        raw.inodes_per_group = 16;
        raw.inodes_count = 32;

        let sb = SuperBlock::try_from(raw).expect("should parse");

        assert_eq!(sb.locate_inode(1).expect("inode 1 exists"), (0, 0));
        assert_eq!(sb.locate_inode(16).expect("inode 16 exists"), (0, 15));
        assert_eq!(sb.locate_inode(17).expect("inode 17 exists"), (1, 0));
        assert_errno(sb.locate_inode(0), Errno::ENOENT);
        assert_errno(sb.locate_inode(33), Errno::ENOENT);
    }

    #[ktest]
    fn a_bad_magic_number_is_rejected() {
        let mut raw = make_valid_raw_super_block();
        raw.magic = 0;

        assert_errno(SuperBlock::try_from(raw), Errno::EINVAL);
    }

    #[ktest]
    fn an_ext2_super_block_is_rejected() {
        let mut raw = make_valid_raw_super_block();
        raw.feature_incompat = FeatureIncompat::FILETYPE.bits();
        raw.desc_size = 0;

        assert_errno(SuperBlock::try_from(raw), Errno::EINVAL);
    }

    #[ktest]
    fn an_unknown_incompat_feature_is_rejected() {
        let mut raw = make_valid_raw_super_block();
        raw.feature_incompat |= FeatureIncompat::INLINE_DATA.bits();

        assert_errno(SuperBlock::try_from(raw), Errno::EINVAL);
    }

    #[ktest]
    fn a_group_larger_than_its_bitmap_is_rejected() {
        let mut raw = make_valid_raw_super_block();
        raw.blocks_per_group = (BLOCK_SIZE * 8) as u32 + 1;

        assert_errno(SuperBlock::try_from(raw), Errno::EINVAL);
    }
}
