// SPDX-License-Identifier: MPL-2.0

//! Ext4 feature words.
//!
//! The superblock carries three of them, and each sets a different obligation
//! on a driver. An incompatible feature changes the on-disk layout, so a
//! driver that does not implement it must refuse the volume. A
//! read-only-compatible feature constrains writing alone, so a driver that
//! cannot maintain it may still read the volume. A compatible feature may be
//! ignored outright.
//!
//! Unknown bits inherit the same obligations, which is why the masks below
//! list what is supported rather than what is rejected.

use crate::prelude::*;

bitflags! {
    /// Features a driver may ignore.
    pub(super) struct FeatureCompat: u32 {
        const DIR_PREALLOC = 1 << 0;
        const IMAGIC_INODES = 1 << 1;
        const HAS_JOURNAL = 1 << 2;
        const EXT_ATTR = 1 << 3;
        const RESIZE_INODE = 1 << 4;
        const DIR_INDEX = 1 << 5;
        const SPARSE_SUPER2 = 1 << 9;
        const FAST_COMMIT = 1 << 10;
        const STABLE_INODES = 1 << 11;
        const ORPHAN_FILE = 1 << 12;
    }
}

bitflags! {
    /// Features that change the on-disk layout.
    pub(super) struct FeatureIncompat: u32 {
        const COMPRESSION = 1 << 0;
        const FILETYPE = 1 << 1;
        const RECOVER = 1 << 2;
        const JOURNAL_DEV = 1 << 3;
        const META_BG = 1 << 4;
        const EXTENTS = 1 << 6;
        const SIZE_64BIT = 1 << 7;
        const MMP = 1 << 8;
        const FLEX_BG = 1 << 9;
        const EA_INODE = 1 << 10;
        const DIRDATA = 1 << 12;
        const CSUM_SEED = 1 << 13;
        const LARGEDIR = 1 << 14;
        const INLINE_DATA = 1 << 15;
        const ENCRYPT = 1 << 16;
        const CASEFOLD = 1 << 17;
    }
}

bitflags! {
    /// Features that constrain writing only.
    pub(super) struct FeatureRoCompat: u32 {
        const SPARSE_SUPER = 1 << 0;
        const LARGE_FILE = 1 << 1;
        const BTREE_DIR = 1 << 2;
        const HUGE_FILE = 1 << 3;
        const GDT_CSUM = 1 << 4;
        const DIR_NLINK = 1 << 5;
        const EXTRA_ISIZE = 1 << 6;
        const HAS_SNAPSHOT = 1 << 7;
        const QUOTA = 1 << 8;
        const BIGALLOC = 1 << 9;
        const METADATA_CSUM = 1 << 10;
        const REPLICA = 1 << 11;
        const READONLY = 1 << 12;
        const PROJECT = 1 << 13;
        const SHARED_BLOCKS = 1 << 14;
        const VERITY = 1 << 15;
        const ORPHAN_PRESENT = 1 << 16;
    }
}
