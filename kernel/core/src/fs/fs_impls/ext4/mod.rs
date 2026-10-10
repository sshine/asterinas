// SPDX-License-Identifier: MPL-2.0

//! Ext4 filesystem implementation.
//!
//! This module is the entry point for ext4 support in Asterinas. A caller
//! registers `Ext4` as a filesystem type via `init`, after which the VFS can
//! mount ext4 volumes and read them through the standard filesystem trait
//! interfaces. Buffered I/O is delegated to the `PageCache` subsystem; this
//! module does not cache block data itself.
//!
//! Ext4 is the default Linux filesystem of the last two decades and the
//! on-disk format that ordinary Linux tooling produces. It descends from ext2
//! and ext3 and shares their magic number, but replaces the block-pointer map
//! with extent trees and adds journalling, 64-bit block addresses, large
//! inodes with nanosecond timestamps, hashed directories, and checksums
//! throughout.
//!
//! # What this iteration covers
//!
//! Mounting and reading: extent-mapped files, linear and hash-indexed
//! directories, symbolic links, and special files. Volumes mount read-only,
//! and every operation that would change one reports `EROFS`. Journalling,
//! metadata checksum verification, extended attributes, inline data and
//! encryption are not implemented; `features` describes which of them refuse a
//! volume and which only force it read-only.
//!
//! # On-disk layout
//!
//! An ext4 volume is divided into fixed-size block groups, each with a block
//! bitmap, an inode bitmap, an inode table, and data blocks. The superblock
//! sits 1024 bytes into the volume and the block group descriptor table
//! follows it. `flex_bg` lets several groups keep their metadata together in
//! one of them, so a group's metadata need not lie inside it.
//!
//! Files and directories are inodes, and an inode maps its logical blocks onto
//! the device through a tree of extents rooted in the 60-byte `i_block` area.
//!
//! # Module structure
//!
//! | Module         | Responsibility                                      |
//! |----------------|-----------------------------------------------------|
//! | `fs`           | Filesystem-level state: superblock, groups, inodes  |
//! | `inode`        | Inode decoding, file I/O, directories, symlinks     |
//! | `block_group`  | The block group descriptor table                    |
//! | `super_block`  | On-disk superblock parsing and validation           |
//! | `features`     | Feature words and the mount policy they imply       |
//! | `impl_for_vfs` | Wires ext4 types into the VFS trait interfaces      |
//! | `fs_type`      | `FsType` registration glue                          |
//! | `prelude`      | Common imports shared across submodules             |
//!
//! # References
//!
//! - <https://www.kernel.org/doc/html/latest/filesystems/ext4/index.html>

// Set this module's log prefix for `ostd::log`.
macro_rules! __log_prefix {
    () => {
        "ext4: "
    };
}

pub(crate) use fs::Ext4;
pub(crate) use inode::{FilePerm, Inode};

pub(in crate::fs) use self::fs_type::EXT4_TYPE;
use crate::fs::vfs::registry;

mod block_group;
mod features;
mod fs;
mod fs_type;
mod impl_for_vfs;
mod inode;
mod prelude;
mod super_block;

#[cfg(ktest)]
pub(in crate::fs) mod test_utils;

/// Registers the ext4 filesystem type with the VFS registry.
pub(super) fn init() {
    registry::register(&EXT4_TYPE).expect("the ext4 FS type is registered exactly once");
}

/// Brings up the subsystems that ext4 registration depends on.
///
/// Kernel-mode tests run without boot-time initialization and in no
/// particular order, so every test that reaches the registry calls this
/// first. Registration itself is not idempotent, hence the guard.
#[cfg(ktest)]
pub(in crate::fs) fn init_for_ktest() {
    use spin::Once;

    static INIT: Once<()> = Once::new();

    INIT.call_once(|| {
        aster_systree::init_for_ktest();
        crate::fs::vfs::init();
        init();
    });
}

#[cfg(ktest)]
mod test {
    use ostd::prelude::*;

    use super::*;
    use crate::fs::vfs::registry::FsProperties;

    #[ktest]
    fn ext4_is_registered_in_the_vfs() {
        init_for_ktest();

        let fs_type = registry::look_up("ext4").expect("the ext4 FS type is not registered");
        assert_eq!(fs_type.name(), "ext4");
        assert!(fs_type.properties().contains(FsProperties::NEED_DISK));
    }
}
