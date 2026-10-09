// SPDX-License-Identifier: MPL-2.0

//! Ext4 filesystem implementation.
//!
//! This module is the entry point for ext4 support in Asterinas. A caller
//! registers `Ext4` as a filesystem type via `init`, after which the VFS
//! can mount ext4 volumes and operate on them through the standard
//! filesystem trait interfaces.
//!
//! The Fourth Extended File System (ext4) shares its block group layout and
//! allocation bitmaps with ext2, and differs chiefly in how an inode maps
//! logical blocks: ext2 walks a tree of indirect block pointers, while ext4
//! stores an extent tree rooted inline in the inode. The two formats also
//! share the magic number 0xEF53 and are told apart by the feature words in
//! the superblock.
//!
//! # Feature policy
//!
//! Three feature words in the superblock describe what a volume requires of
//! its driver. Incompatible features that this driver does not implement
//! reject the mount; read-only-compatible ones that it cannot maintain
//! force the mount read-only rather than refusing it, so that a volume made
//! by a stock `mkfs.ext4` can still be read.
//!
//! # References
//!
//! - <https://www.kernel.org/doc/html/latest/filesystems/ext4/index.html>

pub(in crate::fs) use self::fs_type::EXT4_TYPE;
use crate::fs::vfs::registry;

mod features;
mod fs;
mod fs_type;
mod prelude;
mod super_block;

#[cfg(ktest)]
mod test_utils;

/// Registers the ext4 filesystem type with the VFS registry.
pub(super) fn init() {
    registry::register(&EXT4_TYPE).unwrap();
}

/// Brings up the subsystems that ext4 registration depends on.
///
/// Kernel-mode tests run without boot-time initialization and in no
/// particular order, so every test that reaches the registry calls this
/// first. Registration itself is not idempotent, hence the guard.
#[cfg(ktest)]
fn init_for_ktest() {
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
