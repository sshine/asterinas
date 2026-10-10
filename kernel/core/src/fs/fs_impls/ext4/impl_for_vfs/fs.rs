// SPDX-License-Identifier: MPL-2.0

//! [`FileSystem`] trait implementation for [`Ext4`].

use aster_block::{BLOCK_SIZE, bio::BioStatus};

use crate::{
    fs::{
        fs_impls::ext4::{Ext4, super_block::MAGIC_NUM},
        utils::NAME_MAX,
        vfs::{
            file_system::{FileSystem, FsEventSubscriberStats, FsFlags, SuperBlock},
            inode::Inode,
        },
    },
    prelude::*,
};

impl FileSystem for Ext4 {
    fn name(&self) -> &'static str {
        "ext4"
    }

    fn sync(&self) -> Result<()> {
        if self.is_writable() {
            self.sync_all()?;
        }
        if self.block_device().sync()? != BioStatus::Complete {
            return_errno_with_message!(Errno::EIO, "failed to flush block device");
        }
        Ok(())
    }

    fn root_inode(&self) -> Arc<dyn Inode> {
        // Mounting reads the root inode and fails if it cannot, and the cache
        // holds it for as long as the mount exists.
        self.root_inode()
            .expect("the root inode was read when the volume was mounted")
    }

    fn sb(&self) -> SuperBlock {
        let sb = self.super_block();
        SuperBlock {
            magic: MAGIC_NUM as u64,
            bsize: BLOCK_SIZE,
            blocks: sb.blocks_count as usize,
            bfree: sb.free_blocks_count as usize,
            bavail: sb
                .free_blocks_count
                .saturating_sub(sb.reserved_blocks_count) as usize,
            files: sb.inodes_count as usize,
            ffree: sb.free_inodes_count as usize,
            fsid: 0,
            namelen: NAME_MAX,
            frsize: BLOCK_SIZE,
            flags: 0,
            container_dev_id: self.container_device_id(),
        }
    }

    fn flags(&self) -> FsFlags {
        self.fs_flags()
    }

    fn set_fs_flags(&self, flags: FsFlags, _data: Option<&str>, _ctx: &Context) -> Result<()> {
        // Going from read-only to read-write would need the checks the mount
        // makes, and remounting read-only would need everything cached to be
        // written back first. Neither is implemented.
        if flags.contains(FsFlags::RDONLY) != self.flags().contains(FsFlags::RDONLY) {
            return_errno_with_message!(Errno::EOPNOTSUPP, "ext4 cannot be remounted");
        }
        Ok(())
    }

    fn fs_event_subscriber_stats(&self) -> &FsEventSubscriberStats {
        self.fs_event_subscriber_stats()
    }
}

#[cfg(ktest)]
mod test {
    use ostd::prelude::ktest;

    use super::*;
    use crate::fs::fs_impls::ext4::{
        fs as ext4_fs,
        test_utils::{Ext4Volume, Ext4VolumeBuilder, RESERVED_BLOCKS, TOTAL_BLOCKS, TOTAL_INODES},
    };

    /// Mounts a fixture volume, keeping it so that a test can compare what
    /// `statfs` reports against what the fixture put there.
    fn mount_volume() -> (Ext4Volume, Arc<dyn FileSystem>) {
        let volume = Ext4VolumeBuilder::new().build();
        let fs = ext4_fs::open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        (volume, fs)
    }

    fn mount() -> Arc<dyn FileSystem> {
        mount_volume().1
    }

    #[ktest]
    fn the_filesystem_names_itself_ext4() {
        assert_eq!(mount().name(), "ext4");
    }

    /// `statfs` is read straight out of the superblock, so a field taken from
    /// the wrong one shows up as a plausible but wrong number rather than as a
    /// failure.
    #[ktest]
    fn statfs_reports_the_counts_the_superblock_holds() {
        let (volume, fs) = mount_volume();
        let sb = fs.sb();

        assert_eq!(sb.magic, MAGIC_NUM as u64);
        assert_eq!(sb.bsize, BLOCK_SIZE);
        assert_eq!(sb.frsize, BLOCK_SIZE);
        assert_eq!(sb.blocks, TOTAL_BLOCKS as usize);
        assert_eq!(sb.bfree, volume.free_blocks as usize);
        assert_eq!(sb.files, TOTAL_INODES as usize);
        assert_eq!(sb.ffree, volume.free_inodes as usize);
        assert_eq!(sb.namelen, NAME_MAX);
    }

    /// The blocks an unprivileged writer may use exclude the reserved ones.
    #[ktest]
    fn statfs_withholds_the_reserved_blocks_from_bavail() {
        let (volume, fs) = mount_volume();
        let sb = fs.sb();

        assert_eq!(sb.bavail, (volume.free_blocks - RESERVED_BLOCKS) as usize);
    }

    #[ktest]
    fn the_root_inode_is_the_directory_at_inode_two() {
        let root = mount().root_inode();

        assert_eq!(root.ino(), 2);
        assert_eq!(root.type_(), crate::fs::file::InodeType::Dir);
    }

    #[ktest]
    fn syncing_a_mount_succeeds() {
        mount().sync().expect("should write back");
    }

    /// A volume whose features this driver can maintain mounts writable.
    #[ktest]
    fn a_reduced_feature_volume_mounts_writable() {
        assert!(!mount().flags().contains(FsFlags::RDONLY));
    }
}
