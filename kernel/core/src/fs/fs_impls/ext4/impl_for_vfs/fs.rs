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
        // A read-only mount holds nothing that needs writing back, but the
        // device may still hold writes made before it was mounted.
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
        if !flags.contains(FsFlags::RDONLY) {
            return_errno_with_message!(Errno::EROFS, "ext4 writing is not implemented yet");
        }
        Ok(())
    }

    fn fs_event_subscriber_stats(&self) -> &FsEventSubscriberStats {
        self.fs_event_subscriber_stats()
    }
}
