// SPDX-License-Identifier: MPL-2.0

//! VFS inode trait implementations for the ext4 [`Inode`](super::super::Inode).
//!
//! Every method that would change the volume reports `EROFS`. The timestamp
//! setters cannot, because the VFS declares them infallible, so they do
//! nothing instead; a read-only mount updates no timestamp anyway.

use core::time::Duration;

use device_id::DeviceId;

use crate::{
    device,
    fs::{
        file::{AccessMode, InodeMode, InodeType, PerOpenFileOps, StatusFlags, SyncMode},
        fs_impls::ext4::{FilePerm, Inode as Ext4Inode},
        utils::DirentVisitor,
        vfs::{
            file_system::FileSystem,
            inode::{Extension, FallocMode, FileOps, Inode, Metadata, SymbolicLink},
            path::Dentry,
        },
    },
    prelude::*,
    process::{Gid, Uid},
    vm::page_cache::Vmo,
};

impl FileOps for Ext4Inode {
    fn read_at(
        &self,
        offset: usize,
        writer: &mut VmWriter,
        status_flags: StatusFlags,
    ) -> Result<usize> {
        if status_flags.contains(StatusFlags::O_DIRECT) {
            self.read_direct_at(offset, writer)
        } else {
            self.read_at(offset, writer)
        }
    }

    fn write_at(
        &self,
        _offset: usize,
        _reader: &mut VmReader,
        _status_flags: StatusFlags,
    ) -> Result<usize> {
        return_errno_with_message!(Errno::EROFS, "ext4 writing is not implemented yet")
    }

    fn readdir_at(&self, offset: usize, visitor: &mut dyn DirentVisitor) -> Result<usize> {
        self.readdir_at(offset, visitor)
    }
}

impl Inode for Ext4Inode {
    fn size(&self) -> usize {
        self.file_size()
    }

    fn resize(&self, _self_dentry: &Dentry, _new_size: usize) -> Result<()> {
        return_errno_with_message!(Errno::EROFS, "ext4 writing is not implemented yet")
    }

    fn metadata(&self) -> Result<Metadata> {
        Ok(self.metadata())
    }

    fn ino(&self) -> u64 {
        self.ino() as u64
    }

    fn type_(&self) -> InodeType {
        self.inode_type()
    }

    fn mode(&self) -> Result<InodeMode> {
        Ok(self.mode())
    }

    fn set_mode(&self, _self_dentry: &Dentry, _mode: InodeMode) -> Result<()> {
        return_errno_with_message!(Errno::EROFS, "ext4 writing is not implemented yet")
    }

    fn owner(&self) -> Result<Uid> {
        Ok(Uid::new(self.uid()))
    }

    fn set_owner(&self, _self_dentry: &Dentry, _uid: Uid) -> Result<()> {
        return_errno_with_message!(Errno::EROFS, "ext4 writing is not implemented yet")
    }

    fn group(&self) -> Result<Gid> {
        Ok(Gid::new(self.gid()))
    }

    fn set_group(&self, _self_dentry: &Dentry, _gid: Gid) -> Result<()> {
        return_errno_with_message!(Errno::EROFS, "ext4 writing is not implemented yet")
    }

    fn atime(&self) -> Duration {
        self.atime()
    }

    fn set_atime(&self, _self_dentry: &Dentry, _time: Duration) {}

    fn mtime(&self) -> Duration {
        self.mtime()
    }

    fn set_mtime(&self, _self_dentry: &Dentry, _time: Duration) {}

    fn ctime(&self) -> Duration {
        self.ctime()
    }

    fn set_ctime(&self, _self_dentry: &Dentry, _time: Duration) {}

    fn page_cache(&self) -> Option<Arc<Vmo>> {
        self.page_cache()
    }

    fn open(
        &self,
        _self_dentry: &Dentry,
        access_mode: AccessMode,
        status_flags: StatusFlags,
    ) -> Option<Result<Box<dyn PerOpenFileOps>>> {
        match self.inode_type() {
            inode_type @ (InodeType::BlockDevice | InodeType::CharDevice) => {
                let Some(device_id) = DeviceId::from_encoded_u64(self.device_id()) else {
                    return Some(Err(Error::with_message(
                        Errno::ENODEV,
                        "the device ID is invalid",
                    )));
                };
                let device_type = inode_type
                    .device_type()
                    .expect("BlockDevice and CharDevice always have a device type");
                let Some(device) = device::lookup(device_type, device_id) else {
                    return Some(Err(Error::with_message(
                        Errno::ENODEV,
                        "the required device ID does not exist",
                    )));
                };

                Some(device.open())
            }
            InodeType::NamedPipe => {
                let pipe = self.pipe().expect("NamedPipe inode must have a pipe");
                Some(pipe.open_named(access_mode, status_flags))
            }
            _ => None,
        }
    }

    fn lookup(&self, name: &str) -> Result<Arc<dyn Inode>> {
        Ok(self.lookup(name)?)
    }

    fn read_link(&self) -> Result<SymbolicLink> {
        self.read_link().map(SymbolicLink::Plain)
    }

    fn sync(&self, _mode: SyncMode) -> Result<()> {
        Ok(())
    }

    fn fallocate(&self, _mode: FallocMode, _offset: usize, _len: usize) -> Result<()> {
        return_errno_with_message!(Errno::EROFS, "ext4 writing is not implemented yet")
    }

    fn fs(&self) -> Arc<dyn FileSystem> {
        // The inode must belong to a live filesystem instance.
        self.fs().unwrap()
    }

    fn extension(&self) -> &Extension {
        self.extension()
    }
}

impl From<FilePerm> for InodeMode {
    fn from(perm: FilePerm) -> Self {
        Self::from_bits_truncate(perm.bits() as _)
    }
}
