// SPDX-License-Identifier: MPL-2.0

//! VFS filesystem-type registration for ext4.
//!
//! `Ext4Type` implements the `FsType` trait so the VFS layer can
//! discover and mount ext4 volumes by name (`"ext4"`).

use aster_systree::SysNode;
use device_id::DeviceId;

use crate::{
    fs::vfs::{
        file_system::FileSystem,
        registry::{FsCache, FsCreationCtx, FsProperties, FsType},
    },
    prelude::*,
};

/// VFS-visible Ext4 filesystem type.
pub(in crate::fs) struct Ext4Type {
    cache: FsCache<DeviceId>,
}

/// The VFS filesystem type descriptor for Ext4.
pub(in crate::fs) static EXT4_TYPE: Ext4Type = Ext4Type {
    cache: FsCache::new(),
};

impl FsType for Ext4Type {
    type Key = DeviceId;

    fn name(&self) -> &'static str {
        "ext4"
    }

    fn properties(&self) -> FsProperties {
        FsProperties::NEED_DISK
    }

    fn create(&self, fs_creation_ctx: &mut FsCreationCtx) -> Result<Arc<dyn FileSystem>> {
        fs_creation_ctx.resolve_block_device()?;

        return_errno_with_message!(Errno::ENOSYS, "ext4 mounting is not implemented yet")
    }

    fn obtain_key_and_cache(
        &self,
        fs_creation_ctx: &mut FsCreationCtx,
    ) -> Option<(DeviceId, &FsCache<DeviceId>)> {
        let key = fs_creation_ctx
            .resolve_block_device()
            .ok()
            .map(|disk| disk.id())?;

        Some((key, &self.cache))
    }

    fn sysnode(&self) -> Option<Arc<dyn SysNode>> {
        None
    }
}
