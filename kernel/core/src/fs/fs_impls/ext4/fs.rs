// SPDX-License-Identifier: MPL-2.0

//! Filesystem-level ext4 state.
//!
//! Only the mount-time superblock check lives here so far. The volume handle
//! that holds the block groups and routes allocation arrives with the read
//! path.

use super::{
    prelude::*,
    super_block::{RawSuperBlock, SUPER_BLOCK_OFFSET, SuperBlock},
};
use crate::fs::vfs::file_system::{FileSystem, FsFlags};

/// Opens an ext4 volume on `device`.
///
/// Reports EINVAL for a volume this driver cannot claim, which is what lets
/// the root filesystem candidate list move on to another type.
pub(super) fn open(
    device: Arc<dyn BlockDevice>,
    _flags: FsFlags,
    _data: Option<&str>,
) -> Result<Arc<dyn FileSystem>> {
    let raw_super_block = device.read_val::<RawSuperBlock>(SUPER_BLOCK_OFFSET)?;
    let _super_block = SuperBlock::try_from(raw_super_block)?;

    return_errno_with_message!(Errno::ENOSYS, "ext4 mounting is not implemented yet")
}

#[cfg(ktest)]
mod test {
    use ostd::prelude::*;

    use super::*;
    use crate::fs::fs_impls::ext4::test_utils::{Ext4MemoryDisk, assert_errno};

    /// A zeroed device carries no magic number, so the volume is not ours.
    #[ktest]
    fn opening_a_blank_device_reports_einval() {
        let device = Arc::new(Ext4MemoryDisk::new(64)) as Arc<dyn BlockDevice>;

        assert_errno(open(device, FsFlags::empty(), None), Errno::EINVAL);
    }

    /// ENOSYS rather than EINVAL is what distinguishes a superblock that was
    /// read and accepted from one that was never reached.
    #[ktest]
    fn opening_a_valid_volume_gets_past_the_super_block() {
        let device = Arc::new(Ext4MemoryDisk::with_valid_super_block()) as Arc<dyn BlockDevice>;

        assert_errno(open(device, FsFlags::empty(), None), Errno::ENOSYS);
    }
}
