// SPDX-License-Identifier: MPL-2.0

//! Filesystem-level state for a mounted ext4 volume.
//!
//! `Ext4` owns what the whole volume shares: the superblock, the block group
//! descriptor table, and the cache of live inodes. Everything it holds is read
//! once at mount and never written, because this iteration mounts read-only.
//!
//! # Mounting
//!
//! A mount fails with `EINVAL` when the volume is not one this driver can
//! claim, which is what lets the root filesystem candidate list move on to
//! another type, and with `EUCLEAN` when it is an ext4 volume whose metadata
//! does not hold together. The root inode is read during the mount so that a
//! mount which succeeds is one whose root can be reached.
//!
//! # Inode cache
//!
//! The VFS gives filesystem implementations no shared inode cache, so one
//! lives here. It holds weak references: an inode stays only as long as
//! something outside holds it, and two paths that reach the same inode get the
//! same object, which is what keeps a hard link's page cache coherent.

use core::sync::atomic::Ordering;

use aster_block::bio::BioCompleteFn;
use device_id::DeviceId;

use super::{
    block_group::{self, GroupDesc},
    features,
    inode::{Inode, InodeDesc, RawInode},
    prelude::*,
    super_block::{RawSuperBlock, SUPER_BLOCK_OFFSET, SuperBlock},
};
use crate::fs::vfs::file_system::{AtomicFsFlags, FileSystem, FsEventSubscriberStats, FsFlags};

/// The root inode number defined by the ext4 on-disk format.
pub(super) const ROOT_INO: Ext4Ino = 2;

/// Number of cache entries past which dead weak references are swept.
///
/// Nothing removes an entry when its inode is dropped, so the map is pruned
/// when it grows instead.
const INODE_CACHE_SWEEP_AT: usize = 1024;

/// Top-level handle for a mounted ext4 filesystem.
#[derive(Debug)]
pub(crate) struct Ext4 {
    block_device: Arc<dyn BlockDevice>,
    super_block: SuperBlock,
    group_descs: Vec<GroupDesc>,
    flags: AtomicFsFlags,
    fs_event_subscriber_stats: FsEventSubscriberStats,
    inode_cache: RwMutex<BTreeMap<Ext4Ino, Weak<Inode>>>,
    self_ref: Weak<Ext4>,
}

/// Opens an ext4 volume on `device`.
pub(super) fn open(
    device: Arc<dyn BlockDevice>,
    flags: FsFlags,
    _data: Option<&str>,
) -> Result<Arc<dyn FileSystem>> {
    let raw_super_block = device.read_val::<RawSuperBlock>(SUPER_BLOCK_OFFSET)?;
    let super_block = SuperBlock::try_from(raw_super_block)?;

    // The write path is not implemented, so every volume mounts read-only
    // whatever its features would allow. Evaluating them is still what decides
    // whether the volume may be mounted at all.
    let flags = features::mount_flags(
        super_block.feature_compat,
        super_block.feature_incompat,
        super_block.feature_ro_compat,
        flags,
    ) | FsFlags::RDONLY;

    let group_descs = block_group::load_all(device.as_ref(), &super_block)?;

    let ext4 = Arc::new_cyclic(|weak_self| Ext4 {
        block_device: device,
        super_block,
        group_descs,
        flags: AtomicFsFlags::new(flags),
        fs_event_subscriber_stats: FsEventSubscriberStats::new(),
        inode_cache: RwMutex::new(BTreeMap::new()),
        self_ref: weak_self.clone(),
    });
    ext4.read_inode(ROOT_INO)?;

    Ok(ext4)
}

impl Ext4 {
    pub(super) fn block_device(&self) -> &dyn BlockDevice {
        self.block_device.as_ref()
    }

    pub(super) fn super_block(&self) -> &SuperBlock {
        &self.super_block
    }

    pub(super) fn container_device_id(&self) -> DeviceId {
        self.block_device.id()
    }

    pub(super) fn fs_flags(&self) -> FsFlags {
        self.flags.load(Ordering::Relaxed)
    }

    pub(super) fn fs_event_subscriber_stats(&self) -> &FsEventSubscriberStats {
        &self.fs_event_subscriber_stats
    }

    pub(super) fn root_inode(&self) -> Result<Arc<Inode>> {
        self.read_inode(ROOT_INO)
    }

    /// Returns the inode numbered `ino`, reading it from the device on a miss.
    pub(super) fn read_inode(&self, ino: Ext4Ino) -> Result<Arc<Inode>> {
        if let Some(inode) = self.inode_cache.read().get(&ino).and_then(Weak::upgrade) {
            return Ok(inode);
        }

        let mut cache = self.inode_cache.write();
        // Another thread may have inserted the inode while the read lock was
        // dropped.
        if let Some(inode) = cache.get(&ino).and_then(Weak::upgrade) {
            return Ok(inode);
        }

        let raw_inode = self.read_raw_inode(ino)?;
        let desc = InodeDesc::from_raw(
            &raw_inode,
            self.super_block.inode_size,
            self.super_block.feature_ro_compat,
        )?;
        let inode = Inode::new(
            ino,
            desc,
            self.super_block.blocks_count,
            self.self_ref.clone(),
        )?;

        if cache.len() >= INODE_CACHE_SWEEP_AT {
            cache.retain(|_, weak| weak.strong_count() > 0);
        }
        cache.insert(ino, Arc::downgrade(&inode));
        Ok(inode)
    }

    /// Reads blocks synchronously starting at `bid`.
    pub(super) fn read_blocks(&self, bid: Ext4Bid, bio_segment: BioSegment) -> Result<()> {
        match self
            .block_device
            .read_blocks(Bid::new(bid), vec![bio_segment])?
        {
            BioStatus::Complete => Ok(()),
            _ => return_errno_with_message!(Errno::EIO, "failed to read blocks from the device"),
        }
    }

    /// Submits an asynchronous block read starting at `bid`.
    pub(super) fn read_blocks_async(
        &self,
        bid: Ext4Bid,
        bio_segment: BioSegment,
        complete_fn: Option<BioCompleteFn>,
        io_batch: &mut IoBatch,
    ) -> Result<()> {
        self.block_device.read_blocks_async(
            Bid::new(bid),
            vec![bio_segment],
            complete_fn,
            io_batch,
        )?;
        Ok(())
    }

    /// Reads one on-disk inode record.
    ///
    /// Only the leading bytes this driver understands are read; a larger inode
    /// carries extended attributes after them, and a 128-byte one stops before
    /// the fields ext4 added.
    fn read_raw_inode(&self, ino: Ext4Ino) -> Result<RawInode> {
        let (group_idx, index_in_group) = self.super_block.locate_inode(ino)?;
        let group_desc = self
            .group_descs
            .get(group_idx)
            .ok_or_else(|| Error::with_message(Errno::EIO, "block group index out of range"))?;
        if !group_desc.is_inode_initialized(index_in_group, &self.super_block) {
            return_errno_with_message!(Errno::ENOENT, "inode was never written");
        }

        let offset = Bid::new(group_desc.inode_table_bid).to_offset()
            + index_in_group as usize * self.super_block.inode_size;
        let mut bytes = [0u8; size_of::<RawInode>()];
        let len = self.super_block.inode_size.min(bytes.len());
        self.block_device
            .read_bytes(offset, &mut bytes[..len])
            .map_err(|_| Error::with_message(Errno::EIO, "failed to read inode"))?;

        Ok(RawInode::from_bytes(&bytes))
    }
}

#[cfg(ktest)]
mod test {
    use ostd::prelude::*;

    use super::*;
    use crate::fs::fs_impls::ext4::test_utils::{
        Ext4MemoryDisk, Ext4VolumeBuilder, SECOND_GROUP_INO, assert_errno,
    };

    /// A zeroed device carries no magic number, so the volume is not ours.
    #[ktest]
    fn opening_a_blank_device_reports_einval() {
        let device = Arc::new(Ext4MemoryDisk::new(64)) as Arc<dyn BlockDevice>;

        assert_errno(open(device, FsFlags::empty(), None), Errno::EINVAL);
    }

    #[ktest]
    fn a_volume_mounts_and_reaches_its_root() {
        let volume = Ext4VolumeBuilder::new().build();

        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs.downcast_ref::<Ext4>().expect("should be an ext4 volume");

        let root = ext4.root_inode().expect("root should be readable");
        assert_eq!(root.ino(), ROOT_INO);
        assert_eq!(root.inode_type(), InodeType::Dir);
    }

    /// Two reads of the same inode must yield the same object, or a hard link
    /// would get a second page cache over the same blocks.
    #[ktest]
    fn reading_one_inode_twice_yields_one_object() {
        let volume = Ext4VolumeBuilder::new().build();
        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs.downcast_ref::<Ext4>().expect("should be an ext4 volume");

        let first = ext4.read_inode(ROOT_INO).expect("should read");
        let second = ext4.read_inode(ROOT_INO).expect("should read");

        assert!(Arc::ptr_eq(&first, &second));
    }

    /// An inode in a group other than the first proves the descriptor table is
    /// indexed rather than assumed.
    #[ktest]
    fn an_inode_in_a_later_group_is_readable() {
        let volume = Ext4VolumeBuilder::new().build();
        let ino = SECOND_GROUP_INO;
        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs.downcast_ref::<Ext4>().expect("should be an ext4 volume");

        let inode = ext4.read_inode(ino).expect("should read");

        assert_eq!(inode.inode_type(), InodeType::File);
    }

    /// A volume without `64BIT` keeps its group descriptors in the short form,
    /// which is a different path through the descriptor reader.
    #[ktest]
    fn a_volume_with_short_descriptors_mounts() {
        let volume = Ext4VolumeBuilder::new().desc_size(32).build();

        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs.downcast_ref::<Ext4>().expect("should be an ext4 volume");

        assert_eq!(ext4.super_block().desc_size, 32);
        assert_eq!(
            ext4.read_inode(SECOND_GROUP_INO)
                .expect("should read")
                .inode_type(),
            InodeType::File
        );
    }

    #[ktest]
    fn mounting_a_stock_volume_forces_read_only() {
        let volume = Ext4VolumeBuilder::new().stock_features().build();

        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");

        assert!(fs.flags().contains(FsFlags::RDONLY));
    }
}
