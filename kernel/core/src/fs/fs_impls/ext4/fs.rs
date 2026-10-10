// SPDX-License-Identifier: MPL-2.0

//! Filesystem-level state for a mounted ext4 volume.
//!
//! `Ext4` owns what the whole volume shares: the superblock, the block group
//! descriptor table, and the cache of live inodes. It is also where blocks and
//! inode numbers are handed out, since both come from a group and a file may
//! take them from any of them.
//!
//! # Mounting
//!
//! A mount fails with `EINVAL` when the volume is not one this driver can
//! claim, which is what lets the root filesystem candidate list move on to
//! another type, and with `EUCLEAN` when it is an ext4 volume whose metadata
//! does not hold together. The root inode is read during the mount so that a
//! mount which succeeds is one whose root can be reached.
//!
//! A read-write mount clears the clean flag in the superblock for as long as
//! it lasts, so that a crash leaves the volume marked for `e2fsck` rather than
//! appearing to have been unmounted properly.
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
    block_group::{self, BlockGroup, GroupDesc},
    features,
    inode::{Inode, InodeDesc, RawInode},
    prelude::*,
    super_block::{RawSuperBlock, STATE_VALID_FS, SUPER_BLOCK_OFFSET, SuperBlock},
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
    /// Geometry and features, which no write changes.
    super_block: SuperBlock,
    /// The two superblock counters that allocation does change.
    ///
    /// Keeping them apart from the rest is what lets every other field be read
    /// without a lock, which the read path does constantly.
    counters: RwMutex<Dirty<Counters>>,
    groups: Vec<BlockGroup>,
    flags: AtomicFsFlags,
    fs_event_subscriber_stats: FsEventSubscriberStats,
    inode_cache: RwMutex<BTreeMap<Ext4Ino, Weak<Inode>>>,
    /// Serializes metadata read-modify-writes.
    ///
    /// An inode record is smaller than a sector, so writing one reads and
    /// rewrites the bytes of whatever shares its sector. Without this lock two
    /// such writes would lose one of the two updates.
    metadata_lock: Mutex<()>,
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

    let mut flags = features::mount_flags(
        super_block.feature_compat,
        super_block.feature_incompat,
        super_block.feature_ro_compat,
        flags,
    );

    let groups = block_group::load_all(device.as_ref(), &super_block)?;

    // A group whose bitmaps were never written holds arbitrary bytes where its
    // allocation state should be. Only features that already force a read-only
    // mount produce one, so this is a check on that reasoning; a volume that
    // has one anyway is read rather than refused.
    if !flags.contains(FsFlags::RDONLY) && block_group::has_uninitialized_bitmaps(&groups) {
        warn!("a group has uninitialized bitmaps, so the volume is mounted read-only");
        flags |= FsFlags::RDONLY;
    }

    let counters = Counters {
        free_blocks_count: super_block.free_blocks_count,
        free_inodes_count: super_block.free_inodes_count,
    };

    let ext4 = Arc::new_cyclic(|weak_self| Ext4 {
        block_device: device,
        super_block,
        counters: RwMutex::new(Dirty::new(counters)),
        groups,
        flags: AtomicFsFlags::new(flags),
        fs_event_subscriber_stats: FsEventSubscriberStats::new(),
        inode_cache: RwMutex::new(BTreeMap::new()),
        metadata_lock: Mutex::new(()),
        self_ref: weak_self.clone(),
    });
    ext4.read_inode(ROOT_INO)?;

    // Marking the volume dirty is the last thing the mount does, so a volume
    // that failed to mount is left as it was found.
    if ext4.is_writable() {
        ext4.set_clean_flag(false)?;
    }

    Ok(ext4)
}

/// The superblock counters that allocation changes.
#[derive(Clone, Copy, Debug)]
struct Counters {
    free_blocks_count: u64,
    free_inodes_count: u32,
}

impl Drop for Ext4 {
    fn drop(&mut self) {
        if !self.is_writable() {
            return;
        }

        // Nothing else will write these back: the mount is going away.
        if let Err(err) = self.sync_all() {
            error!("failed to write back the volume on unmount: {:?}", err);
            return;
        }
        if let Err(err) = self.set_clean_flag(true) {
            error!("failed to mark the volume clean on unmount: {:?}", err);
        }
    }
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

    /// Returns how many blocks the volume still has free.
    /// Returns a weak handle to this filesystem, for the inodes it creates.
    pub(super) fn weak_self(&self) -> Weak<Ext4> {
        self.self_ref.clone()
    }

    /// Remembers a newly created inode, so that a later lookup finds the same
    /// object rather than reading a second one off the device.
    pub(super) fn insert_inode(&self, inode: Arc<Inode>) {
        self.inode_cache
            .write()
            .insert(inode.ino(), Arc::downgrade(&inode));
    }

    pub(super) fn free_blocks_count(&self) -> u64 {
        self.counters.read().free_blocks_count
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
            Dirty::new(desc),
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

    /// Submits an asynchronous block write starting at `bid`.
    pub(super) fn write_blocks_async(
        &self,
        bid: Ext4Bid,
        bio_segment: BioSegment,
        complete_fn: Option<BioCompleteFn>,
        io_batch: &mut IoBatch,
    ) -> Result<()> {
        self.block_device.write_blocks_async(
            Bid::new(bid),
            vec![bio_segment],
            complete_fn,
            io_batch,
        )?;
        Ok(())
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

    /// Returns the group holding `ino`, with the inode's index within it.
    fn locate_inode(&self, ino: Ext4Ino) -> Result<(&BlockGroup, u32)> {
        let (group_idx, index_in_group) = self.super_block.locate_inode(ino)?;
        let group = self
            .groups
            .get(group_idx)
            .ok_or_else(|| Error::with_message(Errno::EIO, "block group index out of range"))?;

        Ok((group, index_in_group))
    }

    /// Returns the byte offset of one inode record on the device.
    fn inode_offset(&self, ino: Ext4Ino) -> Result<usize> {
        let (group, index_in_group) = self.locate_inode(ino)?;
        let desc = group.desc();
        if !desc.is_inode_initialized(index_in_group, &self.super_block) {
            return_errno_with_message!(Errno::ENOENT, "inode was never written");
        }

        Ok(Bid::new(desc.inode_table_bid).to_offset()
            + index_in_group as usize * self.super_block.inode_size)
    }

    /// Returns whether this mount may change the volume.
    pub(super) fn is_writable(&self) -> bool {
        !self.fs_flags().contains(FsFlags::RDONLY)
    }

    /// Fails unless this mount may change the volume.
    pub(super) fn check_writable(&self) -> Result<()> {
        if !self.is_writable() {
            return_errno_with_message!(Errno::EROFS, "the volume is mounted read-only");
        }

        Ok(())
    }

    /// Allocates up to `count` consecutive blocks, preferring those near
    /// `goal`.
    ///
    /// A shorter run than asked for is a success: a file is free to use
    /// several extents, and insisting on one run would report `ENOSPC` on a
    /// volume that has the space in pieces.
    pub(super) fn alloc_blocks(&self, goal: Ext4Bid, count: u32) -> Result<Range<Ext4Bid>> {
        self.check_writable()?;
        if count == 0 {
            return_errno_with_message!(Errno::EINVAL, "zero block allocation requested");
        }

        // Starting at the goal's own group is what keeps a file's blocks near
        // each other, and near the inode that names them.
        let goal_group = self.group_of_block(goal).unwrap_or(0);
        for offset in 0..self.groups.len() {
            let group_idx = (goal_group + offset) % self.groups.len();
            let Some(range) = self.groups[group_idx].alloc_blocks(self, count)? else {
                continue;
            };

            let allocated = range.end - range.start;
            let mut counters = self.counters.write();
            counters.free_blocks_count = counters.free_blocks_count.saturating_sub(allocated);
            return Ok(range);
        }

        return_errno_with_message!(Errno::ENOSPC, "no free blocks on the volume")
    }

    /// Frees a range of blocks, which may span several groups.
    pub(super) fn free_blocks(&self, range: Range<Ext4Bid>) -> Result<()> {
        self.check_writable()?;
        if range.is_empty() {
            return Ok(());
        }
        if range.end > self.super_block.blocks_count {
            return_errno_with_message!(Errno::EIO, "freeing blocks outside the volume");
        }

        let mut cursor = range.start;
        while cursor < range.end {
            let group_idx = self
                .group_of_block(cursor)
                .ok_or_else(|| Error::with_message(Errno::EIO, "freeing a reserved block"))?;
            let group = &self.groups[group_idx];
            let group_end = group.first_block() + group.nr_blocks() as u64;
            let end = range.end.min(group_end);

            group.free_blocks(self, cursor..end)?;
            let mut counters = self.counters.write();
            counters.free_blocks_count += end - cursor;
            cursor = end;
        }

        Ok(())
    }

    /// Allocates an inode number, preferring the group that holds `parent`.
    pub(super) fn alloc_ino(&self, parent: Ext4Ino, is_dir: bool) -> Result<Ext4Ino> {
        self.check_writable()?;

        let goal_group = self
            .super_block
            .locate_inode(parent)
            .map_or(0, |(group_idx, _)| group_idx);
        for offset in 0..self.groups.len() {
            let group_idx = (goal_group + offset) % self.groups.len();
            let Some(index) = self.groups[group_idx].alloc_inode(self, is_dir)? else {
                continue;
            };

            let ino = group_idx as u32 * self.super_block.inodes_per_group + index + 1;
            // The format reserves the low inode numbers for itself. A bitmap
            // that offers one is wrong, so it goes straight back.
            if ino < self.super_block.first_ino && ino != ROOT_INO {
                self.groups[group_idx].free_inode(self, index, is_dir)?;
                continue;
            }

            let mut counters = self.counters.write();
            counters.free_inodes_count = counters.free_inodes_count.saturating_sub(1);
            return Ok(ino);
        }

        return_errno_with_message!(Errno::ENOSPC, "no free inodes on the volume")
    }

    /// Returns an inode number to its group.
    pub(super) fn free_ino(&self, ino: Ext4Ino, is_dir: bool) -> Result<()> {
        self.check_writable()?;

        let (group, index_in_group) = self.locate_inode(ino)?;
        group.free_inode(self, index_in_group, is_dir)?;
        let mut counters = self.counters.write();
        counters.free_inodes_count += 1;
        Ok(())
    }

    /// Returns the group holding `bid`, if any does.
    #[cfg(ktest)]
    pub(super) fn group_of(&self, bid: Ext4Bid) -> Option<&BlockGroup> {
        self.group_of_block(bid).map(|idx| &self.groups[idx])
    }

    /// Returns the index of the group holding `bid`, if it holds data at all.
    fn group_of_block(&self, bid: Ext4Bid) -> Option<usize> {
        let first_data_block = self.super_block.first_data_block as u64;
        if bid < first_data_block {
            return None;
        }

        let group_idx =
            ((bid - first_data_block) / self.super_block.blocks_per_group as u64) as usize;
        (group_idx < self.groups.len()).then_some(group_idx)
    }

    /// Writes back everything this mount has changed.
    pub(super) fn sync_all(&self) -> Result<()> {
        // The inodes are collected before any of them is synced, so that the
        // cache lock is not held across device I/O.
        let inodes: Vec<Arc<Inode>> = self
            .inode_cache
            .read()
            .values()
            .filter_map(Weak::upgrade)
            .collect();
        for inode in inodes {
            inode.sync_all()?;
        }

        for group in &self.groups {
            group.sync(self)?;
        }
        self.sync_super_block()
    }

    /// Writes the counters the superblock keeps back into it.
    fn sync_super_block(&self) -> Result<()> {
        let mut counters = self.counters.write();
        if !counters.is_dirty() {
            return Ok(());
        }

        let _guard = self.metadata_lock.lock();
        let mut raw = self
            .block_device
            .read_val::<RawSuperBlock>(SUPER_BLOCK_OFFSET)
            .map_err(|_| Error::with_message(Errno::EIO, "failed to read the superblock"))?;
        raw.free_blocks_count_lo = counters.free_blocks_count as u32;
        raw.free_blocks_count_hi = (counters.free_blocks_count >> 32) as u32;
        raw.free_inodes_count = counters.free_inodes_count;
        self.write_metadata_locked(SUPER_BLOCK_OFFSET, raw.as_bytes())?;

        counters.clear_dirty();
        Ok(())
    }

    /// Marks the volume clean or dirty in its superblock.
    ///
    /// A read-write mount clears the clean flag for as long as it lasts, so
    /// that a crash leaves the volume marked for `e2fsck`.
    fn set_clean_flag(&self, clean: bool) -> Result<()> {
        let _guard = self.metadata_lock.lock();
        let mut raw = self
            .block_device
            .read_val::<RawSuperBlock>(SUPER_BLOCK_OFFSET)
            .map_err(|_| Error::with_message(Errno::EIO, "failed to read the superblock"))?;
        if clean {
            raw.state |= STATE_VALID_FS;
        } else {
            raw.state &= !STATE_VALID_FS;
        }
        self.write_metadata_locked(SUPER_BLOCK_OFFSET, raw.as_bytes())
    }

    /// Writes a run of metadata bytes to the device.
    ///
    /// Every metadata write goes through here, so that journalling later
    /// becomes a change to one function rather than to every writeback path.
    pub(super) fn write_metadata(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        let _guard = self.metadata_lock.lock();
        self.write_metadata_locked(offset, bytes)
    }

    fn write_metadata_locked(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.block_device
            .write_bytes(offset, bytes)
            .map_err(|_| Error::with_message(Errno::EIO, "failed to write metadata"))
    }

    /// Writes one group descriptor back, keeping the fields this driver does
    /// not own.
    pub(super) fn write_back_group_desc(&self, group_idx: usize, desc: &GroupDesc) -> Result<()> {
        let desc_size = self.super_block.desc_size;
        let offset =
            Bid::new(self.super_block.group_descriptors_bid()).to_offset() + group_idx * desc_size;

        let _guard = self.metadata_lock.lock();
        let mut bytes = vec![0u8; desc_size];
        self.block_device
            .read_bytes(offset, &mut bytes)
            .map_err(|_| Error::with_message(Errno::EIO, "failed to read a group descriptor"))?;
        block_group::patch_raw_desc(&mut bytes, desc);
        self.write_metadata_locked(offset, &bytes)
    }

    /// Returns how many leading bytes of an inode record this driver owns.
    ///
    /// A larger inode carries extended attributes after them, and a 128-byte
    /// one stops before the fields ext4 added.
    fn inode_record_len(&self) -> usize {
        self.super_block.inode_size.min(size_of::<RawInode>())
    }

    /// Reads one on-disk inode record.
    fn read_raw_inode(&self, ino: Ext4Ino) -> Result<RawInode> {
        let offset = self.inode_offset(ino)?;
        let mut bytes = [0u8; size_of::<RawInode>()];
        self.block_device
            .read_bytes(offset, &mut bytes[..self.inode_record_len()])
            .map_err(|_| Error::with_message(Errno::EIO, "failed to read inode"))?;

        Ok(RawInode::from_bytes(&bytes))
    }

    /// Writes one inode record back, keeping the bytes this driver does not
    /// own.
    ///
    /// The record is read before it is written so that the extended attributes
    /// and checksums stored past the fields this driver decodes survive.
    pub(super) fn write_back_inode(&self, ino: Ext4Ino, desc: &InodeDesc) -> Result<()> {
        let _guard = self.metadata_lock.lock();

        let offset = self.inode_offset(ino)?;
        let mut raw = self.read_raw_inode(ino)?;
        desc.write_into(
            &mut raw,
            self.super_block.inode_size,
            self.super_block.feature_ro_compat,
        );
        self.block_device
            .write_bytes(offset, &raw.as_bytes()[..self.inode_record_len()])
            .map_err(|_| Error::with_message(Errno::EIO, "failed to write inode"))?;

        Ok(())
    }
}

#[cfg(ktest)]
mod test {
    use ostd::prelude::*;

    use super::*;
    use crate::fs::fs_impls::ext4::test_utils::{
        Ext4MemoryDisk, Ext4VolumeBuilder, HELLO_INO, INODES_PER_GROUP, SECOND_GROUP_INO,
        assert_errno,
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

    /// An inode past the point a group says its table was written holds
    /// whatever was on the disk before, so it must not be served.
    ///
    /// The count is only maintained on a volume that checksums its group
    /// descriptors, which is why the fixture declares the stock features.
    #[ktest]
    fn an_inode_past_the_written_part_of_a_table_is_refused() {
        let volume = Ext4VolumeBuilder::new()
            .stock_features()
            .itable_unused(INODES_PER_GROUP - 10)
            .build();
        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs.downcast_ref::<Ext4>().expect("should be an ext4 volume");

        // Inode 11 sits at index 10, one past the ten that were written.
        assert_errno(ext4.read_inode(HELLO_INO), Errno::ENOENT);
    }

    #[ktest]
    fn an_inode_inside_the_written_part_of_a_table_is_readable() {
        let volume = Ext4VolumeBuilder::new()
            .stock_features()
            .itable_unused(INODES_PER_GROUP - 11)
            .build();
        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs.downcast_ref::<Ext4>().expect("should be an ext4 volume");

        assert_eq!(
            ext4.read_inode(HELLO_INO)
                .expect("should read")
                .inode_type(),
            InodeType::File
        );
    }

    /// Without those features the count means nothing, so every inode in the
    /// table is readable whatever it says.
    #[ktest]
    fn the_written_count_is_ignored_when_nothing_maintains_it() {
        let volume = Ext4VolumeBuilder::new()
            .itable_unused(INODES_PER_GROUP)
            .build();
        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs.downcast_ref::<Ext4>().expect("should be an ext4 volume");

        assert!(ext4.read_inode(HELLO_INO).is_ok());
    }

    #[ktest]
    fn mounting_a_stock_volume_forces_read_only() {
        let volume = Ext4VolumeBuilder::new().stock_features().build();

        let fs = open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");

        assert!(fs.flags().contains(FsFlags::RDONLY));
    }
}
