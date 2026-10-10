// SPDX-License-Identifier: MPL-2.0

//! Ext4 inode implementation.
//!
//! The central type, `Inode`, represents one in-memory ext4 inode and is the
//! concrete inode object exposed to the VFS adapter.
//!
//! # Type aliases
//!
//! - `Iblock` — logical block index within a file (0-based).
//! - `Ext4Bid` — physical block address on the backing device. 64 bits wide,
//!   because an extent carries a 48-bit address.
//! - `Ext4Ino` — on-disk inode number (32 bits; the VFS widens to `u64`).
//!
//! # Data model
//!
//! ```text
//! Inode  →  InodeDesc  →  RawInode
//! ```
//!
//! - `Inode` — the ref-counted handle, holding the decoded descriptor and the
//!   payload its type needs.
//! - `InodeDesc` — a decoded, Rust-typed mirror of the on-disk inode fields.
//! - `RawInode` — the on-disk layout (`#[repr(C)]`), covering the classic
//!   128-byte area and the 32 bytes ext4 adds after it.
//!
//! # Locking
//!
//! `Inode::inner` guards everything a write can change. The inode type and
//! number sit outside it because neither ever changes. An operation that
//! touches two inodes, such as a rename, takes their locks in ascending inode
//! order so that two such operations cannot deadlock against each other.
//!
//! # Submodules
//!
//! | Submodule       | Responsibility                                     |
//! |-----------------|----------------------------------------------------|
//! | `extent`        | The block map: extent tree walking and validation  |
//! | `io_range`      | Block-backed I/O range planning                    |
//! | `block_manager` | Page-cache backend                                 |
//! | `dir`           | Directory entry decoding, lookup, and readdir      |

mod block_manager;
mod create;
mod dir;
mod extent;
mod io_range;

use device_id::{DeviceId, decode_device_numbers, encode_device_numbers};
use ostd::const_assert;

use self::{
    block_manager::InodeBlockManager,
    extent::{EXTENT_ROOT_LEN, ExtentTree},
    io_range::IoRange,
};
use super::{fs::Ext4, prelude::*};
use crate::{
    fs::{
        file::InodeMode,
        pipe::Pipe,
        vfs::inode::{Extension, Metadata},
    },
    process::{Gid, Uid},
    vm::page_cache::Vmo,
};

/// Logical block index within a file (0-based).
pub(super) type Iblock = u32;
/// Physical block address on the device.
pub(super) type Ext4Bid = u64;
/// On-disk inode number (32 bits; the VFS widens to `u64`).
pub(super) type Ext4Ino = u32;

/// The classic inode area, which every ext4 inode begins with.
const CLASSIC_INODE_LEN: usize = 128;

/// Ext4 file permission bits (lower 12 bits of `i_mode`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FilePerm(u16);

impl FilePerm {
    pub(super) fn bits(self) -> u16 {
        self.0
    }
}

/// An in-memory ext4 inode.
///
/// The type never changes once the inode exists, so it sits outside the lock
/// and costs nothing to read. Everything that a write can change lives in
/// `InodeInner`.
pub(crate) struct Inode {
    ino: Ext4Ino,
    type_: InodeType,
    inner: RwMutex<InodeInner>,
    /// The in-memory endpoint of a named pipe, which the format records as a
    /// type and nothing more.
    pipe: Option<Pipe>,
    fs: Weak<Ext4>,
    extension: Extension,
}

/// The mutable interior of an inode.
struct InodeInner {
    desc: Dirty<InodeDesc>,
    payload: InodePayload,
}

impl Debug for Inode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Inode")
            .field("ino", &self.ino)
            .field("type", &self.type_)
            .finish()
    }
}

impl Inode {
    /// Creates an `Inode` from a descriptor just read from the device.
    ///
    /// `volume_blocks` is passed rather than taken from `fs` because an inode
    /// is created while the filesystem is still being constructed, when the
    /// weak reference cannot yet be upgraded.
    pub(super) fn new(
        ino: Ext4Ino,
        desc: Dirty<InodeDesc>,
        volume_blocks: u64,
        fs: Weak<Ext4>,
    ) -> Result<Arc<Self>> {
        let payload = InodePayload::new(&desc, volume_blocks, fs.clone())?;
        let type_ = desc.type_;
        let pipe = (type_ == InodeType::NamedPipe).then(Pipe::new);

        Ok(Arc::new(Self {
            ino,
            type_,
            inner: RwMutex::new(InodeInner { desc, payload }),
            pipe,
            fs,
            extension: Extension::new(),
        }))
    }

    /// Returns the pipe behind a named-pipe inode.
    pub(super) fn pipe(&self) -> Option<&Pipe> {
        self.pipe.as_ref()
    }

    pub(super) fn ino(&self) -> Ext4Ino {
        self.ino
    }

    pub(super) fn inode_type(&self) -> InodeType {
        self.type_
    }

    pub(super) fn file_size(&self) -> usize {
        self.inner.read().desc.size as usize
    }

    pub(super) fn mode(&self) -> InodeMode {
        InodeMode::from_bits_truncate(self.inner.read().desc.perm.bits())
    }

    pub(super) fn uid(&self) -> u32 {
        self.inner.read().desc.uid
    }

    pub(super) fn gid(&self) -> u32 {
        self.inner.read().desc.gid
    }

    pub(super) fn atime(&self) -> Duration {
        self.inner.read().desc.atime
    }

    pub(super) fn mtime(&self) -> Duration {
        self.inner.read().desc.mtime
    }

    pub(super) fn ctime(&self) -> Duration {
        self.inner.read().desc.ctime
    }

    pub(super) fn link_count(&self) -> u16 {
        self.inner.read().desc.link_count
    }

    pub(super) fn extension(&self) -> &Extension {
        &self.extension
    }

    pub(super) fn fs(&self) -> Result<Arc<Ext4>> {
        self.fs
            .upgrade()
            .ok_or_else(|| Error::with_message(Errno::EIO, "filesystem already dropped"))
    }

    /// Returns whether the mount this inode belongs to may be changed.
    fn is_writable(&self) -> bool {
        self.fs().is_ok_and(|fs| fs.is_writable())
    }

    /// Sets the file permission bits.
    pub(super) fn set_mode(&self, mode: InodeMode) -> Result<()> {
        self.fs()?.check_writable()?;
        let mut inner = self.inner.write();
        inner.desc.perm = FilePerm(mode.bits() & 0o7777);
        inner.desc.ctime = now();
        Ok(())
    }

    /// Sets the owning user.
    pub(super) fn set_uid(&self, uid: u32) -> Result<()> {
        self.fs()?.check_writable()?;
        let mut inner = self.inner.write();
        inner.desc.uid = uid;
        inner.desc.ctime = now();
        Ok(())
    }

    /// Sets the owning group.
    pub(super) fn set_gid(&self, gid: u32) -> Result<()> {
        self.fs()?.check_writable()?;
        let mut inner = self.inner.write();
        inner.desc.gid = gid;
        inner.desc.ctime = now();
        Ok(())
    }

    /// Records a new access time, unless the mount forbids changing anything.
    ///
    /// The VFS declares these setters infallible, so a read-only mount can
    /// only ignore them.
    pub(super) fn set_atime(&self, time: Duration) {
        if self.is_writable() {
            self.inner.write().desc.atime = time;
        }
    }

    pub(super) fn set_mtime(&self, time: Duration) {
        if self.is_writable() {
            self.inner.write().desc.mtime = time;
        }
    }

    pub(super) fn set_ctime(&self, time: Duration) {
        if self.is_writable() {
            self.inner.write().desc.ctime = time;
        }
    }

    /// Adds `delta` to the number of names this file has.
    pub(super) fn adjust_link_count(&self, delta: i32) {
        let mut inner = self.inner.write();
        let count = inner.desc.link_count as i32 + delta;
        inner.desc.link_count = count.clamp(0, u16::MAX as i32) as u16;
        inner.desc.ctime = now();
    }

    /// Writes a symbolic link's target, inside the inode where it fits.
    pub(super) fn write_symlink_target(&self, target: &str) -> Result<()> {
        let fs = self.fs()?;
        let mut inner = self.inner.write();

        if target.len() < EXTENT_ROOT_LEN {
            let mut block = [0u8; EXTENT_ROOT_LEN];
            block[..target.len()].copy_from_slice(target.as_bytes());
            inner.desc.block = block;
            inner.desc.size = target.len() as u64;
            inner.payload = InodePayload::FastSymlink { target: block };
            return Ok(());
        }

        // A longer target needs a block, which means the inode needs the
        // extent tree that a data-backed payload carries.
        inner.desc.flags |= FileFlags::EXTENTS;
        inner.payload = InodePayload::new_data_backed(
            &inner.desc,
            fs.super_block().blocks_count,
            self.fs.clone(),
        )?;
        inner.resize(&fs, target.len())?;
        inner
            .page_cache_of_data()?
            .write_bytes(0, target.as_bytes())?;
        Ok(())
    }

    /// Writes the inode record back if anything changed it.
    pub(super) fn sync_metadata(&self) -> Result<()> {
        let mut inner = self.inner.write();
        // Writeback can have extended the extent tree since the last sync, and
        // the inode is what records where that tree now stands.
        inner.collect_block_changes();
        if !inner.desc.is_dirty() {
            return Ok(());
        }

        self.fs()?.write_back_inode(self.ino, &inner.desc)?;
        inner.desc.clear_dirty();
        Ok(())
    }

    /// Writes back cached data pages and then the inode record.
    pub(super) fn sync_all(&self) -> Result<()> {
        {
            let inner = self.inner.read();
            if let Some(page_cache) = inner.payload.page_cache() {
                page_cache.flush_range(0..page_cache.size())?;
            }
        }
        self.sync_metadata()
    }

    /// Records the device number of a character or block special file.
    pub(super) fn set_device_id(&self, device_id: u64) -> Result<()> {
        if self.type_ != InodeType::CharDevice && self.type_ != InodeType::BlockDevice {
            return_errno!(Errno::EINVAL);
        }

        let mut inner = self.inner.write();
        inner.desc.block = encode_device_id(device_id);
        inner.payload = InodePayload::Device { device_id };
        inner.desc.ctime = now();
        Ok(())
    }

    /// Returns the encoded device ID of a character or block special file.
    pub(super) fn device_id(&self) -> u64 {
        match &self.inner.read().payload {
            InodePayload::Device { device_id } => *device_id,
            _ => 0,
        }
    }

    pub(super) fn page_cache(&self) -> Option<Arc<Vmo>> {
        self.inner
            .read()
            .payload
            .page_cache()
            .map(PageCache::as_vmo)
            .cloned()
    }

    pub(super) fn metadata(&self) -> Metadata {
        let container_dev_id = match self.fs.upgrade() {
            Some(fs) => fs.block_device().id(),
            None => DeviceId::null(),
        };
        let self_dev_id = match self.type_ {
            InodeType::CharDevice | InodeType::BlockDevice => {
                DeviceId::from_encoded_u64(self.device_id())
            }
            _ => None,
        };

        let inner = self.inner.read();
        Metadata {
            ino: self.ino as u64,
            size: inner.desc.size as usize,
            optimal_block_size: BLOCK_SIZE,
            nr_sectors_allocated: inner.desc.sector_count as usize,
            last_access_at: inner.desc.atime,
            last_modify_at: inner.desc.mtime,
            last_meta_change_at: inner.desc.ctime,
            type_: self.type_,
            mode: InodeMode::from_bits_truncate(inner.desc.perm.bits()),
            nr_hard_links: inner.desc.link_count as usize,
            uid: Uid::new(inner.desc.uid),
            gid: Gid::new(inner.desc.gid),
            container_dev_id,
            self_dev_id,
            // Unlike ext2, ext4 records a creation time, when the inode is
            // large enough to hold one.
            birth_at: inner.desc.crtime,
        }
    }

    /// Reads file data at `offset` through the page cache.
    pub(super) fn read_at(&self, offset: usize, writer: &mut VmWriter) -> Result<usize> {
        if self.type_ == InodeType::Dir {
            return_errno!(Errno::EISDIR);
        }
        if writer.avail() == 0 {
            return Ok(0);
        }

        let inner = self.inner.read();
        let file_size = inner.desc.size as usize;
        if offset >= file_size {
            return Ok(0);
        }

        let read_len = writer.avail().min(file_size - offset);
        writer.limit(read_len);
        inner.page_cache_of_data()?.read(offset, writer)?;
        Ok(read_len)
    }

    /// Reads file data at `offset` straight from the device.
    pub(super) fn read_direct_at(&self, offset: usize, writer: &mut VmWriter) -> Result<usize> {
        if self.type_ == InodeType::Dir {
            return_errno!(Errno::EISDIR);
        }
        if !offset.is_multiple_of(BLOCK_SIZE) || !writer.avail().is_multiple_of(BLOCK_SIZE) {
            // TODO: Fall back to a buffered read for an unaligned request, as
            // Linux does.
            return_errno_with_message!(Errno::EINVAL, "not block-aligned");
        }

        let inner = self.inner.read();
        let file_size = inner.desc.size as usize;
        if offset >= file_size {
            return Ok(0);
        }
        let read_len = writer.avail().min(file_size - offset);
        if read_len == 0 {
            return Ok(0);
        }
        // Capping the writer is what keeps each run from writing past the end
        // of the file, since a run is measured in whole blocks.
        writer.limit(read_len);

        let fs = self.fs()?;
        let block_manager = inner.payload.block_manager()?;
        let iblock_range =
            (offset / BLOCK_SIZE) as Iblock..(offset + read_len).div_ceil(BLOCK_SIZE) as Iblock;
        let mut io_ranges = block_manager.iter_io_ranges(&fs, iblock_range);

        while let Some(io_range) = io_ranges.next()? {
            if writer.avail() == 0 {
                break;
            }
            match io_range {
                IoRange::Mapped(device_range) => {
                    let nblocks = (device_range.end - device_range.start) as usize;
                    let bio_segment = BioSegment::alloc(nblocks, BioDirection::FromDevice);
                    fs.read_blocks(device_range.start, bio_segment.clone())?;
                    bio_segment.reader()?.read_fallible(writer)?;
                }
                IoRange::Unwritten(range) | IoRange::Hole(range) => {
                    writer.fill_zeros((range.len() * BLOCK_SIZE).min(writer.avail()))?;
                }
            }
        }

        Ok(read_len)
    }

    /// Writes file data at `offset` through the page cache.
    pub(super) fn write_at(&self, offset: usize, reader: &mut VmReader) -> Result<usize> {
        if self.type_ == InodeType::Dir {
            return_errno!(Errno::EISDIR);
        }
        if reader.remain() == 0 {
            return Ok(0);
        }

        let fs = self.fs()?;
        fs.check_writable()?;
        self.inner.write().write_at(&fs, offset, reader)
    }

    /// Truncates or extends the file to `new_size` bytes.
    pub(super) fn resize(&self, new_size: usize) -> Result<()> {
        if self.type_ == InodeType::Dir {
            return_errno!(Errno::EISDIR);
        }
        if self.type_ != InodeType::File && self.type_ != InodeType::SymLink {
            return_errno!(Errno::EINVAL);
        }

        let fs = self.fs()?;
        fs.check_writable()?;
        self.inner.write().resize(&fs, new_size)
    }

    /// Reads the target of a symbolic link.
    pub(super) fn read_link(&self) -> Result<String> {
        if self.type_ != InodeType::SymLink {
            return_errno!(Errno::EINVAL);
        }

        let inner = self.inner.read();
        let len = inner.desc.size as usize;
        let bytes = match &inner.payload {
            InodePayload::FastSymlink { target } => target[..len].to_vec(),
            InodePayload::DataBacked { page_cache, .. } => {
                let mut bytes = vec![0u8; len];
                page_cache.read_bytes(0, &mut bytes)?;
                bytes
            }
            _ => return_errno_with_message!(Errno::EUCLEAN, "symlink has no target storage"),
        };

        String::from_utf8(bytes)
            .map_err(|_| Error::with_message(Errno::EUCLEAN, "symlink target is not UTF-8"))
    }
}

impl Drop for Inode {
    fn drop(&mut self) {
        // A file whose last name is gone is only freed once nothing holds it
        // open, which is what lets a program keep reading what it deleted.
        if self.inner.read().desc.link_count == 0 {
            if let Err(err) = self.reclaim() {
                error!("failed to free deleted inode {}: {:?}", self.ino, err);
            }
            return;
        }

        // Nothing else holds this inode, so a change that was never synced
        // would be lost here rather than merely delayed.
        if let Err(err) = self.sync_metadata() {
            error!("failed to write back inode {}: {:?}", self.ino, err);
        }
    }
}

impl Inode {
    /// Gives a deleted file's blocks and inode number back to the volume.
    fn reclaim(&self) -> Result<()> {
        let fs = self.fs()?;
        if !fs.is_writable() {
            return Ok(());
        }

        {
            let mut inner = self.inner.write();
            if let Ok(manager) = inner.payload.block_manager() {
                manager.clone().truncate(&fs, 0)?;
            }
            inner.desc.size = 0;
            inner.desc.dtime = now();
            inner.collect_block_changes();
        }

        self.sync_metadata()?;
        fs.free_ino(self.ino, self.type_ == InodeType::Dir)
    }
}

impl InodeInner {
    /// Writes `reader` into the file at `offset`, growing it if need be.
    fn write_at(&mut self, fs: &Ext4, offset: usize, reader: &mut VmReader) -> Result<usize> {
        let write_len = reader.remain();
        let end = offset
            .checked_add(write_len)
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "write range overflow"))?;
        let old_size = self.desc.size as usize;

        if end > old_size {
            self.resize_page_cache(end, old_size)?;
        }

        // Blocks are reserved before any data reaches the cache, so a volume
        // that is full leaves the file as it was rather than holding pages
        // that can never be written back.
        let manager = self.payload.block_manager()?.clone();
        let block_range = (offset / BLOCK_SIZE) as Iblock..end.div_ceil(BLOCK_SIZE) as Iblock;
        if let Err(err) = manager.allocate_range(fs, block_range) {
            self.collect_block_changes();
            if end > old_size {
                let _ = self.resize_page_cache(old_size, end);
            }
            return Err(err);
        }

        self.page_cache_of_data()?.write(offset, reader)?;

        if end > old_size {
            self.desc.size = end as u64;
        }
        self.set_modified(now());
        self.collect_block_changes();
        Ok(write_len)
    }

    /// Changes the file's length, freeing what a shorter one no longer needs.
    fn resize(&mut self, fs: &Ext4, new_size: usize) -> Result<()> {
        let old_size = self.desc.size as usize;
        if new_size == old_size {
            return Ok(());
        }

        self.resize_page_cache(new_size, old_size)?;
        if new_size < old_size {
            let manager = self.payload.block_manager()?.clone();
            manager.truncate(fs, new_size.div_ceil(BLOCK_SIZE) as Iblock)?;
        }

        self.desc.size = new_size as u64;
        self.set_modified(now());
        self.collect_block_changes();
        Ok(())
    }

    fn resize_page_cache(&mut self, new_size: usize, old_size: usize) -> Result<()> {
        let InodePayload::DataBacked {
            page_cache,
            block_manager,
        } = &mut self.payload
        else {
            return_errno_with_message!(Errno::EINVAL, "inode has no data page cache");
        };

        page_cache.resize(new_size, old_size)?;
        // The backend refuses I/O past what it believes the file holds, so it
        // has to learn the new length before anything reaches it.
        block_manager.set_npages(new_size.div_ceil(PAGE_SIZE));
        Ok(())
    }

    fn set_modified(&mut self, time: Duration) {
        self.desc.mtime = time;
        self.desc.ctime = time;
    }

    /// Folds what the block manager changed into the descriptor.
    ///
    /// The tree is rooted in the inode's own bytes, and writeback can extend
    /// it with no inode at hand, so the inode collects the change rather than
    /// being told about it.
    fn collect_block_changes(&mut self) {
        let Ok(manager) = self.payload.block_manager() else {
            return;
        };
        if !manager.is_dirty() {
            return;
        }

        let (root, sectors) = manager.clone().take_changes();
        self.desc.block = root;
        self.desc.sector_count = self.desc.sector_count.saturating_add_signed(sectors);
    }

    fn page_cache_of_data(&self) -> Result<&PageCache> {
        self.payload
            .page_cache()
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "inode has no data page cache"))
    }
}

/// Returns the current time, as the inode timestamps record it.
pub(super) fn now() -> Duration {
    crate::time::clocks::RealTimeCoarseClock::get().read_time()
}

/// Decoded in-memory mirror of an on-disk inode's metadata fields.
#[derive(Clone, Copy, Debug)]
pub(super) struct InodeDesc {
    type_: InodeType,
    perm: FilePerm,
    uid: u32,
    gid: u32,
    size: u64,
    atime: Duration,
    ctime: Duration,
    mtime: Duration,
    crtime: Option<Duration>,
    link_count: u16,
    /// When the file was deleted, which is zero for one that still exists.
    dtime: Duration,
    /// Allocated storage in 512-byte sectors, as `stat` reports it.
    sector_count: u64,
    flags: FileFlags,
    file_acl: u64,
    block: [u8; EXTENT_ROOT_LEN],
}

impl InodeDesc {
    /// Builds the descriptor of a file that is about to exist.
    pub(super) fn new(
        type_: InodeType,
        mode: InodeMode,
        uid: u32,
        gid: u32,
        link_count: u16,
        now: Duration,
    ) -> Self {
        Self {
            type_,
            perm: FilePerm(mode.bits() & 0o7777),
            uid,
            gid,
            size: 0,
            atime: now,
            ctime: now,
            mtime: now,
            crtime: Some(now),
            link_count,
            dtime: Duration::ZERO,
            sector_count: 0,
            // Every file this driver creates is mapped by extents, which is
            // the only layout it writes.
            flags: FileFlags::EXTENTS,
            file_acl: 0,
            block: ExtentTree::empty(0).root_bytes(),
        }
    }

    /// Decodes an on-disk inode.
    ///
    /// `inode_size` decides how much of the record exists: the fields past the
    /// classic 128-byte area are only present on a larger inode, and only as
    /// far as `i_extra_isize` says.
    pub(super) fn from_raw(raw: &RawInode, inode_size: usize, ro_compat: u32) -> Result<Self> {
        if raw.link_count == 0 {
            return_errno_with_message!(Errno::ESTALE, "inode has been deleted");
        }

        let type_ = InodeType::from_raw_mode(raw.mode)?;
        let perm = FilePerm(raw.mode & 0o7777);
        let flags = FileFlags::from_bits_truncate(raw.flags);

        if flags.contains(FileFlags::INLINE_DATA) {
            return_errno_with_message!(Errno::EOPNOTSUPP, "inline data is not supported");
        }
        if flags.contains(FileFlags::ENCRYPT) {
            return_errno_with_message!(Errno::EOPNOTSUPP, "encrypted files are not supported");
        }

        let extra = ExtraFields::of(raw, inode_size);
        let mut size = raw.size_lo as u64;
        // The upper half of the size is the historical `i_dir_acl`, so only a
        // regular file is allowed to be larger than 4 GiB.
        if type_ == InodeType::File {
            size |= (raw.size_high as u64) << 32;
        }
        if size > i64::MAX as u64 {
            return_errno_with_message!(Errno::EUCLEAN, "corrupted inode on disk");
        }
        if type_ == InodeType::SymLink && size >= BLOCK_SIZE as u64 {
            return_errno_with_message!(Errno::EUCLEAN, "corrupted symlink on disk");
        }

        Ok(Self {
            type_,
            perm,
            uid: (raw.uid as u32) | ((raw.uid_high as u32) << 16),
            gid: (raw.gid as u32) | ((raw.gid_high as u32) << 16),
            size,
            atime: decode_time(raw.atime, extra.atime_extra),
            ctime: decode_time(raw.ctime, extra.ctime_extra),
            mtime: decode_time(raw.mtime, extra.mtime_extra),
            crtime: extra
                .crtime
                .map(|secs| decode_time(secs, extra.crtime_extra)),
            link_count: raw.link_count,
            dtime: Duration::from_secs(raw.dtime as u64),
            sector_count: decode_sector_count(raw, flags, ro_compat),
            flags,
            file_acl: (raw.file_acl_lo as u64) | ((raw.file_acl_high as u64) << 32),
            block: raw.block,
        })
    }

    /// Returns whether a symlink stores its target inside the inode.
    ///
    /// An extended attribute block is the only storage such an inode may own,
    /// so anything beyond it means the target lives in a data block.
    fn is_fast_symlink(&self) -> bool {
        let xattr_sectors = if self.file_acl == 0 {
            0
        } else {
            (BLOCK_SIZE / SECTOR_SIZE) as u64
        };

        self.sector_count == xattr_sectors && self.size < EXTENT_ROOT_LEN as u64
    }

    /// Overwrites the fields this driver owns in an on-disk inode record.
    ///
    /// `raw` must be the record as it currently stands on the device. Writing
    /// into it rather than building one from scratch is what preserves the
    /// bytes this driver does not understand: the inline extended attributes
    /// past the classic area, the checksum halves, and the version counters.
    ///
    /// `inode_size` decides which of the fields ext4 added exist at all; a
    /// nanosecond field that the inode is too small to hold is left alone, and
    /// the timestamp keeps whole-second resolution.
    pub(super) fn write_into(&self, raw: &mut RawInode, inode_size: usize, ro_compat: u32) {
        raw.mode = (self.type_ as u16) | (self.perm.bits() & 0o7777);
        raw.uid = self.uid as u16;
        raw.uid_high = (self.uid >> 16) as u16;
        raw.gid = self.gid as u16;
        raw.gid_high = (self.gid >> 16) as u16;
        raw.link_count = self.link_count;
        raw.dtime = self.dtime.as_secs() as u32;
        raw.flags = self.flags.bits();
        raw.file_acl_lo = self.file_acl as u32;
        raw.file_acl_high = (self.file_acl >> 32) as u16;
        raw.block = self.block;

        raw.size_lo = self.size as u32;
        // The upper half is the historical `i_dir_acl`, and only a regular
        // file is allowed to use it as a size.
        if self.type_ == InodeType::File {
            raw.size_high = (self.size >> 32) as u32;
        }

        encode_sector_count(raw, self.sector_count, self.flags, ro_compat);

        let present = ExtraFields::of(raw, inode_size);
        let write_time = |secs: &mut u32, extra: &mut u32, was_present: bool, time: Duration| {
            let (encoded_secs, encoded_extra) = encode_time(time);
            *secs = encoded_secs;
            if was_present {
                *extra = encoded_extra;
            }
        };
        write_time(
            &mut raw.atime,
            &mut raw.atime_extra,
            present.atime_extra.is_some(),
            self.atime,
        );
        write_time(
            &mut raw.mtime,
            &mut raw.mtime_extra,
            present.mtime_extra.is_some(),
            self.mtime,
        );
        write_time(
            &mut raw.ctime,
            &mut raw.ctime_extra,
            present.ctime_extra.is_some(),
            self.ctime,
        );
        if let Some(crtime) = self.crtime
            && present.crtime.is_some()
        {
            write_time(
                &mut raw.crtime,
                &mut raw.crtime_extra,
                present.crtime_extra.is_some(),
                crtime,
            );
        }
    }
}

/// Encodes a timestamp into its seconds field and its extra field.
///
/// The inverse of [`decode_time`]: the extra field carries nanoseconds in its
/// upper 30 bits and the two seconds bits past 2038 in its lower two.
fn encode_time(time: Duration) -> (u32, u32) {
    let secs = time.as_secs();
    let epoch_hi = ((secs >> 32) & 0x3) as u32;

    (secs as u32, (time.subsec_nanos() << 2) | epoch_hi)
}

/// Writes a 512-byte sector count back into `i_blocks`.
///
/// The inverse of [`decode_sector_count`]. A count that does not fit the
/// 48-bit field is recorded in filesystem blocks instead, which is what the
/// `HUGE_FILE` inode flag means.
fn encode_sector_count(raw: &mut RawInode, sectors: u64, flags: FileFlags, ro_compat: u32) {
    use super::features::FeatureRoCompat;

    if (ro_compat & FeatureRoCompat::HUGE_FILE.bits()) == 0 {
        raw.blocks_lo = sectors as u32;
        return;
    }

    let count = if flags.contains(FileFlags::HUGE_FILE) {
        sectors / (BLOCK_SIZE / SECTOR_SIZE) as u64
    } else {
        sectors
    };
    raw.blocks_lo = count as u32;
    raw.blocks_high = (count >> 32) as u16;
}

/// Decodes an ext4 timestamp.
///
/// The classic field holds seconds since the epoch. Where the inode is large
/// enough, a second field adds nanoseconds in its upper 30 bits and two more
/// seconds bits in its lower two, which is what pushes the format past 2038.
fn decode_time(secs: u32, extra: Option<u32>) -> Duration {
    let extra = extra.unwrap_or(0);
    let secs = (secs as u64) | (((extra & 0x3) as u64) << 32);
    let nanos = extra >> 2;

    // A corrupt field must not panic `Duration::new`, which overflows when the
    // nanoseconds carry into a second beyond the range of `u64`.
    Duration::new(secs, if nanos < 1_000_000_000 { nanos } else { 0 })
}

/// Decodes `i_blocks` into 512-byte sectors.
///
/// Without the `HUGE_FILE` feature the field is a 32-bit sector count. With
/// it, the count widens to 48 bits, and an inode that sets `HUGE_FILE` counts
/// in filesystem blocks instead so that it can describe a file larger than
/// `2^48` sectors.
fn decode_sector_count(raw: &RawInode, flags: FileFlags, ro_compat: u32) -> u64 {
    use super::features::FeatureRoCompat;

    if (ro_compat & FeatureRoCompat::HUGE_FILE.bits()) == 0 {
        return raw.blocks_lo as u64;
    }

    let count = (raw.blocks_lo as u64) | ((raw.blocks_high as u64) << 32);
    if flags.contains(FileFlags::HUGE_FILE) {
        count * (BLOCK_SIZE / SECTOR_SIZE) as u64
    } else {
        count
    }
}

/// The fields past the classic 128-byte inode area, as far as this inode holds
/// them.
struct ExtraFields {
    ctime_extra: Option<u32>,
    mtime_extra: Option<u32>,
    atime_extra: Option<u32>,
    crtime: Option<u32>,
    crtime_extra: Option<u32>,
}

impl ExtraFields {
    /// Reads the fields that `i_extra_isize` declares present.
    ///
    /// A field counts as present only when it ends within the declared area,
    /// which is how Linux decides, and the declared area can never exceed the
    /// inode itself.
    fn of(raw: &RawInode, inode_size: usize) -> Self {
        let available = if inode_size <= CLASSIC_INODE_LEN {
            0
        } else {
            (raw.extra_isize as usize).min(inode_size - CLASSIC_INODE_LEN)
        };
        let present = |end_offset: usize, value: u32| {
            (end_offset <= CLASSIC_INODE_LEN + available).then_some(value)
        };

        Self {
            ctime_extra: present(136, raw.ctime_extra),
            mtime_extra: present(140, raw.mtime_extra),
            atime_extra: present(144, raw.atime_extra),
            crtime: present(148, raw.crtime),
            crtime_extra: present(152, raw.crtime_extra),
        }
    }
}

/// Type-specific state backed by the inode's 60-byte `i_block` area.
#[derive(Debug)]
enum InodePayload {
    /// Regular files, directories, and symlinks whose target needs a block.
    DataBacked {
        page_cache: PageCache,
        block_manager: Arc<InodeBlockManager>,
    },
    /// A symlink target short enough to live inside the inode.
    FastSymlink { target: [u8; EXTENT_ROOT_LEN] },
    /// A character or block device number encoded in `i_block`.
    Device { device_id: u64 },
    /// Inode types that carry no payload at all.
    NoPayload,
}

impl InodePayload {
    fn new(desc: &InodeDesc, volume_blocks: u64, fs: Weak<Ext4>) -> Result<Self> {
        match desc.type_ {
            InodeType::File | InodeType::Dir => Self::new_data_backed(desc, volume_blocks, fs),
            InodeType::SymLink if desc.is_fast_symlink() => {
                Ok(Self::FastSymlink { target: desc.block })
            }
            InodeType::SymLink => Self::new_data_backed(desc, volume_blocks, fs),
            InodeType::CharDevice | InodeType::BlockDevice => Ok(Self::Device {
                device_id: decode_device_id(&desc.block),
            }),
            _ => Ok(Self::NoPayload),
        }
    }

    pub(super) fn new_data_backed(
        desc: &InodeDesc,
        volume_blocks: u64,
        fs: Weak<Ext4>,
    ) -> Result<Self> {
        // Without the extent flag the file uses the ext2 block-pointer layout,
        // which ext4 still permits and this driver does not read.
        if !desc.flags.contains(FileFlags::EXTENTS) {
            return_errno_with_message!(
                Errno::EOPNOTSUPP,
                "file uses the block-pointer layout rather than extents"
            );
        }

        let extents = ExtentTree::new(desc.block, volume_blocks)?;
        let page_cache_size = (desc.size as usize).align_up(PAGE_SIZE);
        let block_manager = Arc::new(InodeBlockManager::new(
            extents,
            fs,
            page_cache_size / PAGE_SIZE,
        ));
        let backend: Weak<dyn PageCacheBackend> = Arc::downgrade(&block_manager) as _;
        let page_cache = PageCache::new_with_backend(page_cache_size, backend)?;

        Ok(Self::DataBacked {
            page_cache,
            block_manager,
        })
    }

    fn page_cache(&self) -> Option<&PageCache> {
        match self {
            Self::DataBacked { page_cache, .. } => Some(page_cache),
            _ => None,
        }
    }

    fn block_manager(&self) -> Result<&Arc<InodeBlockManager>> {
        match self {
            Self::DataBacked { block_manager, .. } => Ok(block_manager),
            _ => Err(Error::with_message(
                Errno::EINVAL,
                "inode payload has no block manager",
            )),
        }
    }
}

/// Encodes a device number into the `i_block` area a special file uses.
///
/// The historical form holds eight bits of each number in the first word; a
/// number too large for it goes in the second, spread out.
fn encode_device_id(device_id: u64) -> [u8; EXTENT_ROOT_LEN] {
    let (major, minor) = decode_device_numbers(device_id);

    let mut block = [0u8; EXTENT_ROOT_LEN];
    if major < 256 && minor < 256 {
        block[0..4].copy_from_slice(&((major << 8) | minor).to_le_bytes());
    } else {
        let encoded = (minor & 0xff) | ((major & 0xfff) << 8) | ((minor >> 8) << 20);
        block[4..8].copy_from_slice(&encoded.to_le_bytes());
    }
    block
}

/// Decodes the device number a special file stores in its `i_block` area.
fn decode_device_id(block: &[u8; EXTENT_ROOT_LEN]) -> u64 {
    // Destructuring a fixed-size array keeps the decode free of a fallible
    // conversion that could never actually fail.
    let [b0, b1, b2, b3, b4, b5, b6, b7, ..] = *block;
    let first = u32::from_le_bytes([b0, b1, b2, b3]);
    let second = u32::from_le_bytes([b4, b5, b6, b7]);

    let (major, minor) = if first != 0 {
        // The historical encoding: eight bits of each.
        ((first >> 8) & 0xff, first & 0xff)
    } else {
        (
            (second & 0xfff00) >> 8,
            (second & 0xff) | ((second >> 12) & 0xfff00),
        )
    };

    encode_device_numbers(major, minor)
}

/// The on-disk inode structure.
///
/// The first 128 bytes are the classic layout that ext2 also uses; the rest
/// exists only on a volume with larger inodes, and then only as far as
/// `extra_isize` declares.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
pub(super) struct RawInode {
    pub mode: u16,
    pub uid: u16,
    pub size_lo: u32,
    pub atime: u32,
    pub ctime: u32,
    pub mtime: u32,
    pub dtime: u32,
    pub gid: u16,
    pub link_count: u16,
    pub blocks_lo: u32,
    pub flags: u32,
    pub version_lo: u32,
    /// `i_block`: the extent tree root, a fast symlink target, or a device
    /// number, depending on the inode type.
    pub block: [u8; EXTENT_ROOT_LEN],
    pub generation: u32,
    pub file_acl_lo: u32,
    pub size_high: u32,
    pub obso_faddr: u32,
    pub blocks_high: u16,
    pub file_acl_high: u16,
    pub uid_high: u16,
    pub gid_high: u16,
    pub checksum_lo: u16,
    pub reserved: u16,
    pub extra_isize: u16,
    pub checksum_hi: u16,
    pub ctime_extra: u32,
    pub mtime_extra: u32,
    pub atime_extra: u32,
    pub crtime: u32,
    pub crtime_extra: u32,
    pub version_hi: u32,
    pub projid: u32,
}

const_assert!(size_of::<RawInode>() == 160);

bitflags! {
    /// `i_flags`, the per-inode behaviour and layout flags.
    pub(super) struct FileFlags: u32 {
        const SECURE_DEL = 1 << 0;
        const UNDELETE = 1 << 1;
        const COMPRESS = 1 << 2;
        const SYNC_UPDATE = 1 << 3;
        const IMMUTABLE = 1 << 4;
        const APPEND_ONLY = 1 << 5;
        const NO_DUMP = 1 << 6;
        const NO_ATIME = 1 << 7;
        const DIRTY = 1 << 8;
        const COMPRESS_BLK = 1 << 9;
        const NO_COMPRESS = 1 << 10;
        const ENCRYPT = 1 << 11;
        /// The directory is hash-indexed.
        const INDEX_DIR = 1 << 12;
        const IMAGIC = 1 << 13;
        const JOURNAL_DATA = 1 << 14;
        const NO_TAIL = 1 << 15;
        const DIR_SYNC = 1 << 16;
        const TOP_DIR = 1 << 17;
        /// `i_blocks` counts filesystem blocks rather than sectors.
        const HUGE_FILE = 1 << 18;
        /// The file is mapped by an extent tree rather than block pointers.
        const EXTENTS = 1 << 19;
        const VERITY = 1 << 20;
        /// The inode holds one extended attribute value rather than a file.
        const EA_INODE = 1 << 21;
        const EOF_BLOCKS = 1 << 22;
        const SNAPFILE = 1 << 24;
        const SNAPFILE_DELETED = 1 << 26;
        const SNAPFILE_SHRUNK = 1 << 27;
        /// File contents are stored inside the inode.
        const INLINE_DATA = 1 << 28;
        const PROJ_INHERIT = 1 << 29;
        const CASEFOLD = 1 << 30;
        const RESERVED = 1 << 31;
    }
}

#[cfg(ktest)]
mod test {
    use device_id::{MajorId, MinorId};
    use ostd::prelude::ktest;

    use super::*;
    use crate::fs::{
        fs_impls::ext4::{
            fs,
            test_utils::{
                BAD_EXTENT_INO, DEEP_FILL, DEEP_INO, DEVICE_INO, Ext4VolumeBuilder, HELLO_CONTENT,
                HELLO_INO, ROOT_ENTRIES, SPARSE_BLOCKS, SPARSE_FILL, SPARSE_INO,
                SPARSE_TAIL_IBLOCK, SYMLINK_INO, SYMLINK_TARGET, TWO_EXTENT_FILL, TWO_EXTENT_INO,
                assert_errno,
            },
        },
        vfs::file_system::FsFlags,
    };

    /// Mounts a freshly built volume and returns it.
    fn mount() -> Arc<Ext4> {
        let volume = Ext4VolumeBuilder::new().build();
        let fs = fs::open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        fs.downcast_ref::<Ext4>()
            .expect("should be an ext4 volume")
            .root_inode()
            .expect("root should be readable")
            .fs()
            .expect("the filesystem should outlive its root")
    }

    fn read_whole(inode: &Inode) -> Vec<u8> {
        let mut buf = vec![0u8; inode.file_size()];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        let read = inode.read_at(0, &mut writer).expect("should read");

        assert_eq!(read, buf.len());
        buf
    }

    /// Returns what a file's blocks hold, one byte per logical block.
    fn block_fills(content: &[u8]) -> Vec<u8> {
        content
            .chunks(BLOCK_SIZE)
            .map(|block| {
                assert!(
                    block.iter().all(|&byte| byte == block[0]),
                    "the fixture fills each block with one byte"
                );
                block[0]
            })
            .collect()
    }

    #[ktest]
    fn a_file_in_one_extent_reads_back() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        assert_eq!(inode.file_size(), HELLO_CONTENT.len());
        assert_eq!(read_whole(&inode), HELLO_CONTENT);
    }

    /// Two extents are read as two requests, and getting their order or their
    /// physical addresses wrong swaps the contents rather than failing.
    #[ktest]
    fn a_file_in_two_extents_reads_back_in_order() {
        let ext4 = mount();
        let inode = ext4.read_inode(TWO_EXTENT_INO).expect("should read");

        assert_eq!(block_fills(&read_whole(&inode)), TWO_EXTENT_FILL);
    }

    #[ktest]
    fn a_file_behind_an_index_node_reads_back() {
        let ext4 = mount();
        let inode = ext4.read_inode(DEEP_INO).expect("should read");

        assert_eq!(block_fills(&read_whole(&inode)), DEEP_FILL);
    }

    #[ktest]
    fn a_hole_reads_as_zeros() {
        let ext4 = mount();
        let inode = ext4.read_inode(SPARSE_INO).expect("should read");

        let mut expected = vec![0u8; SPARSE_BLOCKS];
        expected[0] = SPARSE_FILL;
        expected[SPARSE_TAIL_IBLOCK] = SPARSE_FILL;
        assert_eq!(block_fills(&read_whole(&inode)), expected);
    }

    /// A direct read bypasses the page cache entirely, so it is a second,
    /// independent route through the extent tree.
    #[ktest]
    fn a_direct_read_agrees_with_a_cached_one() {
        let ext4 = mount();
        let inode = ext4.read_inode(SPARSE_INO).expect("should read");

        let mut direct = vec![0u8; inode.file_size()];
        let mut writer = VmWriter::from(direct.as_mut_slice()).to_fallible();
        inode.read_direct_at(0, &mut writer).expect("should read");

        assert_eq!(direct, read_whole(&inode));
    }

    #[ktest]
    fn a_fast_symlink_target_is_read_from_the_inode() {
        let ext4 = mount();
        let inode = ext4.read_inode(SYMLINK_INO).expect("should read");

        assert_eq!(inode.inode_type(), InodeType::SymLink);
        assert_eq!(inode.read_link().expect("should read"), SYMLINK_TARGET);
    }

    #[ktest]
    fn the_root_directory_lists_its_entries() {
        let ext4 = mount();
        let root = ext4.root_inode().expect("should read");

        let mut names: Vec<String> = Vec::new();
        root.readdir_at(0, &mut names).expect("should list");

        let expected: Vec<String> = ROOT_ENTRIES
            .iter()
            .map(|&(name, _, _)| name.to_string())
            .collect();
        assert_eq!(names, expected);
    }

    /// Resuming from a previous offset must neither repeat an entry nor skip
    /// one, which is what a reader doing a short `getdents` depends on.
    #[ktest]
    fn readdir_resumes_where_it_stopped() {
        let ext4 = mount();
        let root = ext4.root_inode().expect("should read");

        let mut first: Vec<String> = Vec::new();
        let advanced = root
            .readdir_at(0, &mut OneEntry(&mut first))
            .expect("should list");
        let mut rest: Vec<String> = Vec::new();
        root.readdir_at(advanced, &mut rest).expect("should list");

        first.extend(rest);
        let expected: Vec<String> = ROOT_ENTRIES
            .iter()
            .map(|&(name, _, _)| name.to_string())
            .collect();
        assert_eq!(first, expected);
    }

    /// A visitor that accepts one entry and then refuses, as a caller with a
    /// full buffer does.
    struct OneEntry<'a>(&'a mut Vec<String>);

    impl DirentVisitor for OneEntry<'_> {
        fn visit(
            &mut self,
            name: &str,
            _ino: u64,
            _type_: InodeType,
            _offset: usize,
        ) -> Result<()> {
            if !self.0.is_empty() {
                return_errno!(Errno::EINVAL);
            }
            self.0.push(name.to_string());
            Ok(())
        }
    }

    #[ktest]
    fn a_name_resolves_to_the_inode_it_names() {
        let ext4 = mount();
        let root = ext4.root_inode().expect("should read");

        let hello = root.lookup("hello").expect("should resolve");

        assert_eq!(hello.ino(), HELLO_INO);
        assert_eq!(read_whole(&hello), HELLO_CONTENT);
    }

    #[ktest]
    fn an_absent_name_reports_enoent() {
        let ext4 = mount();
        let root = ext4.root_inode().expect("should read");

        assert_errno(root.lookup("nothing-here"), Errno::ENOENT);
    }

    /// A corrupt extent root must be refused when the inode is read, so that
    /// nothing downstream has to cope with a tree it cannot walk.
    #[ktest]
    fn an_inode_with_a_corrupt_extent_root_is_refused() {
        let ext4 = mount();

        assert_errno(ext4.read_inode(BAD_EXTENT_INO), Errno::EUCLEAN);
    }

    #[ktest]
    fn a_special_file_reports_its_device_number() {
        let ext4 = mount();
        let inode = ext4.read_inode(DEVICE_INO).expect("should read");

        let metadata = inode.metadata();
        assert_eq!(metadata.type_, InodeType::CharDevice);
        let device_id = metadata.self_dev_id.expect("a device file has a device ID");
        assert_eq!(device_id, DeviceId::new(MajorId::new(1), MinorId::new(3)));
    }

    /// Reads that start and end inside a block are where an implementation
    /// that works in whole blocks quietly returns the wrong bytes.
    #[ktest]
    fn a_read_inside_one_block_returns_only_what_was_asked_for() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        let mut buf = [0u8; 5];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        let read = inode.read_at(7, &mut writer).expect("should read");

        assert_eq!(read, 5);
        assert_eq!(&buf, &HELLO_CONTENT[7..12]);
    }

    /// A read that spans two logical blocks must join them in order.
    #[ktest]
    fn a_read_across_a_block_boundary_joins_both_blocks() {
        let ext4 = mount();
        let inode = ext4.read_inode(TWO_EXTENT_INO).expect("should read");

        let mut buf = [0u8; 4];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        let read = inode
            .read_at(BLOCK_SIZE - 2, &mut writer)
            .expect("should read");

        assert_eq!(read, 4);
        assert_eq!(
            buf,
            [
                TWO_EXTENT_FILL[0],
                TWO_EXTENT_FILL[0],
                TWO_EXTENT_FILL[1],
                TWO_EXTENT_FILL[1]
            ]
        );
    }

    /// A read that runs past the end stops there, rather than returning the
    /// slack that the last block still holds.
    #[ktest]
    fn a_read_is_clipped_to_the_end_of_the_file() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        let mut buf = [0xffu8; 64];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        let read = inode.read_at(0, &mut writer).expect("should read");

        assert_eq!(read, HELLO_CONTENT.len());
        assert_eq!(&buf[..read], HELLO_CONTENT);
        assert_eq!(buf[read], 0xff, "nothing past the end should be touched");
    }

    #[ktest]
    fn a_read_starting_past_the_end_returns_nothing() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        let mut buf = [0u8; 8];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();

        assert_eq!(
            inode
                .read_at(HELLO_CONTENT.len(), &mut writer)
                .expect("should read"),
            0
        );
    }

    #[ktest]
    fn a_read_of_no_bytes_returns_nothing() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        let mut buf: [u8; 0] = [];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();

        assert_eq!(inode.read_at(0, &mut writer).expect("should read"), 0);
    }

    /// Reading from inside the hole of a sparse file yields zeros without
    /// consulting the device.
    #[ktest]
    fn a_read_that_starts_inside_a_hole_yields_zeros() {
        let ext4 = mount();
        let inode = ext4.read_inode(SPARSE_INO).expect("should read");

        let mut buf = [0xffu8; 16];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        let read = inode
            .read_at(2 * BLOCK_SIZE, &mut writer)
            .expect("should read");

        assert_eq!(read, buf.len());
        assert!(buf.iter().all(|&byte| byte == 0));
    }

    /// A direct read works in whole blocks, so an unaligned one is refused
    /// rather than quietly served from the wrong offset.
    #[ktest]
    fn an_unaligned_direct_read_is_refused() {
        let ext4 = mount();
        let inode = ext4.read_inode(TWO_EXTENT_INO).expect("should read");

        let mut buf = [0u8; BLOCK_SIZE];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();

        assert_errno(inode.read_direct_at(1, &mut writer), Errno::EINVAL);
    }

    /// A 128-byte inode stops before the fields ext4 added, so it carries no
    /// creation time and no nanoseconds.
    #[ktest]
    fn a_classic_sized_inode_has_no_creation_time() {
        let volume = Ext4VolumeBuilder::new().inode_size(128).build();
        let fs = fs::open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs
            .downcast_ref::<Ext4>()
            .expect("should be an ext4 volume")
            .root_inode()
            .expect("root should be readable")
            .fs()
            .expect("the filesystem should outlive its root");

        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        assert_eq!(inode.metadata().birth_at, None);
        assert_eq!(read_whole(&inode), HELLO_CONTENT);
    }

    /// The whole-second part of a timestamp survives a volume whose inodes are
    /// too small to hold the nanosecond half.
    #[ktest]
    fn a_classic_sized_inode_keeps_its_whole_second_timestamps() {
        let volume = Ext4VolumeBuilder::new().inode_size(128).build();
        let fs = fs::open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs
            .downcast_ref::<Ext4>()
            .expect("should be an ext4 volume")
            .root_inode()
            .expect("root should be readable")
            .fs()
            .expect("the filesystem should outlive its root");

        let metadata = ext4.read_inode(HELLO_INO).expect("should read").metadata();

        assert_eq!(metadata.last_access_at.subsec_nanos(), 0);
        assert!(metadata.last_access_at.as_secs() > 0);
    }

    /// Writes within a file that already has the blocks must reach the device
    /// through the page cache and come back.
    #[ktest]
    fn a_write_inside_an_existing_file_reads_back() {
        let ext4 = mount();
        let inode = ext4.read_inode(TWO_EXTENT_INO).expect("should read");

        let payload = [0x5au8; 8];
        let mut reader = VmReader::from(payload.as_slice()).to_fallible();
        let written = inode.write_at(16, &mut reader).expect("should write");

        assert_eq!(written, payload.len());
        let mut buf = [0u8; 8];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        inode.read_at(16, &mut writer).expect("should read");
        assert_eq!(buf, payload);
    }

    /// A write past the end grows the file, which needs a block that nothing
    /// had allocated and an extent that nothing had recorded.
    #[ktest]
    fn a_write_past_the_end_grows_the_file() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");
        let old_sectors = inode.metadata().nr_sectors_allocated;

        let payload = [0x77u8; 32];
        let mut reader = VmReader::from(payload.as_slice()).to_fallible();
        inode
            .write_at(2 * BLOCK_SIZE, &mut reader)
            .expect("should write");

        assert_eq!(inode.file_size(), 2 * BLOCK_SIZE + payload.len());
        assert!(
            inode.metadata().nr_sectors_allocated > old_sectors,
            "growing a file must account for the blocks it took"
        );

        let mut buf = [0u8; 32];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        inode
            .read_at(2 * BLOCK_SIZE, &mut writer)
            .expect("should read");
        assert_eq!(buf, payload);
    }

    /// The bytes between the old end and a write past it read as zeros, as a
    /// hole does, rather than as whatever the block held before.
    #[ktest]
    fn the_gap_a_growing_write_leaves_reads_as_zeros() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        let payload = [0x11u8; 4];
        let mut reader = VmReader::from(payload.as_slice()).to_fallible();
        inode.write_at(64, &mut reader).expect("should write");

        let mut buf = [0xffu8; 16];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        inode.read_at(32, &mut writer).expect("should read");
        assert!(buf.iter().all(|&byte| byte == 0));
    }

    /// A file written in one go should cost one extent, which is the whole
    /// point of a format that records runs rather than blocks.
    #[ktest]
    fn a_file_written_in_one_go_costs_one_extent() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");
        // Starting from nothing, so that the run is chosen rather than
        // constrained by what the file already held.
        inode.resize(0).expect("should truncate");

        let payload = vec![0x2bu8; 4 * BLOCK_SIZE];
        let mut reader = VmReader::from(payload.as_slice()).to_fallible();
        inode.write_at(0, &mut reader).expect("should write");

        assert_eq!(inode.file_size(), 4 * BLOCK_SIZE);
        let inner = inode.inner.read();
        let manager = inner
            .payload
            .block_manager()
            .expect("a regular file has a block manager");
        let fs = inode.fs().expect("the filesystem is alive");
        // One run covering the whole file is what a single extent looks like
        // from the outside.
        let mut ranges = manager.iter_io_ranges(&fs, 0..4);
        let first = ranges.next().expect("should map").expect("one run");
        assert!(
            matches!(first, IoRange::Mapped(ref range) if range.end - range.start == 4),
            "a sequential write should produce one extent, got {:?}",
            first
        );
    }

    /// Shortening a file must give its blocks back, or a volume leaks space
    /// every time a file is rewritten.
    #[ktest]
    fn truncating_a_file_returns_its_blocks() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        let block = vec![0x3cu8; BLOCK_SIZE];
        for index in 0..4 {
            let mut reader = VmReader::from(block.as_slice()).to_fallible();
            inode
                .write_at(index * BLOCK_SIZE, &mut reader)
                .expect("should write");
        }
        let grown_sectors = inode.metadata().nr_sectors_allocated;
        let free_before = ext4.free_blocks_count();

        inode.resize(0).expect("should truncate");

        assert_eq!(inode.file_size(), 0);
        assert!(inode.metadata().nr_sectors_allocated < grown_sectors);
        assert!(
            ext4.free_blocks_count() > free_before,
            "truncating must return blocks to the volume"
        );
    }

    /// What a write changed has to survive being written back and read from
    /// the device again, which is the only thing that proves it was recorded.
    #[ktest]
    fn a_written_file_survives_being_read_from_the_device_again() {
        let volume = Ext4VolumeBuilder::new().build();
        let fs = fs::open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount");
        let ext4 = fs
            .downcast_ref::<Ext4>()
            .expect("should be an ext4 volume")
            .root_inode()
            .expect("root should be readable")
            .fs()
            .expect("the filesystem should outlive its root");

        let payload = vec![0x6du8; BLOCK_SIZE + 7];
        {
            let inode = ext4.read_inode(HELLO_INO).expect("should read");
            let mut reader = VmReader::from(payload.as_slice()).to_fallible();
            inode.write_at(0, &mut reader).expect("should write");
            inode.sync_all().expect("should write back");
        }

        // Mounting the same device afresh reads the inode back off it.
        let reopened =
            fs::open(volume.disk.clone(), FsFlags::empty(), None).expect("should mount again");
        let ext4 = reopened
            .downcast_ref::<Ext4>()
            .expect("should be an ext4 volume");
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        assert_eq!(inode.file_size(), payload.len());
        assert_eq!(read_whole(&inode), payload);
    }

    /// A volume with no room left must say so rather than losing the write.
    #[ktest]
    fn a_write_that_does_not_fit_reports_enospc() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        // The fixture volume is 256 blocks, so this cannot fit.
        let block = vec![0u8; BLOCK_SIZE];
        let mut filled = 0;
        let outcome = loop {
            let mut reader = VmReader::from(block.as_slice()).to_fallible();
            match inode.write_at(filled * BLOCK_SIZE, &mut reader) {
                Ok(_) => filled += 1,
                Err(err) => break err,
            }
            assert!(filled < 512, "the volume should have run out by now");
        };

        assert_eq!(outcome.error(), Errno::ENOSPC);
        assert!(filled > 0, "some of the writes should have fitted");
    }

    /// `i_crtime` lives past the classic 128-byte inode, so a volume with
    /// large inodes reports a creation time where ext2 reports none.
    #[ktest]
    fn a_large_inode_carries_a_creation_time() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        assert!(inode.metadata().birth_at.is_some());
    }
}
