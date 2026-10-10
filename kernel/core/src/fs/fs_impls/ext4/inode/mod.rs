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
//! Nothing here takes a lock, because nothing here mutates: an inode is
//! decoded once when it is read from the device and is immutable afterwards.
//! That is a property of this iteration rather than of the format, and the
//! write path will have to introduce the interior mutability it needs.
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
mod dir;
mod extent;
mod io_range;

use device_id::{DeviceId, encode_device_numbers};
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
pub(crate) struct Inode {
    ino: Ext4Ino,
    desc: InodeDesc,
    payload: InodePayload,
    /// The in-memory endpoint of a named pipe, which the format records as a
    /// type and nothing more.
    pipe: Option<Pipe>,
    fs: Weak<Ext4>,
    extension: Extension,
}

impl Debug for Inode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Inode")
            .field("ino", &self.ino)
            .field("type", &self.desc.type_)
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
        desc: InodeDesc,
        volume_blocks: u64,
        fs: Weak<Ext4>,
    ) -> Result<Arc<Self>> {
        let payload = InodePayload::new(&desc, volume_blocks, fs.clone())?;
        let pipe = (desc.type_ == InodeType::NamedPipe).then(Pipe::new);

        Ok(Arc::new(Self {
            ino,
            desc,
            payload,
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
        self.desc.type_
    }

    pub(super) fn file_size(&self) -> usize {
        self.desc.size as usize
    }

    pub(super) fn mode(&self) -> InodeMode {
        InodeMode::from_bits_truncate(self.desc.perm.bits())
    }

    pub(super) fn uid(&self) -> u32 {
        self.desc.uid
    }

    pub(super) fn gid(&self) -> u32 {
        self.desc.gid
    }

    pub(super) fn atime(&self) -> Duration {
        self.desc.atime
    }

    pub(super) fn mtime(&self) -> Duration {
        self.desc.mtime
    }

    pub(super) fn ctime(&self) -> Duration {
        self.desc.ctime
    }

    pub(super) fn extension(&self) -> &Extension {
        &self.extension
    }

    pub(super) fn fs(&self) -> Result<Arc<Ext4>> {
        self.fs
            .upgrade()
            .ok_or_else(|| Error::with_message(Errno::EIO, "filesystem already dropped"))
    }

    /// Returns the encoded device ID of a character or block special file.
    pub(super) fn device_id(&self) -> u64 {
        match &self.payload {
            InodePayload::Device { device_id } => *device_id,
            _ => 0,
        }
    }

    pub(super) fn page_cache(&self) -> Option<Arc<Vmo>> {
        self.payload.page_cache().map(PageCache::as_vmo).cloned()
    }

    pub(super) fn metadata(&self) -> Metadata {
        let container_dev_id = match self.fs.upgrade() {
            Some(fs) => fs.block_device().id(),
            None => DeviceId::null(),
        };
        let self_dev_id = match self.desc.type_ {
            InodeType::CharDevice | InodeType::BlockDevice => {
                DeviceId::from_encoded_u64(self.device_id())
            }
            _ => None,
        };

        Metadata {
            ino: self.ino as u64,
            size: self.file_size(),
            optimal_block_size: BLOCK_SIZE,
            nr_sectors_allocated: self.desc.sector_count as usize,
            last_access_at: self.desc.atime,
            last_modify_at: self.desc.mtime,
            last_meta_change_at: self.desc.ctime,
            type_: self.desc.type_,
            mode: self.mode(),
            nr_hard_links: self.desc.link_count as usize,
            uid: Uid::new(self.desc.uid),
            gid: Gid::new(self.desc.gid),
            container_dev_id,
            self_dev_id,
            // Unlike ext2, ext4 records a creation time, when the inode is
            // large enough to hold one.
            birth_at: self.desc.crtime,
        }
    }

    /// Reads file data at `offset` through the page cache.
    pub(super) fn read_at(&self, offset: usize, writer: &mut VmWriter) -> Result<usize> {
        if self.desc.type_ == InodeType::Dir {
            return_errno!(Errno::EISDIR);
        }
        if writer.avail() == 0 {
            return Ok(0);
        }

        let file_size = self.file_size();
        if offset >= file_size {
            return Ok(0);
        }

        let read_len = writer.avail().min(file_size - offset);
        writer.limit(read_len);
        self.page_cache_of_data()?.read(offset, writer)?;
        Ok(read_len)
    }

    /// Reads file data at `offset` straight from the device.
    pub(super) fn read_direct_at(&self, offset: usize, writer: &mut VmWriter) -> Result<usize> {
        if self.desc.type_ == InodeType::Dir {
            return_errno!(Errno::EISDIR);
        }
        if !offset.is_multiple_of(BLOCK_SIZE) || !writer.avail().is_multiple_of(BLOCK_SIZE) {
            // TODO: Fall back to a buffered read for an unaligned request, as
            // Linux does.
            return_errno_with_message!(Errno::EINVAL, "not block-aligned");
        }

        let file_size = self.file_size();
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
        let block_manager = self.payload.block_manager()?;
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

    /// Reads the target of a symbolic link.
    pub(super) fn read_link(&self) -> Result<String> {
        if self.desc.type_ != InodeType::SymLink {
            return_errno!(Errno::EINVAL);
        }

        let len = self.file_size();
        let bytes = match &self.payload {
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

    fn page_cache_of_data(&self) -> Result<&PageCache> {
        self.payload
            .page_cache()
            .ok_or_else(|| Error::with_message(Errno::EINVAL, "inode has no data page cache"))
    }
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
    /// Allocated storage in 512-byte sectors, as `stat` reports it.
    sector_count: u64,
    flags: FileFlags,
    file_acl: u64,
    block: [u8; EXTENT_ROOT_LEN],
}

impl InodeDesc {
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

    fn new_data_backed(desc: &InodeDesc, volume_blocks: u64, fs: Weak<Ext4>) -> Result<Self> {
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

/// Decodes the device number a special file stores in its `i_block` area.
fn decode_device_id(block: &[u8; EXTENT_ROOT_LEN]) -> u64 {
    let first = u32::from_le_bytes(block[0..4].try_into().unwrap());
    let second = u32::from_le_bytes(block[4..8].try_into().unwrap());

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

    /// `i_crtime` lives past the classic 128-byte inode, so a volume with
    /// large inodes reports a creation time where ext2 reports none.
    #[ktest]
    fn a_large_inode_carries_a_creation_time() {
        let ext4 = mount();
        let inode = ext4.read_inode(HELLO_INO).expect("should read");

        assert!(inode.metadata().birth_at.is_some());
    }
}
