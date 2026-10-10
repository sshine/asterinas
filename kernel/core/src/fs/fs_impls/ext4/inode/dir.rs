// SPDX-License-Identifier: MPL-2.0

//! Ext4 directory entry layout, lookup, and readdir.
//!
//! A directory's contents are a flat list of variable-length records packed
//! into ordinary data blocks. Each record begins with an 8-byte header and is
//! followed by the entry's name. A record never crosses a block boundary.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │ inode (4 B) │ rec_len (2 B) │ name_len (1 B) │ type (1 B) │ name … │ pad …
//! └───────────────────────────────────────────────────────────┘
//! ```
//!
//! An `ino` of zero marks a record that holds no entry. Three different things
//! use that encoding, and reading treats all of them alike:
//!
//! - a deleted entry, whose space has been merged into its predecessor;
//! - the 12-byte checksum record that ends a block on a volume with
//!   `metadata_csum`, which `file_type` 0xDE distinguishes;
//! - the whole of an interior block of a hash-indexed directory, which
//!   presents itself as one record spanning the block.
//!
//! The last of these is why a linear walk reads a hash-indexed directory
//! correctly: the index hides inside records that carry no entry, so a reader
//! that skips them sees exactly the real entries. Only a writer has to
//! understand the index, which is why this iteration reads such a directory
//! and will refuse to modify one.
//!
//! TODO: The layout above is byte-identical to ext2's, and the two decoders
//! have been written twice. Extracting one is worth doing once a third
//! filesystem needs it, or once the ext4 write path has added the checksum
//! record handling that would make the two diverge.

#![short_vis_path::add(ext4)]

use ostd::const_assert;

use super::{FileFlags, Inode, InodeInner, now};
use crate::fs::{fs_impls::ext4::prelude::*, utils::NAME_MAX};

impl Inode {
    /// Resolves one name in this directory.
    pub(in ext4) fn lookup(&self, name: &str) -> Result<Arc<Inode>> {
        let ino = self.find_entry(name)?;
        self.fs()?.read_inode(ino)
    }

    /// Feeds the entries from `offset` onwards to `visitor`.
    ///
    /// Returns how far the offset advanced, which the caller uses as the
    /// starting point of the next call.
    pub(in ext4) fn readdir_at(
        &self,
        offset: usize,
        visitor: &mut dyn DirentVisitor,
    ) -> Result<usize> {
        if self.inode_type() != InodeType::Dir {
            return_errno!(Errno::ENOTDIR);
        }

        let inner = self.inner.read();
        let size = inner.desc.size as usize;
        let min_rec_len = DirEntryHeader::min_rec_len(1) as usize;
        if size < min_rec_len || offset > size - min_rec_len {
            return Ok(0);
        }

        let page_cache = inner.page_cache_of_data()?;
        let mut current_offset = offset;
        let mut advanced = 0;

        for block_idx in offset / BLOCK_SIZE..size.div_ceil(BLOCK_SIZE) {
            let block_offset = block_idx * BLOCK_SIZE;
            let block = DirBlockView::new(page_cache, block_idx, size);
            let mut entries = block.iter_entries();

            while let Some((entry_offset_in_block, entry)) = entries.next_entry()? {
                let entry_offset = block_offset + entry_offset_in_block;
                let next_offset = entry_offset + entry.header.rec_len as usize;
                if next_offset <= current_offset {
                    continue;
                }
                if entry_offset < current_offset {
                    current_offset = next_offset;
                    continue;
                }

                if entry.header.ino != 0 {
                    let name = core::str::from_utf8(entry.name).map_err(|_| {
                        Error::with_message(Errno::EUCLEAN, "dir entry name is not UTF-8")
                    })?;
                    let type_ = InodeType::from(
                        DirEntryFileType::try_from(entry.header.file_type)
                            .unwrap_or(DirEntryFileType::Unknown),
                    );
                    if visitor
                        .visit(name, entry.header.ino as u64, type_, next_offset)
                        .is_err()
                    {
                        return Ok(current_offset - offset);
                    }
                }

                current_offset = next_offset;
            }

            advanced = current_offset - offset;
        }

        Ok(advanced)
    }

    /// Adds an entry naming `ino`, growing the directory if nothing in it has
    /// room.
    pub(in ext4) fn add_entry(&self, name: &str, ino: Ext4Ino, type_: InodeType) -> Result<()> {
        if name.is_empty() || name.len() > NAME_MAX {
            return_errno!(Errno::ENAMETOOLONG);
        }
        if name.contains('/') {
            return_errno_with_message!(Errno::EINVAL, "a name may not contain a separator");
        }

        let fs = self.fs()?;
        self.check_directory_is_modifiable()?;
        let needed = DirEntryHeader::min_rec_len(name.len()) as usize;
        let mut inner = self.inner.write();

        if let Some(slot) = inner.find_slot(needed)? {
            inner.write_entry(&slot, name, ino, type_)?;
            inner.set_modified(now());
            return Ok(());
        }

        // Nothing had room, so the directory gains a block whose one record
        // spans it.
        let size = inner.desc.size as usize;
        inner.resize(&fs, size + BLOCK_SIZE)?;
        let slot = DirSlot {
            offset: size,
            rec_len: BLOCK_SIZE,
            shrink_previous_to: None,
        };
        inner.write_entry(&slot, name, ino, type_)?;
        inner.set_modified(now());
        Ok(())
    }

    /// Removes the entry naming `name`.
    pub(in ext4) fn remove_entry(&self, name: &str) -> Result<Ext4Ino> {
        self.check_directory_is_modifiable()?;
        let mut inner = self.inner.write();
        let found = inner.find_entry(name)?;
        inner.delete_entry(&found)?;
        inner.set_modified(now());
        Ok(found.ino)
    }

    /// Marks this directory as hash-indexed, for a test that needs one.
    #[cfg(ktest)]
    pub(in ext4) fn mark_hash_indexed(&self) {
        self.inner.write().desc.flags |= FileFlags::INDEX_DIR;
    }

    /// Fails if this directory is one whose entries cannot safely be changed.
    ///
    /// A hash-indexed directory keeps an index beside its records, and adding
    /// or removing a record without maintaining that index makes the names it
    /// no longer matches unreachable. Reading one is safe, because the index
    /// hides in records that carry no entry.
    fn check_directory_is_modifiable(&self) -> Result<()> {
        if self.inner.read().desc.flags.contains(FileFlags::INDEX_DIR) {
            return_errno_with_message!(
                Errno::EOPNOTSUPP,
                "changing a hash-indexed directory is not implemented"
            );
        }

        Ok(())
    }

    /// Returns whether a directory holds nothing but `.` and `..`.
    pub(in ext4) fn is_empty_dir(&self) -> Result<bool> {
        let inner = self.inner.read();
        let size = inner.desc.size as usize;
        let page_cache = inner.page_cache_of_data()?;

        for block_idx in 0..size.div_ceil(BLOCK_SIZE) {
            let block = DirBlockView::new(page_cache, block_idx, size);
            let mut entries = block.iter_entries();
            while let Some((_, entry)) = entries.next_entry()? {
                if entry.header.ino != 0 && entry.name != b"." && entry.name != b".." {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Writes the `.` and `..` of a directory that has just been created.
    pub(in ext4) fn fill_new_dir(&self, parent_ino: Ext4Ino) -> Result<()> {
        let fs = self.fs()?;
        let mut inner = self.inner.write();
        inner.resize(&fs, BLOCK_SIZE)?;

        let page_cache = inner.page_cache_of_data()?;
        let dot = DirEntryHeader::min_rec_len(1) as usize;
        write_record(page_cache, 0, dot, ".", self.ino, InodeType::Dir)?;
        write_record(
            page_cache,
            dot,
            BLOCK_SIZE - dot,
            "..",
            parent_ino,
            InodeType::Dir,
        )?;
        Ok(())
    }
}

/// Where a new entry will go.
struct DirSlot {
    /// Byte offset in the directory at which the new record starts.
    offset: usize,
    /// Bytes the new record may occupy.
    rec_len: usize,
    /// The record before it, which has to give up the space first.
    shrink_previous_to: Option<(usize, u16)>,
}

/// One located entry.
struct FoundEntry {
    ino: Ext4Ino,
    offset: usize,
    rec_len: usize,
}

impl InodeInner {
    /// Finds somewhere in the directory for a record of `needed` bytes.
    ///
    /// A record that holds no entry can be taken whole; a live one can give up
    /// whatever it has beyond what its own name needs.
    fn find_slot(&self, needed: usize) -> Result<Option<DirSlot>> {
        let size = self.desc.size as usize;
        let page_cache = self.page_cache_of_data()?;

        for block_idx in 0..size.div_ceil(BLOCK_SIZE) {
            let block_offset = block_idx * BLOCK_SIZE;
            let block = DirBlockView::new(page_cache, block_idx, size);
            let mut entries = block.iter_entries();

            while let Some((entry_offset, entry)) = entries.next_entry()? {
                let offset = block_offset + entry_offset;
                let rec_len = entry.header.rec_len as usize;

                if entry.header.ino == 0 && rec_len >= needed {
                    return Ok(Some(DirSlot {
                        offset,
                        rec_len,
                        shrink_previous_to: None,
                    }));
                }

                let in_use = DirEntryHeader::min_rec_len(entry.header.name_len as usize) as usize;
                if entry.header.ino != 0 && rec_len >= in_use + needed {
                    return Ok(Some(DirSlot {
                        offset: offset + in_use,
                        rec_len: rec_len - in_use,
                        shrink_previous_to: Some((offset, in_use as u16)),
                    }));
                }
            }
        }

        Ok(None)
    }

    fn write_entry(
        &self,
        slot: &DirSlot,
        name: &str,
        ino: Ext4Ino,
        type_: InodeType,
    ) -> Result<()> {
        let page_cache = self.page_cache_of_data()?;
        if let Some((offset, rec_len)) = slot.shrink_previous_to {
            set_rec_len(page_cache, offset, rec_len)?;
        }
        write_record(page_cache, slot.offset, slot.rec_len, name, ino, type_)
    }

    fn find_entry(&self, name: &str) -> Result<FoundEntry> {
        let size = self.desc.size as usize;
        let page_cache = self.page_cache_of_data()?;
        let wanted = name.as_bytes();

        for block_idx in 0..size.div_ceil(BLOCK_SIZE) {
            let block_offset = block_idx * BLOCK_SIZE;
            let block = DirBlockView::new(page_cache, block_idx, size);
            let mut entries = block.iter_entries();
            while let Some((entry_offset, entry)) = entries.next_entry()? {
                if entry.header.ino != 0 && entry.name == wanted {
                    return Ok(FoundEntry {
                        ino: entry.header.ino,
                        offset: block_offset + entry_offset,
                        rec_len: entry.header.rec_len as usize,
                    });
                }
            }
        }

        return_errno!(Errno::ENOENT)
    }

    /// Clears an entry, giving its space to the record before it.
    fn delete_entry(&self, found: &FoundEntry) -> Result<()> {
        let page_cache = self.page_cache_of_data()?;
        let size = self.desc.size as usize;
        let block_start = (found.offset / BLOCK_SIZE) * BLOCK_SIZE;

        // Walking from the start of the block is the only way to find the
        // record before this one, because records are not back-linked.
        let block = DirBlockView::new(page_cache, found.offset / BLOCK_SIZE, size);
        let mut entries = block.iter_entries();
        let mut previous = None;
        while let Some((entry_offset, entry)) = entries.next_entry()? {
            let offset = block_start + entry_offset;
            if offset == found.offset {
                break;
            }
            previous = Some((offset, entry.header.rec_len as usize));
        }

        match previous {
            Some((offset, rec_len)) => {
                set_rec_len(page_cache, offset, (rec_len + found.rec_len) as u16)?
            }
            // The first record of a block cannot be merged away, so it stays
            // as a record that holds no entry.
            None => page_cache.write_val(found.offset, &0u32)?,
        }
        if previous.is_some() {
            page_cache.write_val(found.offset, &0u32)?;
        }
        Ok(())
    }
}

/// Writes one complete record.
fn write_record(
    page_cache: &PageCache,
    offset: usize,
    rec_len: usize,
    name: &str,
    ino: Ext4Ino,
    type_: InodeType,
) -> Result<()> {
    let header = DirEntryHeader {
        ino,
        rec_len: rec_len as u16,
        name_len: name.len() as u8,
        file_type: DirEntryFileType::from(type_) as u8,
    };

    page_cache.write_val(offset, &header)?;
    page_cache.write_bytes(offset + size_of::<DirEntryHeader>(), name.as_bytes())?;
    Ok(())
}

fn set_rec_len(page_cache: &PageCache, offset: usize, rec_len: u16) -> Result<()> {
    page_cache.write_val(
        offset + core::mem::offset_of!(DirEntryHeader, rec_len),
        &rec_len,
    )?;
    Ok(())
}

impl From<InodeType> for DirEntryFileType {
    fn from(type_: InodeType) -> Self {
        match type_ {
            InodeType::File => Self::File,
            InodeType::Dir => Self::Dir,
            InodeType::CharDevice => Self::CharDevice,
            InodeType::BlockDevice => Self::BlockDevice,
            InodeType::NamedPipe => Self::NamedPipe,
            InodeType::Socket => Self::Socket,
            InodeType::SymLink => Self::SymLink,
            _ => Self::Unknown,
        }
    }
}

impl Inode {
    /// Returns the inode number the given name resolves to.
    fn find_entry(&self, name: &str) -> Result<Ext4Ino> {
        if self.inode_type() != InodeType::Dir {
            return_errno!(Errno::ENOTDIR);
        }
        if name.len() > NAME_MAX {
            return_errno!(Errno::ENAMETOOLONG);
        }

        let inner = self.inner.read();
        let size = inner.desc.size as usize;
        let page_cache = inner.page_cache_of_data()?;
        let wanted = name.as_bytes();

        for block_idx in 0..size.div_ceil(BLOCK_SIZE) {
            let block = DirBlockView::new(page_cache, block_idx, size);
            let mut entries = block.iter_entries();
            while let Some((_, entry)) = entries.next_entry()? {
                if entry.header.ino != 0 && entry.name == wanted {
                    return Ok(entry.header.ino);
                }
            }
        }

        return_errno!(Errno::ENOENT)
    }
}

/// A bounded view of one directory block in the page cache.
struct DirBlockView<'a> {
    page_cache: &'a PageCache,
    /// Absolute byte offset of this block in the directory.
    offset: usize,
    /// Valid bytes within this block.
    limit: usize,
}

impl<'a> DirBlockView<'a> {
    fn new(page_cache: &'a PageCache, block_idx: usize, file_size: usize) -> Self {
        let offset = block_idx * BLOCK_SIZE;
        Self {
            page_cache,
            offset,
            limit: file_size.saturating_sub(offset).min(BLOCK_SIZE),
        }
    }

    fn iter_entries(&self) -> DirBlockViewIter<'_> {
        DirBlockViewIter {
            block: self,
            cursor: self.offset,
            name_buf: [0u8; NAME_MAX],
        }
    }

    /// Reads and validates the record header at an absolute offset.
    fn read_header(&self, offset: usize) -> Result<DirEntryHeader> {
        let header: DirEntryHeader = self
            .page_cache
            .read_val(offset)
            .map_err(|_| Error::with_message(Errno::EIO, "failed to read dir entry header"))?;

        let rec_len = header.rec_len as usize;
        let name_len = header.name_len as usize;
        if !rec_len.is_multiple_of(4) {
            return_errno_with_message!(Errno::EUCLEAN, "dir entry rec_len is not 4-byte aligned");
        }
        if header.ino != 0 && name_len == 0 {
            return_errno_with_message!(Errno::EUCLEAN, "dir entry has an empty name");
        }
        if rec_len < DirEntryHeader::min_rec_len(name_len) as usize {
            return_errno_with_message!(
                Errno::EUCLEAN,
                "dir entry rec_len is smaller than its name needs"
            );
        }
        if offset + rec_len > self.offset + self.limit {
            return_errno_with_message!(Errno::EUCLEAN, "dir entry extends beyond its block");
        }

        Ok(header)
    }
}

/// Iterator over the records of one directory block.
struct DirBlockViewIter<'a> {
    block: &'a DirBlockView<'a>,
    /// Absolute offset of the next record.
    cursor: usize,
    /// Reusable buffer for the current record's name.
    name_buf: [u8; NAME_MAX],
}

impl DirBlockViewIter<'_> {
    /// Returns the next record and its offset within the block.
    ///
    /// A record that holds no entry is returned with an empty name.
    fn next_entry(&mut self) -> Result<Option<(usize, DirEntry<'_>)>> {
        if self.cursor >= self.block.offset + self.block.limit {
            return Ok(None);
        }

        let header = self.block.read_header(self.cursor)?;
        let entry_offset = self.cursor - self.block.offset;
        self.cursor += header.rec_len as usize;

        let name = if header.ino != 0 {
            let name_len = header.name_len as usize;
            self.block.page_cache.read_bytes(
                self.block.offset + entry_offset + size_of::<DirEntryHeader>(),
                &mut self.name_buf[..name_len],
            )?;
            &self.name_buf[..name_len]
        } else {
            &[]
        };

        Ok(Some((entry_offset, DirEntry { header, name })))
    }
}

/// One decoded directory record. `name` borrows the iterator's buffer.
#[derive(Debug)]
struct DirEntry<'a> {
    header: DirEntryHeader,
    name: &'a [u8],
}

/// The on-disk directory record header.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
struct DirEntryHeader {
    ino: u32,
    rec_len: u16,
    name_len: u8,
    file_type: u8,
}

const_assert!(size_of::<DirEntryHeader>() == 8);

impl DirEntryHeader {
    /// Returns the shortest record that can hold a name of `name_len` bytes.
    const fn min_rec_len(name_len: usize) -> u16 {
        ((name_len + size_of::<Self>()).next_multiple_of(4)) as u16
    }
}

/// The `file_type` byte, which records an entry's type so that readdir needs
/// no inode lookup.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, TryFromInt)]
enum DirEntryFileType {
    Unknown = 0,
    File = 1,
    Dir = 2,
    CharDevice = 3,
    BlockDevice = 4,
    NamedPipe = 5,
    Socket = 6,
    SymLink = 7,
}

impl From<DirEntryFileType> for InodeType {
    fn from(file_type: DirEntryFileType) -> Self {
        match file_type {
            DirEntryFileType::Unknown => Self::Unknown,
            DirEntryFileType::File => Self::File,
            DirEntryFileType::Dir => Self::Dir,
            DirEntryFileType::CharDevice => Self::CharDevice,
            DirEntryFileType::BlockDevice => Self::BlockDevice,
            DirEntryFileType::NamedPipe => Self::NamedPipe,
            DirEntryFileType::Socket => Self::Socket,
            DirEntryFileType::SymLink => Self::SymLink,
        }
    }
}

#[cfg(ktest)]
mod test {
    use ostd::prelude::*;

    use super::*;
    use crate::fs::{file::InodeMode, fs_impls::ext4::test_utils::assert_errno};

    #[ktest]
    fn min_rec_len_rounds_up_to_four() {
        assert_eq!(DirEntryHeader::min_rec_len(0), 8);
        assert_eq!(DirEntryHeader::min_rec_len(1), 12);
        assert_eq!(DirEntryHeader::min_rec_len(4), 12);
        assert_eq!(DirEntryHeader::min_rec_len(5), 16);
        assert_eq!(DirEntryHeader::min_rec_len(NAME_MAX), 264);
    }

    /// A zeroed block has `rec_len == 0`, which would otherwise leave the
    /// iterator in place forever.
    #[ktest]
    fn a_zeroed_block_is_rejected() {
        let page_cache = PageCache::new_anon(BLOCK_SIZE).expect("should allocate");
        let view = DirBlockView::new(&page_cache, 0, BLOCK_SIZE);

        assert_errno(view.iter_entries().next_entry(), Errno::EUCLEAN);
    }

    /// The checksum record that ends a block on a volume with `metadata_csum`
    /// has no name and no inode, and must read as a record that holds no entry
    /// rather than as corruption.
    #[ktest]
    fn a_checksum_record_is_skipped() {
        let page_cache = PageCache::new_anon(BLOCK_SIZE).expect("should allocate");
        let mut block = vec![0u8; BLOCK_SIZE];
        // One entry named "a", then the 12-byte checksum record.
        block[0..4].copy_from_slice(&12u32.to_le_bytes());
        block[4..6].copy_from_slice(&((BLOCK_SIZE - 12) as u16).to_le_bytes());
        block[6] = 1;
        block[7] = DirEntryFileType::File as u8;
        block[8] = b'a';
        let tail = BLOCK_SIZE - 12;
        block[tail + 4..tail + 6].copy_from_slice(&12u16.to_le_bytes());
        block[tail + 7] = 0xde;
        page_cache.write_bytes(0, &block).expect("should write");

        let view = DirBlockView::new(&page_cache, 0, BLOCK_SIZE);
        let mut entries = view.iter_entries();

        let (_, first) = entries
            .next_entry()
            .expect("should read")
            .expect("one entry");
        assert_eq!(first.name, b"a");
        let (_, tail_entry) = entries
            .next_entry()
            .expect("should read")
            .expect("the tail");
        assert_eq!(tail_entry.header.ino, 0);
        assert!(tail_entry.name.is_empty());
        assert!(entries.next_entry().expect("should read").is_none());
    }

    /// A hash-indexed directory reads correctly but must not be changed, since
    /// nothing here maintains the index beside its records.
    #[ktest]
    fn a_hash_indexed_directory_refuses_to_change() {
        crate::time::clocks::init_for_ktest();
        let volume = crate::fs::fs_impls::ext4::test_utils::Ext4VolumeBuilder::new().build();
        let fs = crate::fs::fs_impls::ext4::fs::open(
            volume.disk.clone(),
            crate::fs::vfs::file_system::FsFlags::empty(),
            None,
        )
        .expect("should mount");
        let root = fs
            .downcast_ref::<crate::fs::fs_impls::ext4::Ext4>()
            .expect("should be an ext4 volume")
            .root_inode()
            .expect("root should be readable");

        let dir = root
            .create(
                "hashed",
                InodeType::Dir,
                InodeMode::from_bits_truncate(0o755),
            )
            .expect("should create");
        dir.mark_hash_indexed();

        assert_errno(dir.add_entry("name", 2, InodeType::File), Errno::EOPNOTSUPP);
        assert_errno(dir.remove_entry("."), Errno::EOPNOTSUPP);
    }

    #[ktest]
    fn a_record_running_past_its_block_is_rejected() {
        let page_cache = PageCache::new_anon(BLOCK_SIZE).expect("should allocate");
        let mut block = vec![0u8; BLOCK_SIZE];
        block[0..4].copy_from_slice(&12u32.to_le_bytes());
        block[4..6].copy_from_slice(&((BLOCK_SIZE + 4) as u16).to_le_bytes());
        block[6] = 1;
        block[7] = DirEntryFileType::File as u8;
        block[8] = b'a';
        page_cache.write_bytes(0, &block).expect("should write");

        let view = DirBlockView::new(&page_cache, 0, BLOCK_SIZE);

        assert_errno(view.iter_entries().next_entry(), Errno::EUCLEAN);
    }
}
