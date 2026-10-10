// SPDX-License-Identifier: MPL-2.0

//! VFS inode trait implementations for the ext4 [`Inode`](super::super::Inode).
//!
//! The timestamp setters cannot report that a volume is read-only, because the
//! VFS declares them infallible, so they do nothing instead.

use core::time::Duration;

use aster_block::bio::BioStatus;
use device_id::DeviceId;

use crate::{
    device,
    fs::{
        file::{AccessMode, InodeMode, InodeType, PerOpenFileOps, StatusFlags, SyncMode},
        fs_impls::ext4::{FilePerm, Inode as Ext4Inode},
        utils::DirentVisitor,
        vfs::{
            file_system::FileSystem,
            inode::{Extension, FallocMode, FileOps, Inode, Metadata, MknodType, SymbolicLink},
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
        offset: usize,
        reader: &mut VmReader,
        _status_flags: StatusFlags,
    ) -> Result<usize> {
        // TODO: `O_DIRECT` is served through the page cache, which is correct
        // but defeats the point of asking for it.
        self.write_at(offset, reader)
    }

    fn readdir_at(&self, offset: usize, visitor: &mut dyn DirentVisitor) -> Result<usize> {
        self.readdir_at(offset, visitor)
    }
}

impl Inode for Ext4Inode {
    fn size(&self) -> usize {
        self.file_size()
    }

    fn resize(&self, _self_dentry: &Dentry, new_size: usize) -> Result<()> {
        self.resize(new_size)
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

    fn set_mode(&self, _self_dentry: &Dentry, mode: InodeMode) -> Result<()> {
        self.set_mode(mode)
    }

    fn owner(&self) -> Result<Uid> {
        Ok(Uid::new(self.uid()))
    }

    fn set_owner(&self, _self_dentry: &Dentry, uid: Uid) -> Result<()> {
        self.set_uid(uid.into())
    }

    fn group(&self) -> Result<Gid> {
        Ok(Gid::new(self.gid()))
    }

    fn set_group(&self, _self_dentry: &Dentry, gid: Gid) -> Result<()> {
        self.set_gid(gid.into())
    }

    fn atime(&self) -> Duration {
        self.atime()
    }

    fn set_atime(&self, _self_dentry: &Dentry, time: Duration) {
        self.set_atime(time)
    }

    fn mtime(&self) -> Duration {
        self.mtime()
    }

    fn set_mtime(&self, _self_dentry: &Dentry, time: Duration) {
        self.set_mtime(time)
    }

    fn ctime(&self) -> Duration {
        self.ctime()
    }

    fn set_ctime(&self, _self_dentry: &Dentry, time: Duration) {
        self.set_ctime(time)
    }

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

    fn create(
        &self,
        _self_dentry: &Dentry,
        name: &str,
        type_: InodeType,
        mode: InodeMode,
    ) -> Result<Arc<dyn Inode>> {
        Ok(self.create(name, type_, mode)?)
    }

    fn create_symlink(
        &self,
        _self_dentry: &Dentry,
        name: &str,
        target: &str,
        _mode: InodeMode,
    ) -> Result<Arc<dyn Inode>> {
        Ok(self.create_symlink(name, target)?)
    }

    fn mknod(
        &self,
        _self_dentry: &Dentry,
        name: &str,
        mode: InodeMode,
        type_: MknodType,
    ) -> Result<Arc<dyn Inode>> {
        let (inode_type, device_id) = match type_ {
            MknodType::CharDevice(id) => (InodeType::CharDevice, Some(id)),
            MknodType::BlockDevice(id) => (InodeType::BlockDevice, Some(id)),
            MknodType::NamedPipe => (InodeType::NamedPipe, None),
        };

        let new_inode = self.create(name, inode_type, mode)?;
        if let Some(device_id) = device_id {
            new_inode.set_device_id(device_id)?;
        }
        Ok(new_inode)
    }

    fn link(&self, _self_dentry: &Dentry, old_dentry: &Dentry, name: &str) -> Result<()> {
        let old = old_dentry
            .inode()
            .downcast_ref::<Ext4Inode>()
            .ok_or_else(|| Error::with_message(Errno::EXDEV, "the inode is not an ext4 one"))?;
        self.link(old, name)
    }

    fn unlink(&self, child_dentry: &Dentry) -> Result<()> {
        let child = child_dentry
            .inode()
            .downcast_ref::<Ext4Inode>()
            .ok_or_else(|| Error::with_message(Errno::EXDEV, "the inode is not an ext4 one"))?;
        self.unlink(&child_dentry.name(), child)
    }

    fn rmdir(&self, child_dentry: &Dentry) -> Result<()> {
        let child = child_dentry
            .inode()
            .downcast_ref::<Ext4Inode>()
            .ok_or_else(|| Error::with_message(Errno::EXDEV, "the inode is not an ext4 one"))?;
        self.rmdir(&child_dentry.name(), child)
    }

    fn read_link(&self) -> Result<SymbolicLink> {
        self.read_link().map(SymbolicLink::Plain)
    }

    fn sync(&self, mode: SyncMode) -> Result<()> {
        match mode {
            SyncMode::Data | SyncMode::Full => self.sync_all()?,
        }
        if self.fs()?.block_device().sync()? != BioStatus::Complete {
            return_errno_with_message!(Errno::EIO, "failed to flush the block device");
        }
        Ok(())
    }

    fn fallocate(&self, _mode: FallocMode, _offset: usize, _len: usize) -> Result<()> {
        // A read-only mount is the stronger reason to refuse, and the one a
        // caller can do something about.
        self.fs()?.check_writable()?;

        // TODO: Preallocating needs the unwritten extents the reader already
        // understands, which the writer does not produce yet.
        return_errno_with_message!(Errno::EOPNOTSUPP, "ext4 fallocate is not implemented yet")
    }

    fn fs(&self) -> Arc<dyn FileSystem> {
        // The inode must belong to a live filesystem instance.
        self.fs()
            .expect("a mounted filesystem outlives the inodes read from it")
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

#[cfg(ktest)]
mod test {
    use ostd::prelude::ktest;

    use super::*;
    use crate::fs::{
        fs_impls::ext4::{
            fs as ext4_fs,
            test_utils::{
                BLOCK_MAPPED_INO, DELETED_INO, DEVICE_INO, Ext4VolumeBuilder, FIFO_INO,
                HELLO_CONTENT, INLINE_DATA_INO, ROOT_ENTRIES, SLOW_SYMLINK_INO,
                SLOW_SYMLINK_TARGET, SYMLINK_TARGET, WIDE_DIR_FIRST_BLOCK, WIDE_DIR_INO,
                WIDE_DIR_SECOND_BLOCK, assert_errno,
            },
        },
        vfs::file_system::FsFlags,
    };

    /// Mounts a fixture volume and returns it with its root.
    ///
    /// The filesystem must be held for as long as its inodes are used: an
    /// inode refers back to it weakly, as ext2's does, so the mount is what
    /// keeps it alive.
    fn mount() -> (Arc<dyn FileSystem>, Arc<dyn Inode>) {
        mount_with(FsFlags::empty())
    }

    /// Mounts read-only, for the tests about refusing to change the volume.
    fn mount_read_only() -> (Arc<dyn FileSystem>, Arc<dyn Inode>) {
        mount_with(FsFlags::RDONLY)
    }

    fn mount_with(flags: FsFlags) -> (Arc<dyn FileSystem>, Arc<dyn Inode>) {
        let volume = Ext4VolumeBuilder::new().build();
        let fs = ext4_fs::open(volume.disk.clone(), flags, None).expect("should mount");
        let root = fs.root_inode();
        (fs, root)
    }

    fn child(root: &Arc<dyn Inode>, name: &str) -> Arc<dyn Inode> {
        root.lookup(name).expect("the fixture holds this name")
    }

    /// A dentry is what the mutating half of the trait takes, so the tests
    /// that check those need one even though the operations all fail.
    fn root_dentry(root: Arc<dyn Inode>) -> Arc<Dentry> {
        Dentry::new_root(root)
    }

    fn read_all(inode: &Arc<dyn Inode>, status_flags: StatusFlags) -> Vec<u8> {
        let mut buf = vec![0u8; inode.size()];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        let read = inode
            .read_at(0, &mut writer, status_flags)
            .expect("should read");

        assert_eq!(read, buf.len());
        buf
    }

    #[ktest]
    fn reading_through_the_trait_returns_the_file_contents() {
        let (_fs, root) = mount();
        let hello = child(&root, "hello");

        assert_eq!(read_all(&hello, StatusFlags::empty()), HELLO_CONTENT);
    }

    /// `O_DIRECT` takes a separate path that bypasses the page cache, so the
    /// two must agree on what the file holds.
    #[ktest]
    fn a_direct_read_through_the_trait_agrees_with_a_cached_one() {
        let (_fs, root) = mount();
        let two = child(&root, "two");

        assert_eq!(
            read_all(&two, StatusFlags::O_DIRECT),
            read_all(&two, StatusFlags::empty())
        );
    }

    #[ktest]
    fn reading_a_directory_through_the_trait_reports_eisdir() {
        let (_fs, root) = mount();
        let mut buf = [0u8; 16];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();

        assert_errno(
            root.read_at(0, &mut writer, StatusFlags::empty()),
            Errno::EISDIR,
        );
    }

    #[ktest]
    fn writing_through_the_trait_reports_erofs() {
        let (_fs, root) = mount_read_only();
        let hello = child(&root, "hello");
        let buf = [0u8; 4];
        let mut reader = VmReader::from(buf.as_slice()).to_fallible();

        assert_errno(
            hello.write_at(0, &mut reader, StatusFlags::empty()),
            Errno::EROFS,
        );
    }

    /// Every way of changing the volume must refuse, not just the obvious one.
    #[ktest]
    fn changing_a_file_through_the_trait_reports_erofs() {
        let (_fs, root) = mount_read_only();
        let hello = child(&root, "hello");
        let dentry = root_dentry(hello.clone());

        assert_errno(hello.resize(&dentry, 0), Errno::EROFS);
        assert_errno(
            hello.set_mode(&dentry, InodeMode::from_bits_truncate(0)),
            Errno::EROFS,
        );
        assert_errno(hello.set_owner(&dentry, Uid::new(1)), Errno::EROFS);
        assert_errno(hello.set_group(&dentry, Gid::new(1)), Errno::EROFS);
        assert_errno(hello.fallocate(FallocMode::Allocate, 0, 4096), Errno::EROFS);
    }

    /// The VFS declares the timestamp setters infallible, so a read-only mount
    /// can only ignore them. Ignoring them must leave the inode as it was.
    #[ktest]
    fn setting_a_timestamp_on_a_read_only_mount_changes_nothing() {
        let (_fs, root) = mount_read_only();
        let hello = child(&root, "hello");
        let dentry = root_dentry(hello.clone());
        let before = (hello.atime(), hello.mtime(), hello.ctime());

        hello.set_atime(&dentry, Duration::from_secs(1));
        hello.set_mtime(&dentry, Duration::from_secs(2));
        hello.set_ctime(&dentry, Duration::from_secs(3));

        assert_eq!((hello.atime(), hello.mtime(), hello.ctime()), before);
    }

    #[ktest]
    fn metadata_through_the_trait_describes_the_file() {
        let (_fs, root) = mount();
        let hello = child(&root, "hello");

        let metadata = hello.metadata().expect("should read");
        assert_eq!(metadata.ino, hello.ino());
        assert_eq!(metadata.size, HELLO_CONTENT.len());
        assert_eq!(metadata.type_, InodeType::File);
        assert_eq!(metadata.nr_hard_links, 1);
        assert_eq!(hello.mode().expect("should read"), metadata.mode);
        assert_eq!(hello.owner().expect("should read"), metadata.uid);
        assert_eq!(hello.group().expect("should read"), metadata.gid);
    }

    #[ktest]
    fn a_symlink_reads_its_target_through_the_trait() {
        let (_fs, root) = mount();

        let fast = child(&root, "link").read_link().expect("should read");
        let slow = child(&root, "slow-link").read_link().expect("should read");

        assert!(matches!(fast, SymbolicLink::Plain(target) if target == SYMLINK_TARGET));
        assert!(matches!(slow, SymbolicLink::Plain(target) if target == SLOW_SYMLINK_TARGET));
    }

    /// A target too long for the inode lives in a data block, which is a
    /// different payload and a different read path from a short one.
    #[ktest]
    fn a_slow_symlink_is_backed_by_a_data_block() {
        let (_fs, root) = mount();
        let slow = child(&root, "slow-link");

        assert_eq!(slow.ino(), SLOW_SYMLINK_INO as u64);
        assert!(slow.page_cache().is_some());
    }

    #[ktest]
    fn readdir_through_the_trait_lists_every_entry() {
        let (_fs, root) = mount();
        let mut names: Vec<String> = Vec::new();

        root.readdir_at(0, &mut names).expect("should list");

        let expected: Vec<String> = ROOT_ENTRIES
            .iter()
            .map(|&(name, _, _)| name.to_string())
            .collect();
        assert_eq!(names, expected);
    }

    /// A directory of more than one block is where a reader that stops at the
    /// end of the first block silently loses entries.
    #[ktest]
    fn readdir_crosses_from_one_directory_block_to_the_next() {
        let (_fs, root) = mount();
        let wide = child(&root, "wide");
        let mut names: Vec<String> = Vec::new();

        wide.readdir_at(0, &mut names).expect("should list");

        assert_eq!(wide.ino(), WIDE_DIR_INO as u64);
        let expected: Vec<String> = WIDE_DIR_FIRST_BLOCK
            .iter()
            .chain(WIDE_DIR_SECOND_BLOCK.iter())
            .map(|name| name.to_string())
            .collect();
        assert_eq!(names, expected);
    }

    #[ktest]
    fn readdir_on_a_file_reports_enotdir() {
        let (_fs, root) = mount();
        let hello = child(&root, "hello");
        let mut names: Vec<String> = Vec::new();

        assert_errno(hello.readdir_at(0, &mut names), Errno::ENOTDIR);
    }

    #[ktest]
    fn looking_a_name_up_in_a_file_reports_enotdir() {
        let (_fs, root) = mount();
        let hello = child(&root, "hello");

        assert_errno(hello.lookup("anything"), Errno::ENOTDIR);
    }

    #[ktest]
    fn a_name_longer_than_the_limit_reports_enametoolong() {
        let (_fs, root) = mount();
        let name = "x".repeat(crate::fs::utils::NAME_MAX + 1);

        assert_errno(root.lookup(&name), Errno::ENAMETOOLONG);
    }

    /// An inode whose link count has reached zero was deleted, and a directory
    /// entry that still names it must not resurrect it.
    #[ktest]
    fn a_deleted_inode_reports_estale() {
        let (_fs, root) = mount();

        assert_errno(root.lookup("deleted"), Errno::ESTALE);
        assert_eq!(DELETED_INO, 20);
    }

    #[ktest]
    fn a_file_whose_contents_live_in_its_inode_is_refused() {
        let (_fs, root) = mount();

        assert_errno(root.lookup("inline"), Errno::EOPNOTSUPP);
        assert_eq!(INLINE_DATA_INO, 21);
    }

    /// Ext4 still permits the ext2 block-pointer layout, and a driver that
    /// read its `i_block` as an extent root would walk nonsense.
    #[ktest]
    fn a_file_mapped_by_block_pointers_is_refused() {
        let (_fs, root) = mount();

        assert_errno(root.lookup("block-mapped"), Errno::EOPNOTSUPP);
        assert_eq!(BLOCK_MAPPED_INO, 22);
    }

    /// Opening a special file hands off to the device registry rather than
    /// reading the volume, and the dispatch is what this checks.
    #[ktest]
    fn opening_a_character_device_leaves_the_filesystem() {
        let (_fs, root) = mount();
        let null = child(&root, "null");
        let dentry = root_dentry(null.clone());

        assert_eq!(null.ino(), DEVICE_INO as u64);
        assert_eq!(null.type_(), InodeType::CharDevice);
        assert!(
            null.open(&dentry, AccessMode::O_RDONLY, StatusFlags::empty())
                .is_some(),
            "a device file must not be opened as an ordinary file"
        );
    }

    #[ktest]
    fn opening_a_named_pipe_yields_a_pipe() {
        let (_fs, root) = mount();
        let fifo = child(&root, "fifo");
        let dentry = root_dentry(fifo.clone());

        assert_eq!(fifo.ino(), FIFO_INO as u64);
        assert_eq!(fifo.type_(), InodeType::NamedPipe);
        assert!(
            fifo.open(&dentry, AccessMode::O_RDONLY, StatusFlags::O_NONBLOCK)
                .is_some(),
            "a named pipe must not be opened as an ordinary file"
        );
    }

    /// An ordinary file has no per-open state, so the VFS keeps using the
    /// inode itself.
    #[ktest]
    fn opening_an_ordinary_file_needs_no_per_open_state() {
        let (_fs, root) = mount();
        let hello = child(&root, "hello");
        let dentry = root_dentry(hello.clone());

        assert!(
            hello
                .open(&dentry, AccessMode::O_RDONLY, StatusFlags::empty())
                .is_none()
        );
    }

    #[ktest]
    fn an_inode_points_back_at_the_filesystem_that_holds_it() {
        let (_fs, root) = mount();
        let hello = child(&root, "hello");

        assert_eq!(hello.fs().name(), "ext4");
    }

    #[ktest]
    fn syncing_an_inode_on_a_read_only_mount_succeeds() {
        let (_fs, root) = mount();
        let hello = child(&root, "hello");

        hello.sync(SyncMode::Data).expect("nothing to write back");
        hello.sync(SyncMode::Full).expect("nothing to write back");
    }
}
