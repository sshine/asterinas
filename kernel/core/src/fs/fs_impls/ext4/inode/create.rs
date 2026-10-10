// SPDX-License-Identifier: MPL-2.0

//! Creating and removing the files a directory names.
//!
//! An inode and the directory entry naming it are two separate pieces of the
//! volume, and the order they are written in decides what a crash leaves
//! behind. The inode is written first here, so an interrupted creation leaves
//! an inode that nothing names, which `e2fsck` moves to `lost+found`. The
//! other order would leave a name pointing at an inode that was never written,
//! which is a corrupt directory rather than a recoverable one.
//!
//! Removal is the mirror of that: the entry goes first, then the link count.
//! An inode whose last link is gone is not freed until nothing holds it open,
//! which is what lets a program keep reading a file it has already deleted.

#![short_vis_path::add(ext4)]

use super::{Inode, InodeDesc, now};
use crate::{
    fs::{
        file::InodeMode,
        fs_impls::ext4::{fs::Ext4, prelude::*},
    },
    process::posix_thread::AsPosixThread,
    thread::Thread,
};

impl Inode {
    /// Creates a file of `type_` named `name` in this directory.
    pub(in ext4) fn create(
        &self,
        name: &str,
        type_: InodeType,
        mode: InodeMode,
    ) -> Result<Arc<Inode>> {
        if self.inode_type() != InodeType::Dir {
            return_errno!(Errno::ENOTDIR);
        }
        if matches!(type_, InodeType::Unknown | InodeType::SymLink) {
            return_errno_with_message!(Errno::EINVAL, "this type needs its own creation path");
        }

        let fs = self.fs()?;
        fs.check_writable()?;
        if self.lookup(name).is_ok() {
            return_errno!(Errno::EEXIST);
        }

        let child = self.new_child(&fs, type_, mode)?;
        if type_ == InodeType::Dir {
            child.fill_new_dir(self.ino())?;
        }

        // The entry is added last, so that nothing can name an inode that was
        // not written.
        child.sync_metadata()?;
        self.add_entry(name, child.ino(), type_)?;
        if type_ == InodeType::Dir {
            // The child's `..` is a second link to this directory.
            self.adjust_link_count(1);
        }

        fs.insert_inode(child.clone());
        Ok(child)
    }

    /// Creates a symbolic link named `name` pointing at `target`.
    pub(in ext4) fn create_symlink(&self, name: &str, target: &str) -> Result<Arc<Inode>> {
        if self.inode_type() != InodeType::Dir {
            return_errno!(Errno::ENOTDIR);
        }
        if target.is_empty() || target.len() >= BLOCK_SIZE {
            return_errno!(Errno::ENAMETOOLONG);
        }

        let fs = self.fs()?;
        fs.check_writable()?;
        if self.lookup(name).is_ok() {
            return_errno!(Errno::EEXIST);
        }

        let child = self.new_child(
            &fs,
            InodeType::SymLink,
            InodeMode::from_bits_truncate(0o777),
        )?;
        child.write_symlink_target(target)?;
        child.sync_metadata()?;
        self.add_entry(name, child.ino(), InodeType::SymLink)?;

        fs.insert_inode(child.clone());
        Ok(child)
    }

    /// Adds a second name for an existing file.
    pub(in ext4) fn link(&self, old: &Inode, name: &str) -> Result<()> {
        if self.inode_type() != InodeType::Dir {
            return_errno!(Errno::ENOTDIR);
        }
        if old.inode_type() == InodeType::Dir {
            return_errno_with_message!(Errno::EPERM, "a directory may not be hard-linked");
        }

        let fs = self.fs()?;
        fs.check_writable()?;
        if self.lookup(name).is_ok() {
            return_errno!(Errno::EEXIST);
        }
        if old.link_count() >= MAX_LINK_COUNT {
            return_errno!(Errno::EMLINK);
        }

        old.adjust_link_count(1);
        old.sync_metadata()?;
        self.add_entry(name, old.ino(), old.inode_type())
    }

    /// Removes a name for a file that is not a directory.
    pub(in ext4) fn unlink(&self, name: &str, child: &Inode) -> Result<()> {
        if child.inode_type() == InodeType::Dir {
            return_errno!(Errno::EISDIR);
        }

        let fs = self.fs()?;
        fs.check_writable()?;

        // The entry goes first: an inode that nothing names is recoverable,
        // while a name pointing at a freed inode is not.
        self.remove_entry(name)?;
        child.adjust_link_count(-1);
        child.sync_metadata()
    }

    /// Removes an empty subdirectory.
    pub(in ext4) fn rmdir(&self, name: &str, child: &Inode) -> Result<()> {
        if child.inode_type() != InodeType::Dir {
            return_errno!(Errno::ENOTDIR);
        }
        if !child.is_empty_dir()? {
            return_errno!(Errno::ENOTEMPTY);
        }

        let fs = self.fs()?;
        fs.check_writable()?;

        self.remove_entry(name)?;
        // A directory holds a link to itself through `.`, which goes with it.
        child.adjust_link_count(-2);
        child.sync_metadata()?;
        self.adjust_link_count(-1);
        self.sync_metadata()
    }

    /// Builds and registers a new inode, without naming it.
    fn new_child(&self, fs: &Ext4, type_: InodeType, mode: InodeMode) -> Result<Arc<Inode>> {
        let ino = fs.alloc_ino(self.ino(), type_ == InodeType::Dir)?;
        let (uid, gid) = current_owner();
        // A directory starts with two links: the name in its parent, and its
        // own `.`.
        let link_count = if type_ == InodeType::Dir { 2 } else { 1 };
        let desc = InodeDesc::new(type_, mode, uid, gid, link_count, now());

        Inode::new(
            ino,
            Dirty::new_dirty(desc),
            fs.super_block().blocks_count,
            fs.weak_self(),
        )
    }
}

/// The largest number of names one file may have, as the format allows.
const MAX_LINK_COUNT: u16 = 65000;

/// Returns the user and group a newly created file belongs to.
fn current_owner() -> (u32, u32) {
    Thread::current()
        .and_then(|thread| {
            thread
                .as_posix_thread()
                .map(|posix_thread| posix_thread.credentials())
        })
        .map(|credentials| {
            (
                u32::from(credentials.fsuid()),
                u32::from(credentials.fsgid()),
            )
        })
        .unwrap_or((0, 0))
}

#[cfg(ktest)]
mod test {
    use ostd::prelude::ktest;

    use super::*;
    use crate::fs::{
        fs_impls::ext4::{
            fs,
            test_utils::{Ext4VolumeBuilder, assert_errno},
        },
        vfs::file_system::{FileSystem, FsFlags},
    };

    fn mount_with(flags: FsFlags) -> (Arc<dyn FileSystem>, Arc<Inode>) {
        // Creating a file stamps it with the current time, which a kernel test
        // only has once the clocks are up.
        crate::time::clocks::init_for_ktest();

        let volume = Ext4VolumeBuilder::new().build();
        let handle = fs::open(volume.disk.clone(), flags, None).expect("should mount");
        let root = handle
            .downcast_ref::<Ext4>()
            .expect("should be an ext4 volume")
            .root_inode()
            .expect("root should be readable");
        (handle, root)
    }

    fn mount() -> (Arc<dyn FileSystem>, Arc<Inode>) {
        mount_with(FsFlags::empty())
    }

    fn mode(bits: u16) -> InodeMode {
        InodeMode::from_bits_truncate(bits)
    }

    #[ktest]
    fn a_created_file_can_be_found_again() {
        let (_fs, root) = mount();

        let created = root
            .create("new-file", InodeType::File, mode(0o644))
            .expect("should create");
        let found = root.lookup("new-file").expect("should resolve");

        assert_eq!(found.ino(), created.ino());
        assert_eq!(found.inode_type(), InodeType::File);
        assert_eq!(found.file_size(), 0);
    }

    /// A new file has to be usable, not merely present.
    #[ktest]
    fn a_created_file_holds_what_is_written_to_it() {
        let (_fs, root) = mount();
        let file = root
            .create("written", InodeType::File, mode(0o644))
            .expect("should create");

        let payload = [0x9eu8; 300];
        let mut reader = VmReader::from(payload.as_slice()).to_fallible();
        file.write_at(0, &mut reader).expect("should write");

        let mut buf = [0u8; 300];
        let mut writer = VmWriter::from(buf.as_mut_slice()).to_fallible();
        file.read_at(0, &mut writer).expect("should read");
        assert_eq!(buf, payload);
    }

    #[ktest]
    fn creating_a_name_that_exists_reports_eexist() {
        let (_fs, root) = mount();
        root.create("twice", InodeType::File, mode(0o644))
            .expect("should create");

        assert_errno(
            root.create("twice", InodeType::File, mode(0o644)),
            Errno::EEXIST,
        );
    }

    #[ktest]
    fn a_name_holding_a_separator_is_refused() {
        let (_fs, root) = mount();

        assert_errno(
            root.create("a/b", InodeType::File, mode(0o644)),
            Errno::EINVAL,
        );
    }

    #[ktest]
    fn creating_on_a_read_only_mount_reports_erofs() {
        let (_fs, root) = mount_with(FsFlags::RDONLY);

        assert_errno(
            root.create("nope", InodeType::File, mode(0o644)),
            Errno::EROFS,
        );
    }

    /// A new directory has to name itself and its parent, or nothing can walk
    /// out of it again.
    #[ktest]
    fn a_created_directory_names_itself_and_its_parent() {
        let (_fs, root) = mount();

        let dir = root
            .create("subdir", InodeType::Dir, mode(0o755))
            .expect("should create");

        assert_eq!(dir.inode_type(), InodeType::Dir);
        assert_eq!(dir.lookup(".").expect("should resolve").ino(), dir.ino());
        assert_eq!(dir.lookup("..").expect("should resolve").ino(), root.ino());
        assert!(dir.is_empty_dir().expect("should walk"));
    }

    /// The `..` of a subdirectory is a second name for its parent.
    #[ktest]
    fn a_created_directory_adds_a_link_to_its_parent() {
        let (_fs, root) = mount();
        let before = root.link_count();

        root.create("counted", InodeType::Dir, mode(0o755))
            .expect("should create");

        assert_eq!(root.link_count(), before + 1);
    }

    #[ktest]
    fn an_unlinked_name_stops_resolving() {
        let (_fs, root) = mount();
        let file = root
            .create("doomed", InodeType::File, mode(0o644))
            .expect("should create");

        root.unlink("doomed", &file).expect("should unlink");

        assert_errno(root.lookup("doomed"), Errno::ENOENT);
    }

    /// The inode number has to come back, or a volume runs out of inodes long
    /// before it runs out of blocks.
    #[ktest]
    fn unlinking_the_last_name_frees_the_inode() {
        let (fs, root) = mount();
        let before = fs.sb().ffree;

        let file = root
            .create("temporary", InodeType::File, mode(0o644))
            .expect("should create");
        root.unlink("temporary", &file).expect("should unlink");
        // The inode is freed once nothing holds it, not when it is unlinked.
        drop(file);

        assert_eq!(fs.sb().ffree, before);
    }

    #[ktest]
    fn unlinking_a_directory_reports_eisdir() {
        let (_fs, root) = mount();
        let dir = root
            .create("a-dir", InodeType::Dir, mode(0o755))
            .expect("should create");

        assert_errno(root.unlink("a-dir", &dir), Errno::EISDIR);
    }

    #[ktest]
    fn removing_a_directory_that_holds_something_reports_enotempty() {
        let (_fs, root) = mount();
        let dir = root
            .create("full", InodeType::Dir, mode(0o755))
            .expect("should create");
        dir.create("inside", InodeType::File, mode(0o644))
            .expect("should create");

        assert_errno(root.rmdir("full", &dir), Errno::ENOTEMPTY);
    }

    #[ktest]
    fn an_empty_directory_can_be_removed() {
        let (_fs, root) = mount();
        let dir = root
            .create("empty", InodeType::Dir, mode(0o755))
            .expect("should create");
        let links_before = root.link_count();

        root.rmdir("empty", &dir).expect("should remove");

        assert_errno(root.lookup("empty"), Errno::ENOENT);
        assert_eq!(root.link_count(), links_before - 1);
    }

    #[ktest]
    fn a_hard_link_is_a_second_name_for_one_file() {
        let (_fs, root) = mount();
        let file = root
            .create("original", InodeType::File, mode(0o644))
            .expect("should create");

        root.link(&file, "another").expect("should link");

        assert_eq!(
            root.lookup("another").expect("should resolve").ino(),
            file.ino()
        );
        assert_eq!(file.link_count(), 2);
    }

    /// Removing one name of a file leaves the other working.
    #[ktest]
    fn unlinking_one_name_leaves_the_other() {
        let (_fs, root) = mount();
        let file = root
            .create("name-a", InodeType::File, mode(0o644))
            .expect("should create");
        root.link(&file, "name-b").expect("should link");

        root.unlink("name-a", &file).expect("should unlink");

        assert_errno(root.lookup("name-a"), Errno::ENOENT);
        assert_eq!(
            root.lookup("name-b").expect("should resolve").ino(),
            file.ino()
        );
        assert_eq!(file.link_count(), 1);
    }

    #[ktest]
    fn a_directory_may_not_be_hard_linked() {
        let (_fs, root) = mount();
        let dir = root
            .create("dir", InodeType::Dir, mode(0o755))
            .expect("should create");

        assert_errno(root.link(&dir, "alias"), Errno::EPERM);
    }

    /// A short target lives in the inode and a long one in a block, so both
    /// have to be written and read back.
    #[ktest]
    fn both_kinds_of_symlink_read_back() {
        let (_fs, root) = mount();
        let long = "l".repeat(100);

        let short_link = root
            .create_symlink("short", "target")
            .expect("should create");
        let long_link = root.create_symlink("long", &long).expect("should create");

        assert_eq!(short_link.read_link().expect("should read"), "target");
        assert_eq!(long_link.read_link().expect("should read"), long);
    }

    /// A directory outgrows its first block once enough names are in it, which
    /// is where adding an entry has to allocate rather than find room.
    ///
    /// The names are links to one file, because the fixture volume holds far
    /// fewer inodes than it does room for names.
    #[ktest]
    fn a_directory_grows_a_block_when_the_first_one_fills() {
        let (_fs, root) = mount();
        let dir = root
            .create("growing", InodeType::Dir, mode(0o755))
            .expect("should create");
        let file = dir
            .create("first", InodeType::File, mode(0o644))
            .expect("should create");

        // A name this long costs 100 bytes a record, so forty of them fill a
        // block.
        let padding = "n".repeat(80);
        for index in 0..45 {
            let name = alloc::format!("{}-{:03}", padding, index);
            dir.link(&file, &name).expect("should link");
        }

        assert!(
            dir.file_size() > BLOCK_SIZE,
            "the directory should have taken a second block"
        );
        let mut names: Vec<String> = Vec::new();
        dir.readdir_at(0, &mut names).expect("should list");
        // The links, the file they name, and `.` and `..`.
        assert_eq!(names.len(), 48);
    }
}
