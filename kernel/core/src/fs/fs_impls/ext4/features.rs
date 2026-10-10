// SPDX-License-Identifier: MPL-2.0

//! Ext4 feature words and the mount policy they imply.
//!
//! The superblock carries three of them, and each sets a different obligation
//! on a driver. An incompatible feature changes the on-disk layout, so a
//! driver that does not implement it must refuse the volume. A
//! read-only-compatible feature constrains writing alone, so a driver that
//! cannot maintain it may still read the volume. A compatible feature may be
//! ignored outright.
//!
//! Unknown bits inherit the same obligations, which is why the masks below
//! list what is supported rather than what is rejected, and why the checks
//! run against the raw words rather than the truncated bitflags.
//!
//! # Policy
//!
//! A stock `mkfs.ext4` volume carries `has_journal` and `metadata_csum`, and
//! neither can be honoured by a write path that does not implement them.
//! Refusing such a volume would make the driver useless against anything
//! ordinary Linux tooling produces, so instead they force a read-only mount.
//! The three outcomes are:
//!
//! | Feature set                               | Outcome          |
//! |-------------------------------------------|------------------|
//! | incompatible, outside what is implemented | reject, `EINVAL` |
//! | read-only-compatible or journalling       | force read-only  |
//! | everything else                           | read-write       |
//!
//! No combination of bits can therefore produce a write that corrupts.

use super::prelude::*;
use crate::fs::vfs::file_system::FsFlags;

bitflags! {
    /// Features a driver may ignore.
    pub(super) struct FeatureCompat: u32 {
        const DIR_PREALLOC = 1 << 0;
        const IMAGIC_INODES = 1 << 1;
        const HAS_JOURNAL = 1 << 2;
        const EXT_ATTR = 1 << 3;
        const RESIZE_INODE = 1 << 4;
        const DIR_INDEX = 1 << 5;
        const SPARSE_SUPER2 = 1 << 9;
        const FAST_COMMIT = 1 << 10;
        const STABLE_INODES = 1 << 11;
        const ORPHAN_FILE = 1 << 12;
    }
}

bitflags! {
    /// Features that change the on-disk layout.
    pub(super) struct FeatureIncompat: u32 {
        const COMPRESSION = 1 << 0;
        const FILETYPE = 1 << 1;
        const RECOVER = 1 << 2;
        const JOURNAL_DEV = 1 << 3;
        const META_BG = 1 << 4;
        const EXTENTS = 1 << 6;
        const SIZE_64BIT = 1 << 7;
        const MMP = 1 << 8;
        const FLEX_BG = 1 << 9;
        const EA_INODE = 1 << 10;
        const DIRDATA = 1 << 12;
        const CSUM_SEED = 1 << 13;
        const LARGEDIR = 1 << 14;
        const INLINE_DATA = 1 << 15;
        const ENCRYPT = 1 << 16;
        const CASEFOLD = 1 << 17;

        /// Features this driver parses and will be able to maintain.
        const READ_WRITE = Self::FILETYPE.bits
            | Self::EXTENTS.bits
            | Self::SIZE_64BIT.bits
            | Self::FLEX_BG.bits;

        /// Features that change nothing this driver reads, but that it cannot
        /// keep correct while writing.
        ///
        /// `CSUM_SEED` moves the checksum seed from the UUID into the
        /// superblock. Nothing but a checksum depends on it, and this driver
        /// verifies none, so a volume carrying it reads exactly as one without.
        const READ_ONLY = Self::CSUM_SEED.bits;
    }
}

bitflags! {
    /// Features that constrain writing only.
    pub(super) struct FeatureRoCompat: u32 {
        const SPARSE_SUPER = 1 << 0;
        const LARGE_FILE = 1 << 1;
        const BTREE_DIR = 1 << 2;
        const HUGE_FILE = 1 << 3;
        const GDT_CSUM = 1 << 4;
        const DIR_NLINK = 1 << 5;
        const EXTRA_ISIZE = 1 << 6;
        const HAS_SNAPSHOT = 1 << 7;
        const QUOTA = 1 << 8;
        const BIGALLOC = 1 << 9;
        const METADATA_CSUM = 1 << 10;
        const REPLICA = 1 << 11;
        const READONLY = 1 << 12;
        const PROJECT = 1 << 13;
        const SHARED_BLOCKS = 1 << 14;
        const VERITY = 1 << 15;
        const ORPHAN_PRESENT = 1 << 16;

        /// Features this driver will be able to maintain while writing.
        const READ_WRITE = Self::SPARSE_SUPER.bits
            | Self::LARGE_FILE.bits
            | Self::HUGE_FILE.bits
            | Self::DIR_NLINK.bits
            | Self::EXTRA_ISIZE.bits;
    }
}

/// Rejects a volume whose incompatible features change a layout this driver
/// cannot parse.
///
/// The error is `EINVAL` in particular, because the root filesystem candidate
/// list treats any other error as fatal rather than moving on to the next type.
pub(super) fn check_incompat(incompat: u32) -> Result<()> {
    let supported = FeatureIncompat::READ_WRITE.bits() | FeatureIncompat::READ_ONLY.bits();
    if (incompat & !supported) != 0 {
        return_errno_with_message!(Errno::EINVAL, "unsupported incompat feature");
    }

    Ok(())
}

/// Returns the flags a volume declaring these features may be mounted with.
///
/// Call only after [`check_incompat`] has accepted the volume.
pub(super) fn mount_flags(
    compat: u32,
    incompat: u32,
    ro_compat: u32,
    requested: FsFlags,
) -> FsFlags {
    // A journal may hold transactions that were never checkpointed, and
    // writing past an unreplayed journal loses them.
    let has_journal = (compat & FeatureCompat::HAS_JOURNAL.bits()) != 0;
    let writes_unsupported_incompat = (incompat & FeatureIncompat::READ_ONLY.bits()) != 0;
    let writes_unsupported_ro_compat = (ro_compat & !FeatureRoCompat::READ_WRITE.bits()) != 0;

    if has_journal || writes_unsupported_incompat || writes_unsupported_ro_compat {
        requested | FsFlags::RDONLY
    } else {
        requested
    }
}

#[cfg(ktest)]
mod test {
    use ostd::prelude::*;

    use super::*;
    use crate::fs::fs_impls::ext4::test_utils::assert_errno;

    /// The feature words `mkfs.ext4` writes with no options given.
    const STOCK_COMPAT: u32 = 0x103c;
    const STOCK_INCOMPAT: u32 = 0x22c2;
    const STOCK_RO_COMPAT: u32 = 0x46b;

    /// The feature words left by suppressing everything this driver cannot
    /// maintain: `-O ^has_journal,^metadata_csum,^metadata_csum_seed`.
    const REDUCED_COMPAT: u32 = 0x38;
    const REDUCED_INCOMPAT: u32 = 0x2c2;
    const REDUCED_RO_COMPAT: u32 = 0x6b;

    #[ktest]
    fn a_stock_volume_mounts_read_only() {
        let flags = mount_flags(
            STOCK_COMPAT,
            STOCK_INCOMPAT,
            STOCK_RO_COMPAT,
            FsFlags::empty(),
        );

        assert!(flags.contains(FsFlags::RDONLY));
    }

    #[ktest]
    fn a_reduced_volume_mounts_read_write() {
        let flags = mount_flags(
            REDUCED_COMPAT,
            REDUCED_INCOMPAT,
            REDUCED_RO_COMPAT,
            FsFlags::empty(),
        );

        assert!(!flags.contains(FsFlags::RDONLY));
    }

    /// A caller that asked for a read-only mount gets one whatever the volume
    /// would have allowed.
    #[ktest]
    fn a_requested_read_only_mount_is_never_widened() {
        let flags = mount_flags(
            REDUCED_COMPAT,
            REDUCED_INCOMPAT,
            REDUCED_RO_COMPAT,
            FsFlags::RDONLY,
        );

        assert!(flags.contains(FsFlags::RDONLY));
    }

    /// An unknown read-only-compatible bit must force read-only rather than
    /// reject, which is the half of the policy a whitelist gets wrong when it
    /// is written as a blacklist.
    #[ktest]
    fn an_unknown_ro_compat_feature_forces_read_only() {
        let flags = mount_flags(
            REDUCED_COMPAT,
            REDUCED_INCOMPAT,
            REDUCED_RO_COMPAT | (1 << 20),
            FsFlags::empty(),
        );

        assert!(flags.contains(FsFlags::RDONLY));
    }

    #[ktest]
    fn an_unrecoverable_journal_is_rejected() {
        assert_errno(
            check_incompat(REDUCED_INCOMPAT | FeatureIncompat::RECOVER.bits()),
            Errno::EINVAL,
        );
    }

    #[ktest]
    fn an_unknown_incompat_feature_is_rejected() {
        assert_errno(check_incompat(REDUCED_INCOMPAT | (1 << 20)), Errno::EINVAL);
    }

    #[ktest]
    fn a_stock_incompat_word_is_accepted() {
        assert!(check_incompat(STOCK_INCOMPAT).is_ok());
    }
}
