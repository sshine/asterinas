// SPDX-License-Identifier: MPL-2.0

//! VFS trait implementations for the ext4 filesystem.
//!
//! This module bridges the ext4-specific types (`Ext4`, `Inode`) to the
//! generic VFS layer. The two child modules each contain `impl` blocks for the
//! corresponding VFS traits:
//!
//! * `fs` — `FileSystem` for `Ext4` (sync, stat, root).
//! * `inode` — `FileOps` and `Inode` for `Inode` (I/O, metadata, lookup).
//!
//! An operation that would change a volume mounted read-only reports `EROFS`;
//! one this driver has not implemented at all reports `EOPNOTSUPP`.

mod fs;
mod inode;
