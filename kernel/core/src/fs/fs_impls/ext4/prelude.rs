// SPDX-License-Identifier: MPL-2.0

//! Re-exports used throughout the ext4 module.

pub(super) use core::{ops::Range, time::Duration};

pub(super) use align_ext::AlignExt;
pub(super) use aster_block::{
    BLOCK_SIZE, BlockDevice, SECTOR_SIZE,
    bio::{BioDirection, BioSegment, BioStatus},
    id::Bid,
};
pub(super) use io_util::batch::IoBatch;
pub(super) use ostd::{
    mm::{VmIo, VmIoFill, io::util::HasVmReaderWriter},
    sync::{RwMutex, RwMutexReadGuard},
};

pub(super) use super::inode::{Ext4Bid, Ext4Ino, Iblock};
pub(super) use crate::{
    fs::{
        file::InodeType,
        utils::{DirentVisitor, Dirty},
    },
    prelude::*,
    vm::page_cache::{BlockAsPageCacheBackend, PageCache, PageCacheBackend},
};
