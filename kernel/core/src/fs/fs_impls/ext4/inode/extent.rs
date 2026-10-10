// SPDX-License-Identifier: MPL-2.0

//! The extent tree that maps a file's logical blocks onto the device.
//!
//! Where ext2 records one device block per logical block, ext4 records runs:
//! an extent names a logical block, a device block, and a length, so a
//! contiguous file costs one record however large it is. The records live in a
//! B-tree whose root is the inode's 60-byte `i_block` area and whose interior
//! and leaf nodes each occupy one block.
//!
//! # Node layout
//!
//! Every node begins with a 12-byte header and is followed by 12-byte entries.
//! A header with `depth == 0` is a leaf and its entries are extents; a header
//! with a greater depth is an index node and its entries name the child that
//! covers a range of logical blocks. Entries are sorted by logical block.
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │ magic │ entries │ max │ depth │ generation │ entry … │
//! └────────────────────────────────────────────────────────┘
//!   2 B      2 B      2 B    2 B       4 B        12 B each
//! ```
//!
//! # Unwritten extents
//!
//! An extent whose length exceeds 32768 is unwritten: the blocks are allocated
//! but were never written, so they read as zeros. `fallocate` produces them,
//! and keeping them distinct from holes is what will let a later write convert
//! one in place rather than allocating again.
//!
//! # Validation
//!
//! Every node is checked before its entries are used: the magic number, the
//! entry count against the node's capacity, the depth against the parent's,
//! the logical blocks against the order the format requires, and each device
//! address against the size of the volume. A malformed tree yields `EUCLEAN`,
//! never a panic and never a read outside the volume.

use ostd::const_assert;

use crate::fs::fs_impls::ext4::prelude::*;

/// Number of bytes of `i_block` that hold the root node.
pub(super) const EXTENT_ROOT_LEN: usize = 60;

const EXTENT_MAGIC: u16 = 0xf30a;

/// The deepest tree ext4 defines.
const MAX_DEPTH: u16 = 5;

/// Lengths above this mark an extent as unwritten, and the excess is the
/// actual length.
const UNWRITTEN_MARK: u16 = 32768;

/// What backs a run of logical blocks starting at a requested one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Mapping {
    /// `len` logical blocks sit at `start` and onwards on the device.
    Mapped { start: Ext4Bid, len: u32 },
    /// `len` logical blocks are allocated but never written, and read as zeros.
    Unwritten { len: u32 },
    /// `len` logical blocks have no mapping at all, and read as zeros.
    Hole { len: u32 },
}

impl Mapping {
    /// Returns how many logical blocks this mapping covers.
    pub(super) fn len(&self) -> u32 {
        match *self {
            Self::Mapped { len, .. } | Self::Unwritten { len } | Self::Hole { len } => len,
        }
    }
}

/// A file's extent tree, rooted in the inode.
#[derive(Debug)]
pub(super) struct ExtentTree {
    root: [u8; EXTENT_ROOT_LEN],
    /// Size of the volume, used to reject an address pointing off the device.
    volume_blocks: u64,
}

impl ExtentTree {
    /// Creates a tree from an inode's `i_block` area.
    pub(super) fn new(root: [u8; EXTENT_ROOT_LEN], volume_blocks: u64) -> Result<Self> {
        let tree = Self {
            root,
            volume_blocks,
        };
        // Reject a root the moment the inode is read, rather than on the first
        // access to a file that was never readable.
        Node::new(&tree.root, None)?;
        Ok(tree)
    }

    /// Returns what backs the logical blocks starting at `iblock`.
    ///
    /// The result covers at most `max_blocks` and never fewer than one block,
    /// so a caller can walk a range by advancing over each result in turn.
    pub(super) fn lookup(
        &self,
        device: &dyn BlockDevice,
        iblock: Iblock,
        max_blocks: u32,
    ) -> Result<Mapping> {
        debug_assert!(max_blocks > 0);

        // The logical block at which the subtree being descended into ends.
        // Anything past it belongs to a sibling, so a hole must stop there.
        let mut boundary = None;
        let mut block = Vec::new();
        let mut expected_depth = None;

        for level in 0..=MAX_DEPTH {
            // The node borrows `block`, which the descent then overwrites, so
            // it must not outlive the step that reads it.
            let (child, child_depth) = {
                let node = Node::new(if level == 0 { &self.root } else { &block }, expected_depth)?;

                if node.depth() == 0 {
                    return self.lookup_in_leaf(&node, iblock, max_blocks, boundary);
                }

                match node.find_child(iblock, &mut boundary)? {
                    Descend::Child(bid) => (bid, node.depth() - 1),
                    Descend::Hole => {
                        return Ok(Mapping::Hole {
                            len: hole_len(iblock, max_blocks, boundary),
                        });
                    }
                }
            };

            self.check_address(child, 1)?;
            block = vec![0u8; BLOCK_SIZE];
            device
                .read_bytes(Bid::new(child).to_offset(), &mut block)
                .map_err(|_| Error::with_message(Errno::EIO, "failed to read extent node"))?;
            expected_depth = Some(child_depth);
        }

        return_errno_with_message!(
            Errno::EUCLEAN,
            "extent tree is deeper than the format allows"
        )
    }

    fn lookup_in_leaf(
        &self,
        leaf: &Node<'_>,
        iblock: Iblock,
        max_blocks: u32,
        boundary: Option<Iblock>,
    ) -> Result<Mapping> {
        let mut previous_end = None;
        for entry_idx in 0..leaf.entries() {
            let extent = leaf.extent(entry_idx);
            let (len, unwritten) = extent.decoded_len()?;
            if len == 0 {
                return_errno_with_message!(Errno::EUCLEAN, "extent of zero length");
            }

            // Entries are sorted and may not overlap, so a later entry starting
            // before the previous one ended means the node is corrupt.
            if previous_end.is_some_and(|end| extent.block < end) {
                return_errno_with_message!(Errno::EUCLEAN, "extents are out of order");
            }
            previous_end = Some(extent.block + len as Iblock);

            if iblock < extent.block {
                // The request falls before this extent, so everything up to it
                // is a hole.
                return Ok(Mapping::Hole {
                    len: hole_len(iblock, max_blocks, Some(extent.block)),
                });
            }
            if iblock >= extent.block + len as Iblock {
                continue;
            }

            let offset = iblock - extent.block;
            let run = (len - offset).min(max_blocks);
            if unwritten {
                return Ok(Mapping::Unwritten { len: run });
            }

            let start = extent.start();
            self.check_address(start, len as u64)?;
            return Ok(Mapping::Mapped {
                start: start + offset as u64,
                len: run,
            });
        }

        Ok(Mapping::Hole {
            len: hole_len(iblock, max_blocks, boundary),
        })
    }

    /// Rejects a run of device blocks that does not lie within the volume.
    fn check_address(&self, start: Ext4Bid, len: u64) -> Result<()> {
        if start == 0 || start.saturating_add(len) > self.volume_blocks {
            return_errno_with_message!(Errno::EUCLEAN, "extent points outside the volume");
        }

        Ok(())
    }
}

/// Returns the length of a hole that starts at `iblock`.
///
/// A hole must never be reported longer than it is, because its blocks are
/// served as zeros without consulting the device. It therefore stops at the
/// first logical block known to belong to something else.
fn hole_len(iblock: Iblock, max_blocks: u32, boundary: Option<Iblock>) -> u32 {
    match boundary {
        Some(boundary) if boundary > iblock => (boundary - iblock).min(max_blocks),
        Some(_) => 1,
        None => max_blocks,
    }
}

/// Where a descent through an index node leads.
enum Descend {
    Child(Ext4Bid),
    /// No child covers the requested block.
    Hole,
}

/// A validated extent tree node borrowing the bytes that hold it.
struct Node<'a> {
    bytes: &'a [u8],
    header: RawExtentHeader,
}

impl<'a> Node<'a> {
    /// Validates the node in `bytes`, which must sit at `expected_depth` when
    /// its parent states one.
    fn new(bytes: &'a [u8], expected_depth: Option<u16>) -> Result<Self> {
        let header = RawExtentHeader::from_first_bytes(bytes);
        if header.magic != EXTENT_MAGIC {
            return_errno_with_message!(Errno::EUCLEAN, "bad extent node magic number");
        }
        if header.depth > MAX_DEPTH {
            return_errno_with_message!(Errno::EUCLEAN, "extent node is too deep");
        }
        if expected_depth.is_some_and(|depth| depth != header.depth) {
            return_errno_with_message!(Errno::EUCLEAN, "extent node depth does not match parent");
        }
        if header.entries > header.max {
            return_errno_with_message!(
                Errno::EUCLEAN,
                "extent node holds more entries than it may"
            );
        }

        let needed = size_of::<RawExtentHeader>() + header.max as usize * ENTRY_LEN;
        if needed > bytes.len() {
            return_errno_with_message!(Errno::EUCLEAN, "extent node does not fit its block");
        }

        Ok(Self { bytes, header })
    }

    fn depth(&self) -> u16 {
        self.header.depth
    }

    fn entries(&self) -> usize {
        self.header.entries as usize
    }

    fn entry_bytes(&self, entry_idx: usize) -> &[u8] {
        let offset = size_of::<RawExtentHeader>() + entry_idx * ENTRY_LEN;
        &self.bytes[offset..offset + ENTRY_LEN]
    }

    fn extent(&self, entry_idx: usize) -> RawExtent {
        RawExtent::from_bytes(self.entry_bytes(entry_idx))
    }

    fn index(&self, entry_idx: usize) -> RawExtentIdx {
        RawExtentIdx::from_bytes(self.entry_bytes(entry_idx))
    }

    /// Returns the child covering `iblock`, narrowing `boundary` to the
    /// logical block at which that child's range ends.
    fn find_child(&self, iblock: Iblock, boundary: &mut Option<Iblock>) -> Result<Descend> {
        let mut found = None;
        for entry_idx in 0..self.entries() {
            let index = self.index(entry_idx);
            if entry_idx > 0 && index.block <= self.index(entry_idx - 1).block {
                return_errno_with_message!(Errno::EUCLEAN, "index entries are out of order");
            }

            if index.block <= iblock {
                found = Some(index);
                continue;
            }

            // The first entry past the request ends the range of the one
            // before it, whether or not that one exists.
            narrow(boundary, index.block);
            break;
        }

        match found {
            Some(index) => Ok(Descend::Child(index.leaf())),
            None => Ok(Descend::Hole),
        }
    }
}

fn narrow(boundary: &mut Option<Iblock>, candidate: Iblock) {
    *boundary = Some(match *boundary {
        Some(current) => current.min(candidate),
        None => candidate,
    });
}

const ENTRY_LEN: usize = 12;

/// The header every extent tree node begins with.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
struct RawExtentHeader {
    magic: u16,
    entries: u16,
    max: u16,
    depth: u16,
    generation: u32,
}

/// An entry of an index node, naming the block that holds a child node.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
struct RawExtentIdx {
    block: u32,
    leaf_lo: u32,
    leaf_hi: u16,
    unused: u16,
}

impl RawExtentIdx {
    fn leaf(&self) -> Ext4Bid {
        ((self.leaf_hi as u64) << 32) | self.leaf_lo as u64
    }
}

/// An entry of a leaf node, mapping a run of logical blocks onto the device.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod)]
struct RawExtent {
    block: u32,
    len: u16,
    start_hi: u16,
    start_lo: u32,
}

impl RawExtent {
    fn start(&self) -> Ext4Bid {
        ((self.start_hi as u64) << 32) | self.start_lo as u64
    }

    /// Returns the length in blocks and whether the extent is unwritten.
    fn decoded_len(&self) -> Result<(u32, bool)> {
        if self.len <= UNWRITTEN_MARK {
            return Ok((self.len as u32, false));
        }

        let len = self.len - UNWRITTEN_MARK;
        if len > UNWRITTEN_MARK {
            return_errno_with_message!(Errno::EUCLEAN, "extent is longer than the format allows");
        }
        Ok((len as u32, true))
    }
}

const_assert!(size_of::<RawExtentHeader>() == ENTRY_LEN);
const_assert!(size_of::<RawExtentIdx>() == ENTRY_LEN);
const_assert!(size_of::<RawExtent>() == ENTRY_LEN);

#[cfg(ktest)]
mod test {
    use ostd::prelude::ktest;

    use super::*;
    use crate::fs::fs_impls::ext4::test_utils::{
        Ext4MemoryDisk, assert_errno, extent_index_block, extent_index_root,
        extent_index_root_at_depth, extent_leaf_block, extent_root,
    };

    /// Blocks on the scratch device, which is large enough for every address
    /// these tests use.
    const VOLUME_BLOCKS: u64 = 64;

    fn scratch_disk() -> Ext4MemoryDisk {
        Ext4MemoryDisk::new(VOLUME_BLOCKS as usize)
    }

    fn lookup(
        root: [u8; EXTENT_ROOT_LEN],
        disk: &Ext4MemoryDisk,
        iblock: Iblock,
        max_blocks: u32,
    ) -> Result<Mapping> {
        ExtentTree::new(root, VOLUME_BLOCKS)?.lookup(disk, iblock, max_blocks)
    }

    #[ktest]
    fn one_extent_covers_a_run_of_blocks() {
        let disk = scratch_disk();
        let root = extent_root(&[(0, 20, 4)]);

        let mapping = lookup(root, &disk, 1, 8).expect("should map");

        // The run is clipped to what the extent holds, not to what was asked.
        assert_eq!(mapping, Mapping::Mapped { start: 21, len: 3 });
    }

    /// Two extents are two runs even where their logical blocks are adjacent,
    /// which is what a reader must not merge across.
    #[ktest]
    fn adjacent_extents_stay_separate_runs() {
        let disk = scratch_disk();
        let root = extent_root(&[(0, 20, 1), (1, 40, 1)]);

        assert_eq!(
            lookup(root, &disk, 0, 2).expect("should map"),
            Mapping::Mapped { start: 20, len: 1 }
        );
        assert_eq!(
            lookup(root, &disk, 1, 1).expect("should map"),
            Mapping::Mapped { start: 40, len: 1 }
        );
    }

    /// A hole must stop where the next extent begins. Reporting it longer
    /// would serve a mapped block as zeros.
    #[ktest]
    fn a_hole_stops_at_the_next_extent() {
        let disk = scratch_disk();
        let root = extent_root(&[(0, 20, 1), (4, 40, 1)]);

        assert_eq!(
            lookup(root, &disk, 1, 16).expect("should map"),
            Mapping::Hole { len: 3 }
        );
    }

    #[ktest]
    fn a_hole_before_the_first_extent_stops_at_it() {
        let disk = scratch_disk();
        let root = extent_root(&[(2, 20, 1)]);

        assert_eq!(
            lookup(root, &disk, 0, 16).expect("should map"),
            Mapping::Hole { len: 2 }
        );
    }

    #[ktest]
    fn a_hole_past_the_last_extent_runs_to_the_end_of_the_request() {
        let disk = scratch_disk();
        let root = extent_root(&[(0, 20, 1)]);

        assert_eq!(
            lookup(root, &disk, 1, 5).expect("should map"),
            Mapping::Hole { len: 5 }
        );
    }

    #[ktest]
    fn an_unwritten_extent_is_told_apart_from_a_mapping() {
        let disk = scratch_disk();
        let root = extent_root(&[(0, 20, UNWRITTEN_MARK + 2)]);

        assert_eq!(
            lookup(root, &disk, 0, 4).expect("should map"),
            Mapping::Unwritten { len: 2 }
        );
    }

    #[ktest]
    fn a_tree_one_level_deep_is_walked_to_its_leaf() {
        let disk = scratch_disk();
        let leaf_bid = 8;
        disk.write_block(leaf_bid, &extent_leaf_block(&[(0, 20, 2)]));
        let root = extent_index_root(&[(0, leaf_bid)]);

        assert_eq!(
            lookup(root, &disk, 1, 1).expect("should map"),
            Mapping::Mapped { start: 21, len: 1 }
        );
    }

    /// Past the end of a leaf, the hole ends where the parent says the next
    /// child begins, which is the one boundary a leaf cannot see.
    #[ktest]
    fn a_hole_stops_at_the_next_index_entry() {
        let disk = scratch_disk();
        let (first_leaf, second_leaf) = (8, 9);
        disk.write_block(first_leaf, &extent_leaf_block(&[(0, 20, 1)]));
        disk.write_block(second_leaf, &extent_leaf_block(&[(8, 40, 1)]));
        let root = extent_index_root(&[(0, first_leaf), (8, second_leaf)]);

        assert_eq!(
            lookup(root, &disk, 1, 32).expect("should map"),
            Mapping::Hole { len: 7 }
        );
    }

    #[ktest]
    fn a_node_without_the_magic_number_is_rejected() {
        let disk = scratch_disk();
        let mut root = extent_root(&[(0, 20, 1)]);
        root[0] = 0;

        assert_errno(lookup(root, &disk, 0, 1), Errno::EUCLEAN);
    }

    #[ktest]
    fn a_node_claiming_more_entries_than_fit_is_rejected() {
        let disk = scratch_disk();
        let mut root = extent_root(&[(0, 20, 1)]);
        // Four entries fit in the inode; this claims a block's worth.
        root[2..4].copy_from_slice(&340u16.to_le_bytes());
        root[4..6].copy_from_slice(&340u16.to_le_bytes());

        assert_errno(lookup(root, &disk, 0, 1), Errno::EUCLEAN);
    }

    #[ktest]
    fn a_child_at_the_wrong_depth_is_rejected() {
        let disk = scratch_disk();
        let leaf_bid = 8;
        let mut leaf = extent_leaf_block(&[(0, 20, 1)]);
        // A child of a depth-1 node must be a leaf.
        leaf[6..8].copy_from_slice(&1u16.to_le_bytes());
        disk.write_block(leaf_bid, &leaf);

        assert_errno(
            lookup(extent_index_root(&[(0, leaf_bid)]), &disk, 0, 1),
            Errno::EUCLEAN,
        );
    }

    /// An address off the end of the volume would otherwise read whatever the
    /// device has there, or fail far from the cause.
    #[ktest]
    fn an_extent_outside_the_volume_is_rejected() {
        let disk = scratch_disk();
        let root = extent_root(&[(0, VOLUME_BLOCKS - 1, 4)]);

        assert_errno(lookup(root, &disk, 0, 4), Errno::EUCLEAN);
    }

    /// Two levels of index node above the leaves, which is where a walk that
    /// assumes a single level stops at an index entry and reads it as an
    /// extent.
    #[ktest]
    fn a_tree_two_levels_deep_is_walked_to_its_leaf() {
        let disk = scratch_disk();
        let (index_bid, leaf_bid) = (8, 9);
        disk.write_block(index_bid, &extent_index_block(&[(0, leaf_bid)], 1));
        disk.write_block(leaf_bid, &extent_leaf_block(&[(0, 20, 4)]));
        let root = extent_index_root_at_depth(&[(0, index_bid)], 2);

        assert_eq!(
            lookup(root, &disk, 2, 2).expect("should map"),
            Mapping::Mapped { start: 22, len: 2 }
        );
    }

    /// A tree deeper than the format allows must stop rather than follow a
    /// cycle that a corrupt image could describe.
    #[ktest]
    fn a_tree_deeper_than_the_format_allows_is_rejected() {
        let disk = scratch_disk();
        // Every node claims to be an index one level above the next, and the
        // chain never reaches a leaf.
        for bid in 8..16 {
            disk.write_block(bid, &extent_index_block(&[(0, bid + 1)], 1));
        }

        assert_errno(
            lookup(extent_index_root_at_depth(&[(0, 8)], 2), &disk, 0, 1),
            Errno::EUCLEAN,
        );
    }

    #[ktest]
    fn an_extent_at_block_zero_is_rejected() {
        let disk = scratch_disk();
        let root = extent_root(&[(0, 0, 1)]);

        assert_errno(lookup(root, &disk, 0, 1), Errno::EUCLEAN);
    }

    #[ktest]
    fn extents_out_of_order_are_rejected() {
        let disk = scratch_disk();
        let root = extent_root(&[(4, 20, 1), (0, 40, 1)]);

        assert_errno(lookup(root, &disk, 8, 1), Errno::EUCLEAN);
    }
}
