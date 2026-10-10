#!/usr/bin/env bash

# SPDX-License-Identifier: MPL-2.0

# Creates an ext4 image and fills it with a known tree.
#
# The image is made by `mkfs.ext4` and populated by `debugfs`, so it carries
# exactly the features and layout that ordinary Linux tooling produces, which
# is what makes it worth testing against. `debugfs` writes the image directly,
# so neither step needs root or a loop device.
#
# The tree this creates is read by `test/initramfs/src/regression/fs/ext4`.
# Change the two together.

set -e

IMAGE="$1"
SIZE="${2:-64M}"

if [ -z "$IMAGE" ]; then
    echo "usage: $0 <image> [size]" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# `truncate` leaves the image sparse, so an unused gigabyte costs nothing.
rm -f "$IMAGE"
truncate -s "$SIZE" "$IMAGE"
mkfs.ext4 -q -F -b 4096 -I 256 -L asterinas-ext4 "$IMAGE"

printf 'hello from a real ext4 volume\n' > "$WORK_DIR/hello.txt"
printf 'nested\n' > "$WORK_DIR/nested.txt"
printf 'leaf\n' > "$WORK_DIR/leaf.txt"
printf 'x' > "$WORK_DIR/one-byte"

# A file of many blocks, which `mkfs.ext4` lays out as one extent and which a
# reader therefore has to serve as a run rather than block by block.
yes 'asterinas ext4 regression payload' 2>/dev/null \
    | head -c $((1024 * 1024)) > "$WORK_DIR/large.bin"

# A target of more than 59 bytes does not fit inside an inode, so it needs a
# data block and the slow path that reads one.
LONG_TARGET="$(printf 'd%.0s' $(seq 1 70))"

{
    echo "write $WORK_DIR/hello.txt hello.txt"
    echo "write $WORK_DIR/large.bin large.bin"
    echo "write $WORK_DIR/one-byte one-byte"
    echo "ln hello.txt hardlink.txt"
    echo "sif hello.txt links_count 2"
    echo "symlink link hello.txt"
    echo "symlink longlink $LONG_TARGET"
    echo "mkdir /dir"
    echo "write $WORK_DIR/nested.txt /dir/nested.txt"
    echo "mkdir /dir/deep"
    echo "write $WORK_DIR/leaf.txt /dir/deep/leaf.txt"
    echo "mkdir /many"
    # Enough entries that the directory outgrows a single block.
    for i in $(seq 1 400); do
        echo "write $WORK_DIR/one-byte /many/entry-$(printf '%03d' "$i")"
    done
} > "$WORK_DIR/commands"

debugfs -w -f "$WORK_DIR/commands" "$IMAGE" > /dev/null 2>&1

# A tree this driver cannot read is not worth testing against, and `e2fsck` is
# the arbiter of whether `debugfs` left it consistent.
e2fsck -fn "$IMAGE" > /dev/null
