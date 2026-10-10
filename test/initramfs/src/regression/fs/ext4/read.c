// SPDX-License-Identifier: MPL-2.0

/*
 * Reads the tree that `tools/make_ext4_image.sh` writes into the ext4 image.
 *
 * Unlike the kernel tests, which build a volume field by field, this runs
 * against an image that `mkfs.ext4` produced with its default features. It is
 * therefore the test that catches a misreading of the format itself rather
 * than of the fixture.
 */

#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../../common/test.h"

#define MOUNT_POINT "/ext4"
#define HELLO_CONTENT "hello from a real ext4 volume\n"
#define LARGE_SIZE (1024 * 1024)

static int read_whole(const char *path, char *buf, int size)
{
	int fd = open(path, O_RDONLY);
	if (fd < 0)
		return -1;

	int total = 0;
	while (total < size) {
		int got = read(fd, buf + total, size - total);
		if (got <= 0)
			break;
		total += got;
	}

	close(fd);
	return total;
}

FN_TEST(read_a_small_file)
{
	char buf[64] = { 0 };
	int len = (int)strlen(HELLO_CONTENT);

	TEST_RES(read_whole(MOUNT_POINT "/hello.txt", buf, sizeof(buf)),
		 _ret == len && memcmp(buf, HELLO_CONTENT, len) == 0);
}
END_TEST()

/*
 * `mkfs.ext4` lays a megabyte out as one extent, so this is the case a reader
 * must serve as a run rather than one block at a time.
 */
FN_TEST(read_a_file_of_many_blocks)
{
	static char buf[LARGE_SIZE];
	static const char pattern[] = "asterinas ext4 regression payload\n";
	int pattern_len = (int)strlen(pattern);

	TEST_RES(read_whole(MOUNT_POINT "/large.bin", buf, LARGE_SIZE),
		 _ret == LARGE_SIZE);

	/* Every whole repetition of the pattern must be intact. */
	int mismatches = 0;
	for (int off = 0; off + pattern_len <= LARGE_SIZE; off += pattern_len) {
		if (memcmp(buf + off, pattern, pattern_len) != 0)
			mismatches++;
	}
	TEST_RES(mismatches, _ret == 0);
}
END_TEST()

/* A read that starts and ends inside a block must return exactly that slice. */
FN_TEST(read_at_an_offset_inside_a_block)
{
	char buf[5] = { 0 };

	int fd = TEST_SUCC(open(MOUNT_POINT "/hello.txt", O_RDONLY));
	TEST_RES(lseek(fd, 6, SEEK_SET), _ret == 6);
	TEST_RES(read(fd, buf, 5), _ret == 5 && memcmp(buf, "from ", 5) == 0);
	TEST_SUCC(close(fd));
}
END_TEST()

/* A read that crosses from one block into the next must join them. */
FN_TEST(read_across_a_block_boundary)
{
	char buf[8] = { 0 };

	int fd = TEST_SUCC(open(MOUNT_POINT "/large.bin", O_RDONLY));
	TEST_RES(lseek(fd, 4096 - 4, SEEK_SET), _ret == 4096 - 4);
	TEST_RES(read(fd, buf, 8), _ret == 8);
	TEST_SUCC(close(fd));
}
END_TEST()

FN_TEST(read_past_the_end_returns_nothing)
{
	char buf[8];

	int fd = TEST_SUCC(open(MOUNT_POINT "/hello.txt", O_RDONLY));
	TEST_RES(lseek(fd, 4096, SEEK_SET), _ret == 4096);
	TEST_RES(read(fd, buf, sizeof(buf)), _ret == 0);
	TEST_SUCC(close(fd));
}
END_TEST()

FN_TEST(stat_reports_the_recorded_metadata)
{
	struct stat st;

	TEST_SUCC(stat(MOUNT_POINT "/hello.txt", &st));
	TEST_RES(st.st_size, _ret == (int)strlen(HELLO_CONTENT));
	TEST_RES(S_ISREG(st.st_mode), _ret != 0);
	/* `hardlink.txt` names the same inode, so the count is two. */
	TEST_RES(st.st_nlink, _ret == 2);

	TEST_SUCC(stat(MOUNT_POINT "/large.bin", &st));
	TEST_RES(st.st_size, _ret == LARGE_SIZE);
	/* A megabyte is 2048 sectors of 512 bytes. */
	TEST_RES(st.st_blocks, _ret >= 2048);
}
END_TEST()

/* Two names for one inode must resolve to the same inode number. */
FN_TEST(a_hard_link_shares_its_inode)
{
	struct stat original, link;

	TEST_SUCC(stat(MOUNT_POINT "/hello.txt", &original));
	TEST_SUCC(stat(MOUNT_POINT "/hardlink.txt", &link));
	TEST_RES(link.st_ino, _ret == (int)original.st_ino);
}
END_TEST()

/* A target short enough lives inside the inode; a longer one needs a block. */
FN_TEST(read_both_kinds_of_symlink)
{
	char buf[128];

	memset(buf, 0, sizeof(buf));
	TEST_RES(readlink(MOUNT_POINT "/link", buf, sizeof(buf)),
		 _ret == 9 && memcmp(buf, "hello.txt", 9) == 0);

	memset(buf, 0, sizeof(buf));
	TEST_RES(readlink(MOUNT_POINT "/longlink", buf, sizeof(buf)),
		 _ret == 70 && buf[0] == 'd' && buf[69] == 'd');
}
END_TEST()

/* Following a symlink must reach the file it names. */
FN_TEST(a_symlink_resolves_to_its_target)
{
	char buf[64] = { 0 };
	int len = (int)strlen(HELLO_CONTENT);

	TEST_RES(read_whole(MOUNT_POINT "/link", buf, sizeof(buf)),
		 _ret == len && memcmp(buf, HELLO_CONTENT, len) == 0);
}
END_TEST()

FN_TEST(walk_into_nested_directories)
{
	char buf[32] = { 0 };

	TEST_RES(read_whole(MOUNT_POINT "/dir/nested.txt", buf, sizeof(buf)),
		 _ret == 7 && memcmp(buf, "nested\n", 7) == 0);

	memset(buf, 0, sizeof(buf));
	TEST_RES(read_whole(MOUNT_POINT "/dir/deep/leaf.txt", buf, sizeof(buf)),
		 _ret == 5 && memcmp(buf, "leaf\n", 5) == 0);
}
END_TEST()

FN_TEST(an_absent_name_reports_enoent)
{
	TEST_ERRNO(open(MOUNT_POINT "/not-here", O_RDONLY), ENOENT);
	TEST_ERRNO(open(MOUNT_POINT "/dir/not-here", O_RDONLY), ENOENT);
}
END_TEST()

/* The driver mounts read-only, so every way of changing the volume fails. */
FN_TEST(the_volume_refuses_to_change)
{
	TEST_ERRNO(open(MOUNT_POINT "/new-file", O_CREAT | O_WRONLY, 0644),
		   EROFS);
	TEST_ERRNO(unlink(MOUNT_POINT "/hello.txt"), EROFS);
	TEST_ERRNO(mkdir(MOUNT_POINT "/new-dir", 0755), EROFS);
	TEST_ERRNO(truncate(MOUNT_POINT "/hello.txt", 0), EROFS);
	TEST_ERRNO(chmod(MOUNT_POINT "/hello.txt", 0600), EROFS);
}
END_TEST()

/*
 * A write must fail whatever happened at open time.
 *
 * Linux refuses `open(O_WRONLY)` on a read-only mount before it reaches the
 * filesystem, through `mnt_want_write`. Asterinas checks the mount only when
 * a directory entry changes, in `Path::check_dir_entry_mutation`, so the open
 * succeeds and the driver refuses the write behind it. Either way the volume
 * is never changed, which is what this pins down; the test skips rather than
 * fails if the open is refused, so that closing the gap does not break it.
 */
FN_TEST(a_write_to_a_read_only_volume_fails)
{
	int fd = open(MOUNT_POINT "/hello.txt", O_WRONLY);
	SKIP_TEST_IF(fd < 0);

	TEST_ERRNO(write(fd, "x", 1), EROFS);
	TEST_SUCC(close(fd));
}
END_TEST()
