// SPDX-License-Identifier: MPL-2.0

/*
 * Lists the directories of the ext4 image that `tools/make_ext4_image.sh`
 * builds, including one large enough to span several blocks.
 */

#include <dirent.h>
#include <string.h>
#include <sys/stat.h>

#include "../../common/test.h"

#define MOUNT_POINT "/ext4"

/* Counts the entries of a directory, excluding "." and "..". */
static int count_entries(const char *path)
{
	DIR *dir = opendir(path);
	if (dir == NULL)
		return -1;

	int count = 0;
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL) {
		if (strcmp(entry->d_name, ".") == 0 ||
		    strcmp(entry->d_name, "..") == 0)
			continue;
		count++;
	}

	closedir(dir);
	return count;
}

static int has_entry(const char *path, const char *name)
{
	DIR *dir = opendir(path);
	if (dir == NULL)
		return 0;

	int found = 0;
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL) {
		if (strcmp(entry->d_name, name) == 0) {
			found = 1;
			break;
		}
	}

	closedir(dir);
	return found;
}

FN_TEST(the_root_holds_what_was_put_there)
{
	TEST_RES(has_entry(MOUNT_POINT, "hello.txt"), _ret == 1);
	TEST_RES(has_entry(MOUNT_POINT, "large.bin"), _ret == 1);
	TEST_RES(has_entry(MOUNT_POINT, "link"), _ret == 1);
	TEST_RES(has_entry(MOUNT_POINT, "dir"), _ret == 1);
	TEST_RES(has_entry(MOUNT_POINT, "many"), _ret == 1);
	TEST_RES(has_entry(MOUNT_POINT, "lost+found"), _ret == 1);
	TEST_RES(has_entry(MOUNT_POINT, "not-here"), _ret == 0);
}
END_TEST()

/*
 * A directory of more than one block is where a reader that stops at the end
 * of the first block loses entries without failing.
 */
FN_TEST(a_directory_of_several_blocks_lists_every_entry)
{
	TEST_RES(count_entries(MOUNT_POINT "/many"), _ret == 400);
	TEST_RES(has_entry(MOUNT_POINT "/many", "entry-001"), _ret == 1);
	TEST_RES(has_entry(MOUNT_POINT "/many", "entry-400"), _ret == 1);
}
END_TEST()

/*
 * `lost+found` is preallocated to several blocks and then left empty, so its
 * records are the ones a reader must skip rather than report.
 */
FN_TEST(an_empty_but_wide_directory_lists_nothing)
{
	TEST_RES(count_entries(MOUNT_POINT "/lost+found"), _ret == 0);
}
END_TEST()

FN_TEST(readdir_reports_the_type_of_each_entry)
{
	DIR *dir = opendir(MOUNT_POINT);
	TEST_RES(dir != NULL, _ret == 1);

	int saw_file = 0, saw_dir = 0, saw_link = 0;
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL) {
		if (strcmp(entry->d_name, "hello.txt") == 0)
			saw_file = (entry->d_type == DT_REG);
		if (strcmp(entry->d_name, "dir") == 0)
			saw_dir = (entry->d_type == DT_DIR);
		if (strcmp(entry->d_name, "link") == 0)
			saw_link = (entry->d_type == DT_LNK);
	}
	closedir(dir);

	TEST_RES(saw_file, _ret == 1);
	TEST_RES(saw_dir, _ret == 1);
	TEST_RES(saw_link, _ret == 1);
}
END_TEST()

FN_TEST(listing_a_file_reports_enotdir)
{
	TEST_ERRNO(opendir(MOUNT_POINT "/hello.txt") == NULL ? -1 : 0, ENOTDIR);
}
END_TEST()
