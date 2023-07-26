"""
Remove any file that matches the specific pattern (regex) from all the commits in the repo.

for matching any file that ends with .npy, use the pattern: .+\.npy$


"""

import os
import io
import re
import sys
from tqdm import tqdm
import subprocess
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = argparse.ArgumentParser(
    description="Remove any file that matches the specific pattern from all the commits in the repo."
)
parser.add_argument("-p", "--pattern", type=str, help="The pattern to match.")

args = parser.parse_args()

if not args.pattern:
    print("Please provide a pattern to match.")
    sys.exit(1)


def get_commits():
    """Get a list of all the commits in the repo."""
    log = logging.getLogger("get_commits")
    log.info("Getting commits.")
    commits = subprocess.check_output(["git", "log", "--pretty=format:%H"]).decode("utf-8").split("\n")
    log.info(f"Found {len(commits)} commits.")
    return commits


def get_commit_contents(commit):
    """Get the contents of a commit."""
    log = logging.getLogger("get_commit_contents")
    log.info(f"Getting contents of commit {commit}.")
    contents = subprocess.check_output(["git", "show", "--name-only", commit]).decode("utf-8")
    log.info(f"Found {len(contents)} lines in commit {commit}.")
    return contents


def files_changed_list(contents):
    """Get a list of all the files changed in a commit."""
    log = logging.getLogger("files_changed_list")
    log.info("Getting list of files changed.")

    lines = contents.split("\n")
    files = []
    for line in list(reversed(lines))[1:]:
        if line == "":
            break
        files.append(line)
    return files


def get_matches(files, pattern):
    """Get the files that match the pattern."""
    log = logging.getLogger("get_matches")
    log.info(f"Getting files that match pattern {pattern}.")
    matches = []
    for file in files:
        if re.match(pattern, file):
            matches.append(file)
    return matches


def remove_file(file):
    """Remove a file from the repo."""
    subprocess.run(
        [
            "git",
            "filter-branch",
            "--index-filter",
            f"git rm --cached --ignore-unmatch {file}",
            "--prune-empty",
            "-f",
            "--",
            "--all",
        ],
    )


commits = get_commits()
executor = ThreadPoolExecutor(max_workers=10)
futures = []
for commit in commits:
    contents = get_commit_contents(commit)
    files = files_changed_list(contents)
    matches = get_matches(files, args.pattern)
    if len(matches) == 0:
        continue
    for file in matches:
        executor.submit(remove_file, file)

# for root, dirs, files in os.walk("."):
#     for file in files:
#         if re.match(args.pattern, file):
#             print(os.path.join(root, file))
#             executor.submit(remove_file, os.path.join(root, file))

for future in tqdm(as_completed(futures), total=len(futures)):
    pass
