#!/bin/bash

# Find all .sh files in the repository and update the index
find . -name '*.sh' | while read file; do
    git update-index --skip-worktree "$file"
    echo "Skipped: $file"
done

# Verify which files are now marked as skip-worktree
echo "Files marked as skip-worktree:"
git ls-files -v | grep '^S'

# To undo this later, you can use:
# find . -name '*.sh' | while read file; do
#     git update-index --no-skip-worktree "$file"
# done