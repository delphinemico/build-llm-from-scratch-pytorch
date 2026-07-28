#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: ./new_chapter.sh chapter_XX"
    echo "Example: ./new_chapter.sh chapter_01"
    exit 1
fi

chapter_name="$1"

if [[ ! "$chapter_name" =~ ^chapter_[0-9]{2}$ ]]; then
    echo "Error: chapter name must follow the format chapter_XX."
    echo "Example: chapter_01"
    exit 1
fi

if [ ! -d "_template" ]; then
    echo "Error: _template directory was not found."
    echo "Run this script from the repository root."
    exit 1
fi

if [ -e "$chapter_name" ]; then
    echo "Error: '$chapter_name' already exists."
    exit 1
fi

cp -R "_template" "$chapter_name"

chapter_number="${chapter_name#chapter_}"
chapter_number=$((10#$chapter_number))

# Replace the generic template heading with the chapter number.
sed -i "s/Chapter XX/Chapter ${chapter_number}/g" \
    "$chapter_name/README.md" \
    "$chapter_name/RESULTS.md" \
    "$chapter_name/exercises/README.md"

echo "Created '$chapter_name' from _template."
echo
echo "Files created:"
find "$chapter_name" -maxdepth 3 -type f | sort