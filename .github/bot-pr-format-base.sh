#!/bin/bash

source .github/bot-pr-base.sh

EXTENSION_REGEX='\.(cuh?|hpp|hpp\.inc?|cpp)$'
FORMAT_HEADER_REGEX='^(benchmark|core|cuda|hip|include/ginkgo/core|omp|reference|dpcpp)/'
FORMAT_REGEX='^(common|examples|test_install)/'

echo "Retrieving PR file list"
PR_FILES=$(bot_get_all_changed_files ${PR_URL})
NUM=$(echo "${PR_FILES}" | wc -l)
echo "PR has ${NUM} changed files"

TO_FORMAT="$(echo "$PR_FILES" | grep -E $EXTENSION_REGEX || true)"

git remote add fork "$HEAD_URL"
git fetch fork "$HEAD_BRANCH"

git config user.email "ginkgo.library@gmail.com"
git config user.name "ginkgo-bot"

# save scripts from develop
pushd dev_tools/scripts
cp add_license.sh format_header.sh update_ginkgo_header.sh /tmp
popd

# checkout current PR head
LOCAL_BRANCH=format-tmp-$HEAD_BRANCH
git checkout -b $LOCAL_BRANCH fork/$HEAD_BRANCH

# restore files from develop
cp /tmp/add_license.sh dev_tools/scripts/
cp /tmp/format_header.sh dev_tools/scripts/
cp /tmp/update_ginkgo_header.sh dev_tools/scripts/

# format files
CLANG_FORMAT=clang-format-8
dev_tools/scripts/add_license.sh
dev_tools/scripts/update_ginkgo_header.sh
for f in $(echo "$TO_FORMAT" | grep -E $FORMAT_HEADER_REGEX); do dev_tools/scripts/format_header.sh "$f"; done
for f in $(echo "$TO_FORMAT" | grep -E $FORMAT_REGEX); do "$CLANG_FORMAT" -i -style=file "$f"; done

# restore formatting scripts so they don't appear in the diff
git checkout -- dev_tools/scripts/*.sh
