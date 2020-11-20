#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/bot-pr-comment-base.sh

if [[ "$PR_MERGED" == "true" ]]; then
  bot_error "PR already merged!"
fi

git remote set-url origin "$BASE_URL"
git remote add fork "$HEAD_URL"

# make sure branches are up-to-date
git fetch origin $BASE_BRANCH
git fetch fork $HEAD_BRANCH

git config user.email "ginkgo.library@gmail.com"
git config user.name "ginkgo-bot"

# save scripts from develop
cd dev_tools/scripts
cp add_license.sh update_ginkgo_header.sh format_header.sh ../../../
cd ../../

LOCAL_BRANCH=format-tmp-$HEAD_BRANCH
git checkout -b $LOCAL_BRANCH fork/$HEAD_BRANCH

# restore files from develop
cp ../add_license.sh dev_tools/scripts/
cp ../update_ginkgo_header.sh dev_tools/scripts/
cp ../format_header.sh dev_tools/scripts/

# format files
dev_tools/scripts/update_ginkgo_header.sh
find . -type f \( -name '*.cuh' -o -name '*.hpp' -o -name '*.hpp.in' -o -name '*.cpp' -o -name '*.cu' -o -name '*.hpp.inc' \) -exec clang-format-8 -i "{}" \;
dev_tools/scripts/add_license.sh
  
# restore formatting scripts so they don't appear in the diff
git checkout -- dev_tools/scripts/*.sh

# check for changed files, replace newlines by \n
LIST_FILES=$(git diff --name-only | sed '$!s/$/\\n/' | tr -d '\n')

# commit changes if necessary
if [[ "$LIST_FILES" != "" ]]; then
  git commit -a -m "Format files

Co-authored-by: $USER_COMBINED"
  git push fork $LOCAL_BRANCH:$HEAD_BRANCH 2>&1 || bot_error "Cannot push formatted branch, are edits for maintainers allowed?"
  bot_comment "Formatted the following files:\n"'```'"\n$LIST_FILES\n"'```'
else
  bot_comment "Nothing to format"
fi
