#!/bin/bash

cp .github/bot-pr-format-base.sh /tmp
source /tmp/bot-pr-format-base.sh

# check for changed files, replace newlines by \n
LIST_FILES=$(git diff --name-only | sed '$!s/$/\\n/' | tr -d '\n')

# commit changes if necessary
if [[ "$LIST_FILES" != "" ]]; then
  git commit -a -m "Format files

Co-authored-by: $USER_COMBINED"
  git push fork $LOCAL_BRANCH:$HEAD_BRANCH 2>&1 || bot_error "Cannot push formatted branch, are edits for maintainers allowed?"
fi
