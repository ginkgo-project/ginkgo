#!/usr/bin/env bash

cp .github/bot-pr-format-base.sh /tmp
source /tmp/bot-pr-format-base.sh

# format files
echo "Formatting files"

pipx run pre-commit run --from-ref "origin/$BASE_BRANCH" --to-ref HEAD || true

# restore formatting scripts so they don't appear in the diff
git restore --staged .pre-commit-config.yaml
git checkout -- dev_tools/scripts/*.sh .pre-commit-config.yaml .clang-format

# check for changed files, replace newlines by \n
CHANGES=$(git diff --name-only | sed '$!s/$/\\n/' | tr -d '\n')

echo "$CHANGES"

# commit changes if necessary
if [[ "$CHANGES" != "" ]]; then
  git commit -a -m "Format files

Co-authored-by: $USER_COMBINED"
  git push fork "$LOCAL_BRANCH:$HEAD_BRANCH" 2>&1 || bot_error "Cannot push formatted branch, are edits for maintainers allowed?"
fi
