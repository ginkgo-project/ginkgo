#!/usr/bin/env bash

cp .github/bot-pr-format-base.sh /tmp
source /tmp/bot-pr-format-base.sh

echo "Retrieving PR file list"
PR_FILES=$(bot_get_all_changed_files ${PR_URL})
NUM=$(echo "${PR_FILES}" | wc -l)
echo "PR has ${NUM} changed files"

TO_FORMAT="$(echo "$PR_FILES" | grep -E $EXTENSION_REGEX || true)"

# format files
dev_tools/scripts/update_ginkgo_header.sh
for f in $(echo "$TO_FORMAT" | grep -E $FORMAT_HEADER_REGEX); do dev_tools/scripts/format_header.sh "$f"; done
pipx run pre-commit run --files $TO_FORMAT || true

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
