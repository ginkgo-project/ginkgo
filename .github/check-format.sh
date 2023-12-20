#!/usr/bin/env bash

cp .github/bot-pr-format-base.sh /tmp
source /tmp/bot-pr-format-base.sh

echo "Run Pre-Commit checks"

pipx run pre-commit run --show-diff-on-failure --color=always --from-ref "origin/$BASE_BRANCH" --to-ref HEAD || true

echo -n "Collecting information on changed files"

git restore --staged .pre-commit-config.yaml
git checkout -- dev_tools/scripts/*.sh .pre-commit-config.yaml .clang-format

# check for changed files, replace newlines by \n
LIST_FILES=$(git diff --name-only | sed '$!s/$/\\n/' | tr -d '\n')
echo -n .

git diff > /tmp/format.patch
mv /tmp/format.patch .
echo -n .

bot_delete_comments_matching "Error: The following files need to be formatted"
echo -n .

if [[ "$LIST_FILES" != "" ]]; then
  MESSAGE="The following files need to be formatted:\n"'```'"\n$LIST_FILES\n"'```'
  MESSAGE="$MESSAGE\nYou can find a formatting patch under **Artifacts** [here]"
  MESSAGE="$MESSAGE($JOB_URL) or run "'`format!` if you have write access to Ginkgo'
  bot_error "$MESSAGE"
fi
echo .
