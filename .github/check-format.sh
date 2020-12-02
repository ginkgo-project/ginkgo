#!/bin/bash

cp .github/bot-pr-format-base.sh /tmp
source /tmp/bot-pr-format-base.sh

# check for changed files, replace newlines by \n
LIST_FILES=$(git diff --name-only | sed '$!s/$/\\n/' | tr -d '\n')

git diff > /tmp/format.patch
mv /tmp/format.patch .

bot_delete_comments_matching "Error: The following files need to be formatted"

if [[ "$LIST_FILES" != "" ]]; then
  MESSAGE="The following files need to be formatted:\n"'```'"\n$LIST_FILES\n"'```'
  MESSAGE="$MESSAGE\nYou can find a formatting patch under **Artifacts** [here]"
  MESSAGE="$MESSAGE($JOB_URL) or run "'`format!` if you have write access to Ginkgo'
  bot_error "$MESSAGE"
fi
