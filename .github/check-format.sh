#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/bot-base.sh

echo -n "Collecting information on triggering PR"
PR_URL=$(jq -r .pull_request.url "$GITHUB_EVENT_PATH")
if [[ "$PR_URL" == "null" ]]; then
  echo -n ........
  PR_URL=$(jq -er .issue.pull_request.url "$GITHUB_EVENT_PATH")
  echo -n .
fi
echo -n .
PR_JSON=$(api_get $PR_URL)
echo -n .
ISSUE_URL=$(echo "$PR_JSON" | jq -er ".issue_url")
echo -n .
HEAD_REPO=$(echo "$PR_JSON" | jq -er .head.repo.full_name)
echo -n .
HEAD_BRANCH=$(echo "$PR_JSON" | jq -er .head.ref)
echo .
HEAD_URL="https://${GITHUB_ACTOR}:${GITHUB_TOKEN}@github.com/$HEAD_REPO"

bot_comment() {
  api_post "$ISSUE_URL/comments" "{\"body\":\"$1\"}" > /dev/null
}

bot_error() {
  echo "$1"
  bot_comment "Error: $1"
  exit 1
}

# save scripts from develop
cd dev_tools/scripts
cp add_license.sh update_ginkgo_header.sh format_header.sh ../../../
cd ../../

git remote add fork "${HEAD_URL}"
git fetch fork "$HEAD_BRANCH"
git checkout -b format-tmp-$HEAD_BRANCH "fork/$HEAD_BRANCH"

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

# replace newlines by \n
LIST_FILES=$(git diff --name-only | sed '$!s/$/\\n/' | tr -d '\n')

if [[ "$LIST_FILES" != "" ]]; then
  bot_error "The following files need to be formatted:\n"'```'"\n$LIST_FILES\n"'```'
  git diff
fi
