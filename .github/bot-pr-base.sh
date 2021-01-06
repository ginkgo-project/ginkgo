#!/bin/bash

source .github/bot-base.sh

echo -n "Collecting information on triggering PR"
PR_URL=$(jq -r .pull_request.url "$GITHUB_EVENT_PATH")
if [[ "$PR_URL" == "null" ]]; then
  # if this was triggered by an issue comment: get PR and commenter
  echo -n .............
  PR_URL=$(jq -er .issue.pull_request.url "$GITHUB_EVENT_PATH")
  echo -n .
  USER_LOGIN=$(jq -er ".comment.user.login" "$GITHUB_EVENT_PATH")
  echo -n .
  USER_URL=$(jq -er ".comment.user.url" "$GITHUB_EVENT_PATH")
  echo -n .
else
  # else it was triggered by a PR sync: get PR creator
  USER_LOGIN=$(jq -er ".pull_request.user.login" "$GITHUB_EVENT_PATH")
  echo -n .
  USER_URL=$(jq -er ".pull_request.user.url" "$GITHUB_EVENT_PATH")
  echo -n .
fi
echo -n .
PR_JSON=$(api_get $PR_URL)
echo -n .
PR_MERGED=$(echo "$PR_JSON" | jq -r .merged)
echo -n .
ISSUE_URL=$(echo "$PR_JSON" | jq -er ".issue_url")
echo -n .
BASE_REPO=$(echo "$PR_JSON" | jq -er .base.repo.full_name)
echo -n .
BASE_BRANCH=$(echo "$PR_JSON" | jq -er .base.ref)
echo -n .
HEAD_REPO=$(echo "$PR_JSON" | jq -er .head.repo.full_name)
echo -n .
HEAD_BRANCH=$(echo "$PR_JSON" | jq -er .head.ref)
echo .

BASE_URL="https://${GITHUB_ACTOR}:${GITHUB_TOKEN}@github.com/$BASE_REPO"
HEAD_URL="https://${GITHUB_ACTOR}:${GITHUB_TOKEN}@github.com/$HEAD_REPO"

JOB_URL="https://github.com/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"

bot_delete_comments_matching() {
  local search_matching="$1"
  COMMENTS=$(api_get "$ISSUE_URL/comments" | jq -r '.[] | select((.user.login == "ginkgo-bot") and (.body | startswith('"\"$search_matching\""'))) | .url')
  for URL in $COMMENTS; do
    api_delete "$URL" > /dev/null
  done
}

bot_comment() {
  api_post "$ISSUE_URL/comments" "{\"body\":\"$1\"}" > /dev/null
}

bot_error() {
  echo "$1"
  bot_comment "Error: $1"
  exit 1
}

bot_get_all_changed_files() {
  local pr_url="$1"
  local pr_files=""
  local page="1"
  while true; do
    # this api allows 100 items per page
    # github action uses `bash -e`. The last empty page will leads jq error, use `|| :` to ignore the error.
    local pr_page_files=$(api_get "$pr_url/files?&per_page=100&page=${page}" | jq -er '.[] | .filename' || :)
    if [ "${pr_page_files}" = "" ]; then
      break
    fi
    if [ ! "${pr_files}" = "" ]; then
      # add the same new line format as jq output
      pr_files="${pr_files}"$'\n'
    fi
    pr_files="${pr_files}${pr_page_files}"
    page=$(( page + 1 ))
  done
  echo "${pr_files}"
}

# collect info on the user that invoked the bot
echo -n "Collecting information on triggering user"
USER_JSON=$(api_get $USER_URL)
echo .

USER_NAME=$(echo "$USER_JSON" | jq -r ".name")
if [[ "$USER_NAME" == "null" ]]; then
	USER_NAME=$USER_LOGIN
fi
USER_EMAIL=$(echo "$USER_JSON" | jq -r ".email")
if [[ "$USER_EMAIL" == "null" ]]; then
	USER_EMAIL="$USER_LOGIN@users.noreply.github.com"
fi
USER_COMBINED="$USER_NAME <$USER_EMAIL>"

if [[ "$PR_MERGED" == "true" ]]; then
  bot_error "PR already merged!"
fi
