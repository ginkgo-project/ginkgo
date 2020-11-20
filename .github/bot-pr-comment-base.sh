#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/bot-base.sh

echo -n "Collecting information on triggering event"
ISSUE_URL=$(jq -er ".issue.url" "$GITHUB_EVENT_PATH")
echo -n .
USER_LOGIN=$(jq -er ".comment.user.login" "$GITHUB_EVENT_PATH")
echo -n .
USER_URL=$(jq -er ".comment.user.url" "$GITHUB_EVENT_PATH")
echo -n .
PR_URL=$(jq -er ".issue.pull_request.url" "$GITHUB_EVENT_PATH")
echo .

bot_comment() {
  api_post $ISSUE_URL/comments "{\"body\":\"$1\"}" > /dev/null
}

bot_error() {
  echo "$1"
  bot_comment "Error: $1"
  exit 1
}

# collect info on the PR we are checking
echo -n "Collecting information on pull request"
PR_JSON=$(api_get $PR_URL)
echo -n .

PR_MERGED=$(echo "$PR_JSON" | jq -r .merged)
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
