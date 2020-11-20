#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/bot-pr-comment-base.sh

if [[ "$PR_MERGED" == "true" ]]; then
  bot_error "PR already merged!"
fi

git remote add base "$BASE_URL"
git remote add fork "$HEAD_URL"

git fetch base $BASE_BRANCH
git fetch fork $HEAD_BRANCH

git config user.email "$USER_EMAIL"
git config user.name "$USER_NAME"

LOCAL_BRANCH=rebase-tmp-$HEAD_BRANCH
git checkout -b $LOCAL_BRANCH fork/$HEAD_BRANCH

# do the rebase
git rebase base/$BASE_BRANCH 2>&1 || bot_error "Rebasing failed"

# push back
git push --force-with-lease fork $LOCAL_BRANCH:$HEAD_BRANCH 2>&1 || bot_error "Cannot push rebased branch, are edits for maintainers allowed?"

bot_comment "Rebasing succeeded"
