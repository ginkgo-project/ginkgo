#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

source .github/bot-pr-base.sh

git remote add base "$BASE_URL"
git remote add fork "$HEAD_URL"

git remote -v

git fetch base $BASE_BRANCH
git fetch fork $HEAD_BRANCH

git config user.email "$USER_EMAIL"
git config user.name "$USER_NAME"

LOCAL_BRANCH=rebase-tmp-$HEAD_BRANCH
git checkout -b $LOCAL_BRANCH fork/$HEAD_BRANCH

# save scripts from develop
pushd dev_tools/scripts
cp add_license.sh format_header.sh update_ginkgo_header.sh /tmp
popd

bot_delete_comments_matching "Error: Rebase failed"

DIFF_COMMAND="git diff --name-only --no-renames --diff-filter=AM HEAD~ | grep -E '$EXTENSION_REGEX'"

# do the formatting rebase
git rebase --rebase-merges --empty=drop --no-keep-empty \
    --exec "cp /tmp/add_license.sh /tmp/format_header.sh /tmp/update_ginkgo_header.sh dev_tools/scripts/ && \
            dev_tools/scripts/add_license.sh && dev_tools/scripts/update_ginkgo_header.sh && \
            for f in \$($DIFF_COMMAND | grep -E '$FORMAT_HEADER_REGEX'); do dev_tools/scripts/format_header.sh \$f; done && \
            for f in \$($DIFF_COMMAND | grep -E '$FORMAT_REGEX'); do $CLANG_FORMAT -i \$f; done && \
            git checkout dev_tools/scripts && (git diff >> /tmp/difflog; true) && (git diff --quiet || git commit -a --amend --no-edit --allow-empty)" \
    base/$BASE_BRANCH 2>&1 || bot_error "Rebase failed, see the related [Action]($JOB_URL) for details"

# repeat rebase to delete empty commits
git rebase --rebase-merges --empty=drop --no-keep-empty --exec true \
    base/$BASE_BRANCH 2>&1 || bot_error "Rebase failed, see the related [Action]($JOB_URL) for details"

cp /tmp/difflog diff.patch

if [ -s diff.patch ]
then
    bot_comment "Formatting rebase introduced changes, see Artifacts [here]($JOB_URL) to review them"
fi

# push back
git push --force-with-lease fork $LOCAL_BRANCH:$HEAD_BRANCH 2>&1 || bot_error "Cannot push rebased branch, are edits for maintainers allowed, or were changes pushed while the rebase was running?"
