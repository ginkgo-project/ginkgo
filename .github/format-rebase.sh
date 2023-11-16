#!/usr/bin/env bash

cp .github/bot-pr-format-base.sh /tmp
source /tmp/bot-pr-format-base.sh

bot_delete_comments_matching "Error: Rebase failed"

DIFF_COMMAND="git diff --name-only --no-renames --diff-filter=AM HEAD~ | grep -E '$EXTENSION_REGEX'"

# do the formatting rebase
git rebase --rebase-merges --empty=drop --no-keep-empty \
    --exec "cp /tmp/add_license.sh dev_tools/scripts/ && \
            dev_tools/scripts/add_license.sh && \
            pipx run pre-commit run && \
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
