#!/usr/bin/env bash

source .github/bot-pr-base.sh

echo "Checking whether the PR contains the 1:ST:no-changelog-entry label"
SKIP_CHECK=$(api_get "$PR_URL" | jq -r 'any( [.labels | .[] | .name ] | .[] ; . == "1:ST:no-changelog-entry")')

if [[ "$SKIP_CHECK" == "true" ]]; then
  echo "The PR contains the 1:ST:no-changelog-entry label. Skipping the wiki check."
elif [[ "$HEAD_BRANCH" =~ ^release-.* ]]; then
  echo "The PR branch name starts with release. Skipping the wiki check."
else
  curl https://raw.githubusercontent.com/wiki/ginkgo-project/ginkgo/Changelog.md > Changelog.md
  PR="\[\#${PR_NUMBER}\]\(https://github.com/ginkgo-project/ginkgo/(issues|pull)/${PR_NUMBER}\)"
  HAS_WIKI="$(cat Changelog.md | grep -E ${PR} || echo false)"
  if [[ "${HAS_WIKI}" == "false" ]]; then
    echo "This PR does not create the corresponding entry in wiki/Changelog"
    echo "Please add [#${PR_NUMBER}](https://github.com/ginkgo-project/ginkgo/pull/${PR_NUMBER}) in the wiki/Changelog page"
    echo "Alternatively, use the label 1:ST:no-changelog-entry to skip this check."
    exit 1
  else
    echo "wiki/Changelog already has this PR entry."
  fi
fi
