#!/usr/bin/env bash

cp .github/bot-pr-base.sh /tmp
source /tmp/bot-pr-base.sh

echo "Set-up working tree"

git remote add fork "$HEAD_URL"
git fetch fork "$HEAD_BRANCH"
git fetch origin "$BASE_BRANCH"

# checkout current PR head
LOCAL_BRANCH=format-tmp-$HEAD_BRANCH
git checkout -b $LOCAL_BRANCH fork/$HEAD_BRANCH

git config user.email "ginkgo.library@gmail.com"
git config user.name "ginkgo-bot"

# save scripts from develop
pushd dev_tools/scripts || exit 1
cp add_license.sh format_header.sh update_ginkgo_header.sh /tmp
popd || exit 1

# checkout current PR head
LOCAL_BRANCH=format-tmp-$HEAD_BRANCH
git checkout -b $LOCAL_BRANCH fork/$HEAD_BRANCH

# restore files from develop
cp /tmp/add_license.sh dev_tools/scripts/
cp /tmp/format_header.sh dev_tools/scripts/
cp /tmp/update_ginkgo_header.sh dev_tools/scripts/
