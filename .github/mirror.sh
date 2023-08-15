#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

BRANCH_NAME=${BRANCH_NAME##*/}

git remote add fork "git@github.com:${GITHUB_REPO}.git"
git remote add gitlab "git@gitlab.com:ginkgo-project/ginkgo-public-ci.git"

git remote -v

# Setup ssh
eval $(ssh-agent -s)
echo "${BOT_KEY}" | tr -d '\r' | ssh-add - >/dev/null
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keyscan -t rsa gitlab.com github.com >>~/.ssh/known_hosts

# Fetch from github
git fetch fork "$BRANCH_NAME"
git checkout -B "$BRANCH_NAME"
git reset --hard fork/"$BRANCH_NAME"
# Push to gitlab
git push -u --force gitlab HEAD:$BRANCH_NAME
