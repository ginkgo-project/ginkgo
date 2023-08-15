#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

pushd old
source .github/bot-pr-base.sh
popd

bot_delete_comments_matching "Note: This PR changes the Ginkgo ABI"

abidiff build-old/lib/libginkgod.so build-new/lib/libginkgod.so &> abi.diff || true # (bot_comment "Note: This PR changes the Ginkgo ABI:\n\`\`\`\n$(head -n2 abi.diff | tr '\n' ';' | sed 's/;/\\n/g')\`\`\`\nFor details check the full ABI diff under **Artifacts** [here]($JOB_URL)")
