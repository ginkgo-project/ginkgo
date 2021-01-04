#!/bin/bash

source .github/bot-pr-base.sh

echo "Retrieving PR file list"
PR_FILES=""
PAGE="1"
while true; do
  # this api allows 100 items per page
  PR_PAGE_FILES=$(api_get "$PR_URL/files?&per_page=100&page=${PAGE}" | jq -er '.[] | .filename')
  if [ "${PR_PAGE_FILES}" = "" ]; then
    break
  fi
  echo "Retrieving PR file list - ${PAGE} pages"
  if [ ! "${PR_FILES}" = "" ]; then
    # add the same new line format
    PR_FILES="${PR_FILES}"$'\n'
  fi
  PR_FILES="${PR_FILES}${PR_PAGE_FILES}"
  PAGE=$(( PAGE + 1 ))
done
NUM=$(echo "${PR_FILES}" | wc -l)
PR_FILES_ARRAY=(${PR_FILES})
echo "PR has ${#PR_FILES_ARRAY[@]} or ${NUM} changed files"

echo "Retrieving PR label list"
OLD_LABELS=$(api_get "$ISSUE_URL" | jq -er '[.labels | .[] | .name]')


label_match() {
  if echo "$PR_FILES" | grep -qE "$2"; then
    echo "+[\"$1\"]"
  fi
}

LABELS="[]"
LABELS=$LABELS$(label_match mod:core '(^core/|^include/)')
LABELS=$LABELS$(label_match mod:reference '^reference/')
LABELS=$LABELS$(label_match mod:openmp '^omp/')
LABELS=$LABELS$(label_match mod:cuda '(^cuda/|^common/)')
LABELS=$LABELS$(label_match mod:hip '(^hip/|^common/)')
LABELS=$LABELS$(label_match mod:dpcpp '^dpcpp/')
LABELS=$LABELS$(label_match reg:benchmarking '^benchmark/')
LABELS=$LABELS$(label_match reg:example '^examples/')
LABELS=$LABELS$(label_match reg:build '(cm|CM)ake')
LABELS=$LABELS$(label_match reg:ci-cd '(^\.github/|\.yml$)')
LABELS=$LABELS$(label_match reg:documentation '^doc/')
LABELS=$LABELS$(label_match reg:testing /test/)
LABELS=$LABELS$(label_match reg:helper-scripts '^dev_tools/')
LABELS=$LABELS$(label_match type:factorization /factorization/)
LABELS=$LABELS$(label_match type:matrix-format /matrix/)
LABELS=$LABELS$(label_match type:multigrid /multigrid/)
LABELS=$LABELS$(label_match type:preconditioner /preconditioner/)
LABELS=$LABELS$(label_match type:reordering /reorder/)
LABELS=$LABELS$(label_match type:solver /solver/)
LABELS=$LABELS$(label_match type:stopping-criteria /stop/)

# if all mod: labels present: replace by mod:all
LABELS=$(echo "$LABELS" | sed 's/.*mod:.*mod:.*mod:.*mod:.*mod:.*mod:[^"]*"\]/[]+["mod:all"]/')

PATCH_BODY=$(jq -rn "{labels:($OLD_LABELS + $LABELS | unique)}")
api_patch "$ISSUE_URL" "$PATCH_BODY" > /dev/null
