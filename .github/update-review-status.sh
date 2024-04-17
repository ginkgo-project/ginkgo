#!/usr/bin/env bash

set -e

if [[ $# -lt 2 ]]; then
  echo "This script needs to be called with 2 arguments:"
  echo "    $(basename $0) field-name field-value"
  exit 1
fi


FIELD_NAME="$1"
FIELD_VALUE="$2"


function check_errors {
  local json=$1

  if jq -e ".errors" "$json" > /dev/null; then
    echo "The last query resulted in the following error:"
    cat "$json"
    exit 1
  fi
}


echo "Updating field: \"$FIELD_NAME\" with value: \"$FIELD_VALUE\""

echo "Querying project ID and item ID of the PR within the project ... "
gh api graphql -F id="\"$PR_NODE_ID\"" -f query='
query($id: ID!){
  node(id: $id) {
    ... on PullRequest {
      projectItems(first: 1) {
        nodes {
          id
          project {
            id
          }
        }
      }
    }
  }
}
' > response
check_errors response
if [[ $(jq ".data.node.projectItems.nodes | length" response) -eq 0 ]]; then
  echo
  echo "This PR doesn't belong to any project, so no status will be updated."
  exit
fi
jq ".data.node.projectItems.nodes[0] | {itemId: .id, projectId: .project.id}" response > projectAndItemId
echo "done"


echo -n "Querying field ID of field named '$FIELD_NAME' ... "
gh api graphql -F projectId="$(jq ".projectId" projectAndItemId)" -f query="
query (\$projectId: ID!) {
  node(id: \$projectId) {
    ... on ProjectV2 {
      field(name: \"$FIELD_NAME\") {
        ... on ProjectV2FieldCommon {
          id
        }
      }
    }
  }
}
" > response
check_errors response
jq -s "{fieldId: .[0].data.node.field.id} + .[1] " response projectAndItemId > inputIds
echo "done"

echo -n "Querying the IDs of the values of field '$FIELD_NAME' ... "
gh api graphql -F fieldId="$(jq ".fieldId" inputIds)" -f query='
query($fieldId: ID!){
  node(id: $fieldId){
   ... on ProjectV2SingleSelectField{
     options{
       id
       name
     }
   }
  }
}' > response
check_errors response
jq '.data.node.options[] | {(.name): .id}' response | jq -s 'reduce .[] as $m (null; . + $m)' > values
echo "done"

echo "Combined inputs for mutation call:"
jq -s ".[0] + {value: .[1].\"$FIELD_VALUE\"}" inputIds values > input
jq "." input

echo -n "Updating project review status ..."
gh api graphql -f query='
mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: String!){
  updateProjectV2ItemFieldValue(input: {projectId: $projectId, itemId: $itemId, fieldId: $fieldId,
                                        value: { singleSelectOptionId: $value }}){
    projectV2Item{
      id
    }
  }
}' -F projectId="$(jq ".projectId" input)" -F itemId="$(jq ".itemId" input)" \
   -F fieldId="$(jq ".fieldId" input)" -F value="$(jq -r ".value" input)" > response
# somehow ONLY the value input needs to get the raw output from jq, no idea why
check_errors response
echo "done"
