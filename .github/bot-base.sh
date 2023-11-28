#!/usr/bin/env bash

set -e

API_HEADER="Accept: application/vnd.github.v3+json"
AUTH_HEADER="Authorization: token $GITHUB_TOKEN"

api_get() {
  curl -X GET -s -H "${AUTH_HEADER}" -H "${API_HEADER}" "$1" > $RUNNER_TEMP/output.json
  cat $RUNNER_TEMP/output.json 
  if [[ "$RUNNER_DEBUG" == "1" ]]; then
    cat $RUNNER_TEMP/output.json 1>&2
  fi
}

api_post() {
  curl -X POST -s -H "${AUTH_HEADER}" -H "${API_HEADER}" "$1" -d "$2"
}

api_patch() {
  curl -X PATCH -s -H "${AUTH_HEADER}" -H "${API_HEADER}" "$1" -d "$2"
}

api_delete() {
  curl -X DELETE -s -H "${AUTH_HEADER}" -H "${API_HEADER}" "$1"
}
