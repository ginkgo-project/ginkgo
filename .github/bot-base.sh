#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

set -e

API_HEADER="Accept: application/vnd.github.v3+json"
AUTH_HEADER="Authorization: token $GITHUB_TOKEN"

api_get() {
  curl -X GET -s -H "${AUTH_HEADER}" -H "${API_HEADER}" "$1"
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
