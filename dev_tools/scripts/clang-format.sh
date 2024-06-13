#!/usr/bin/env bash

PRE_COMMIT_CACHE="${HOME}/.cache/pre-commit"
# get the python environment folder from pre-commit
VENV="$(sqlite3 -header -csv ${PRE_COMMIT_CACHE}/db.db "select * from repos;" | grep clang-format,v14.0.0 | sed -E 's/.*,.*,(.*)/\1/g')"
ACTIVATE="$(find ${VENV} -name activate -print -quit)"
# activate env
source "${ACTIVATE}"
# forward the arguments
clang-format "$@"
