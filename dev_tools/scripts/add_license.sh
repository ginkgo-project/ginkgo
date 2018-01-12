#!/bin/bash
shopt -s globstar
shopt -s extglob

THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
GINKGO_ROOT_DIR="${THIS_DIR}/../.."
LICENSE_FILE="${GINKGO_ROOT_DIR}/LICENSE"
PATTERNS="cpp|hpp|cuh|cu"

echo -e "/*\n" "$(cat ${LICENSE_FILE} | sed -e 's/^/ \* /')" "\n*/\n" > commented_license.tmp

for i in ${GINKGO_ROOT_DIR}/**/*.@(${PATTERNS})
do
    if ! grep -q Copyright $i
    then
        cat commented_license.tmp $i >$i.new && mv $i.new $i
    fi
done

rm commented_license.tmp
