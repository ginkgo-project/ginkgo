#!/usr/bin/env bash
shopt -s globstar
shopt -s extglob

THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
GINKGO_ROOT_DIR="${THIS_DIR}/../.."
LICENSE_FILE="${GINKGO_ROOT_DIR}/LICENSE"
PATTERNS="cpp|hpp|cuh|cu|hpp.in"
EXCLUDED_DIRECTORIES="build|third_party"
GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"

echo -e "/*${GINKGO_LICENSE_BEACON}\n$(cat ${LICENSE_FILE})\n${GINKGO_LICENSE_BEACON}*/\n" > commented_license.tmp

for i in ${GINKGO_ROOT_DIR}/!(${EXCLUDED_DIRECTORIES})/**/*.@(${PATTERNS})
do
    if ! grep -q "${GINKGO_LICENSE_BEACON}" $i
    then
        cat commented_license.tmp $i >$i.new && mv $i.new $i
    else
        beginning=$(grep -n "/\*${GINKGO_LICENSE_BEACON}" $i | cut -d":" -f1)
        end=$(grep -n "${GINKGO_LICENSE_BEACON}.*/" $i | cut -d":" -f1)
        end=$((end+1))
        diff -u <(sed -n "${beginning},${end}p" $i) commented_license.tmp > diff.patch
        if [ "$(cat diff.patch)" != "" ]
        then
            patch $i diff.patch
        fi
        rm diff.patch
    fi
done

rm commented_license.tmp
