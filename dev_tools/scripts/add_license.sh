#!/usr/bin/env bash

THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
# ${THIS_DIR} is expected to be: ${GINKGO_ROOT_DIR}/dev_tools/scripts

# Use local paths, so there is less chance of a newline being in a path of a found file
cd "${THIS_DIR}/../.."
GINKGO_ROOT_DIR="."


LICENSE_FILE="${GINKGO_ROOT_DIR}/LICENSE"
GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"

# These two files are temporary files which will be created (and deleted).
# Therefore, the files should not already exist.
COMMENTED_LICENSE_FILE="${THIS_DIR}/commented_license.tmp"
DIFF_FILE="${THIS_DIR}/diff.patch.tmp"

# Test if required commands are present on the system:
command -v find &> /dev/null
if [ ${?} -ne 0 ]; then
    echo 'The command `find` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
command -v diff &> /dev/null
if [ ${?} -ne 0 ]; then
    echo 'The command `diff` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
command -v patch &> /dev/null
if [ ${?} -ne 0 ]; then
    echo 'The command `patch` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
command -v grep &> /dev/null
if [ ${?} -ne 0 ]; then
    echo 'The command `grep` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
command -v sed &> /dev/null
if [ ${?} -ne 0 ]; then
    echo 'The command `sed` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi
command -v cut &> /dev/null
if [ ${?} -ne 0 ]; then
    echo 'The command `cut` is required for this script to work, but not supported by your system.' 1>&2
    exit 1
fi


echo -e "/*${GINKGO_LICENSE_BEACON}\n$(cat ${LICENSE_FILE})\n${GINKGO_LICENSE_BEACON}*/\n" > "${COMMENTED_LICENSE_FILE}"

# Does not work if a found file (including the path) contains a newline
find "${GINKGO_ROOT_DIR}" \
    ! \( -name "build" -prune -o -name "third_party" -prune -o -name "external-lib-interfacing.cpp" -prune \) \
    \( -name '*.cuh' -o -name '*.hpp' -o -name '*.hpp.in' -o -name '*.cpp' -o -name '*.cu' -o -name '*.hpp.inc' \) \
    -type f -print \
    | \
    while IFS='' read -r i; do
        # `grep -F` is important here because the characters in the beacon should be matched against
        # and not interpreted as an expression.
        if ! grep -F -q -e "${GINKGO_LICENSE_BEACON}" "${i}"
        then
            cat "${COMMENTED_LICENSE_FILE}" "${i}" >"${i}.new" && mv "${i}.new" "${i}"
        else
            beginning=$(grep -F -n -e "/*${GINKGO_LICENSE_BEACON}" "${i}" | cut -d":" -f1)
            end=$(grep -F -n -e "${GINKGO_LICENSE_BEACON}*/" "${i}" | cut -d":" -f1)
            end=$((end+1))
            diff -u <(sed -n "${beginning},${end}p" "${i}") "${COMMENTED_LICENSE_FILE}" > "${DIFF_FILE}"
            if [ "$(cat "${DIFF_FILE}")" != "" ]
            then
                patch "${i}" "${DIFF_FILE}"
            fi
            rm "${DIFF_FILE}"
        fi
    done

rm "${COMMENTED_LICENSE_FILE}"
