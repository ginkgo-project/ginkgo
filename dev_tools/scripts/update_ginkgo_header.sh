#!/usr/bin/env bash

# Note: This script is supposed to support the developer and not to hinder
#       the development process by setting more restrictions.
#       This is the reason why every exit code is 0, otherwise, the whole
#       `make` procedure would fail.

PLACE_HOLDER="#PUBLIC_HEADER_PLACE_HOLDER"

WARNING_PREFIX="[WARNING] ginkgo.hpp update script failed because:"

RM_PARAMETER="--interactive=never"


THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
INCLUDE_DIR="${THIS_DIR}/../../include"

# Use local paths, so there is less chance of a newline being in a path of a found file
cd "${INCLUDE_DIR}"
TOP_HEADER_FOLDER="."

GINKGO_HEADER_FILE="ginkgo/ginkgo.hpp"
GINKGO_HEADER_TEMPLATE_FILE="${GINKGO_HEADER_FILE}.in"

HEADER_LIST="global_includes.hpp.tmp"

# Test if required commands are present on the system:
command -v find &> /dev/null
if [ ${?} -ne 0 ]; then
    echo "${WARNING_PREFIX} "'The command `find` is required for this script to work, but not supported by your system.' 1>&2
    exit 0
fi
command -v sort &> /dev/null
if [ ${?} -ne 0 ]; then
    echo "${WARNING_PREFIX} "'The command `sort` is required for this script to work, but not supported by your system.' 1>&2
    exit 0
fi
command -v cmp &> /dev/null
if [ ${?} -ne 0 ]; then
    echo "${WARNING_PREFIX} "'The command `cmp` is required for this script to work, but not supported by your system.' 1>&2
    exit 0
fi


# Put all header files as a list (separated by newlines) in the file ${HEADER_LIST}
# Requires detected files (including the path) to not contain newlines
find "${TOP_HEADER_FOLDER}" -name '*.hpp' -type f -print > "${HEADER_LIST}"

if [ ${?} -ne 0 ]; then
    echo "${WARNING_PREFIX} "'The `find` command returned with an error!' 1>&2
    rm "${RM_PARAMETER}" "${HEADER_LIST}"
    exit 0
fi

# It must be a POSIX locale in order to sort according to ASCII
export LC_ALL=C
# Sorting is necessary to group them according to the folders the header are in
sort -o "${HEADER_LIST}" "${HEADER_LIST}"

if [ ${?} -ne 0 ]; then
    echo "${WARNING_PREFIX} "'The `sort` command returned with an error!' 1>&2
    rm "${RM_PARAMETER}" "${HEADER_LIST}"
    exit 0
fi

if [ ! -r "${GINKGO_HEADER_TEMPLATE_FILE}" ]; then
    echo "${WARNING_PREFIX} The file '${GINKGO_HEADER_TEMPLATE_FILE}' can not be read!" 1>&2
    rm "${RM_PARAMETER}" "${HEADER_LIST}"
    exit 0
fi

# Detect the end of line type (CRLF/LF) by ${GINKGO_HEADER_TEMPLATE_FILE}
END="";
if [[ "$(file "${GINKGO_HEADER_TEMPLATE_FILE}")" == *"CRLF"* ]]; then
    END="\r"
fi

# Generate a new, temporary ginkgo header file.
# It will get compared at the end to the existing file in order to prevent 
# the rebuilding of targets which depend on the global header
# (e.g. benchmarks and examples)
GINKGO_HEADER_TMP="${GINKGO_HEADER_FILE}.tmp"

# See if we have write permissions to ${GINKGO_HEADER_TMP}
echo "Test for write permissions" > "${GINKGO_HEADER_TMP}"
if [ ${?} -ne 0 ]; then
    echo "${WARNING_PREFIX} No write permissions for temporary file '${GINKGO_HEADER_TMP}'!" 1>&2
    rm "${RM_PARAMETER}" "${HEADER_LIST}"
    exit 0
fi
# Remove file again, so the test does not corrupt the result
rm "${RM_PARAMETER}" "${GINKGO_HEADER_TMP}"



PREVIOUS_FOLDER=""
# "IFS=''" sets the word delimiters for read.
# An empty ${IFS} means the given name (after `read`) will be set to the whole line,
# and in this case it means it will not ignore leading and trailing whitespaces.
while IFS='' read -r line; do
    # Use echo to remove '\r' in Windows.
    line="$(echo "$line")"
    if [ "$line" != "${PLACE_HOLDER}" ]; then
        # Split context and newline interpretor to avoid unexpected interpretation.
        echo -n "${line}" >> "${GINKGO_HEADER_TMP}"
        echo -e "${END}" >> "${GINKGO_HEADER_TMP}"
    else
        READING_FIRST_LINE=true
        while IFS='' read -r prefixed_file; do
            # Remove the include directory from the file name
            file="${prefixed_file#${TOP_HEADER_FOLDER}/}"
            
            # Do not include yourself
            if [ "${file}" == "${GINKGO_HEADER_FILE}" ]; then
                continue
            fi
            
            CURRENT_FOLDER="$(dirname ${file})"
            # add newline between different include folder
            if [ "${READING_FIRST_LINE}" != true ] && \
               [ "${CURRENT_FOLDER}" != "${PREVIOUS_FOLDER}" ]
            then
                echo -e "${END}" >> "${GINKGO_HEADER_TMP}"
            fi
            PREVIOUS_FOLDER="${CURRENT_FOLDER}"
            echo -n "#include <${file}>" >> "${GINKGO_HEADER_TMP}"
            echo -e "${END}" >> "${GINKGO_HEADER_TMP}"
            READING_FIRST_LINE=false
        done < "${HEADER_LIST}"
    fi
done < "${GINKGO_HEADER_TEMPLATE_FILE}"

rm "${RM_PARAMETER}" "${HEADER_LIST}"

# Use the generated file ONLY when the public header does not exist yet
# or the generated one is different to the existing one
if [ ! -f "${GINKGO_HEADER_FILE}" ] || \
   ! cmp -s "${GINKGO_HEADER_TMP}" "${GINKGO_HEADER_FILE}"
then
    mv "${GINKGO_HEADER_TMP}" "${GINKGO_HEADER_FILE}"
    if [ ${?} -ne 0 ]; then
        echo "${WARNING_PREFIX} No permission to replace the header '${GINKGO_HEADER_FILE}'!" 1>&2
        rm "${RM_PARAMETER}" "${GINKGO_HEADER_TMP}"
        exit 0
    fi
else
    rm "${RM_PARAMETER}" "${GINKGO_HEADER_TMP}"
fi
