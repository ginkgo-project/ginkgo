#!/usr/bin/env bash
shopt -s globstar
shopt -s extglob

PLACE_HOLDER="#PUBLIC_HEADER_PLACE_HOLDER"

THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

ROOT_DIR="${THIS_DIR}/../../"
INCLUDE_DIR="${ROOT_DIR}/include"

cd ${INCLUDE_DIR}

GINKGO_HEADER_FILE="ginkgo/ginkgo.hpp"
GINKGO_HEADER_TEMPLATE_FILE="${GINKGO_HEADER_FILE}.in"

HEADER_LIST="global_includes.hpp.tmp"

# Add every header file inside the ginkgo folder to the file ${HEADER_LIST}
for file in ginkgo/**/*.hpp; do
    if [ "${file}" == "${GINKGO_HEADER_FILE}" ]; then
        continue
    fi
    echo "${file}" >> ${HEADER_LIST}
done

# It must be a POSIX locale in order to sort according to ASCII
export LC_ALL=C
# Sorting is necessary to group them according to the folders the header are in
sort -o ${HEADER_LIST} ${HEADER_LIST}

# Generate a new, temporary ginkgo header file.
# It will get compared at the end to the existing file in order to prevent 
# the rebuilding of targets which depend on the global header
# (e.g. benchmarks and examples)
GINKGO_HEADER_TMP="${GINKGO_HEADER_FILE}.tmp"

PREVIOUS_FOLDER=""
# "IFS=''" sets the word delimiters for read.
# An empty $IFS means the given name (after `read`) will be set to the whole line.
while IFS='' read -r line; do
    if [ "${line}" != ${PLACE_HOLDER} ]; then
        echo "${line}" >> ${GINKGO_HEADER_TMP}
    else
        READING_FIRST_LINE=true
        while IFS='' read -r file; do
            CURRENT_FOLDER=$(dirname ${file})
            # add newline between different include folder
            if [ "${READING_FIRST_LINE}" != true ] && \
               [ "${CURRENT_FOLDER}" != "${PREVIOUS_FOLDER}" ]
            then
                echo "" >> ${GINKGO_HEADER_TMP}
            fi
            PREVIOUS_FOLDER=${CURRENT_FOLDER}
            echo "#include <${file}>" >> ${GINKGO_HEADER_TMP}
            READING_FIRST_LINE=false
        done < "${HEADER_LIST}"
    fi
done < ${GINKGO_HEADER_TEMPLATE_FILE}

# Use the generated file ONLY when the public header does not exist yet
# or the generated one is different to the existing one
if [ ! -f "${GINKGO_HEADER_FILE}" ] || \
   ! cmp -s "${GINKGO_HEADER_TMP}" "${GINKGO_HEADER_FILE}"
then
    mv "${GINKGO_HEADER_TMP}" "${GINKGO_HEADER_FILE}"
else
    rm ${GINKGO_HEADER_TMP}
fi

rm ${HEADER_LIST}
