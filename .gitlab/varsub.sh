#!/usr/bin/env bash

# grep all %VARIABLE%
input_file=$1
variables=( $(grep -E -oh "%[a-zA-Z0-9_]*%" "${input_file}" | sed "s/%//g" | sort -u) )
sed_line=""
for var in "${variables[@]}"; do
    if [ -z ${!var} ]; then
        >&2 echo "${var} variable is not set"
        exit 1
    fi
    sed_line="${sed_line};s/%${var}%/${!var}/g"
done
sed_line="${sed_line#;}"
if [ "${sed_line}" == "" ]; then
    cat "${input_file}"
else
    sed ${sed_line#;} "${input_file}"
fi
# add a newline to avoid the concatenate issue
echo ""
