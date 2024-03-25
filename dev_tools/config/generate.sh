#!/usr/bin/env bash

CLANG_FORMAT=${CLANG_FORMAT:="clang-format"}

# copy the input file to current folder with clang-format allowing long line to avoid the concate the line issue
input_file="$1"
# TODO: do more safe here
cp $input_file dev_tools/config/temp.hpp
pushd dev_tools/config
"${CLANG_FORMAT}" -i -style=file temp.hpp
popd

# grab the factory parameter out
param=$(grep "GKO_FACTORY" dev_tools/config/temp.hpp)
# <param_type> GKO_FACTORY(<param_name>, ...
regex="^ *(.*) GKO_FACTORY[^\(]*\(([^,]*),"
pointer_regex="shared_ptr<(.*)>"
echo "${param}"
echo "===="
while IFS='' read -r line || [[ -n $line ]]; do
    if [[ "${line}" =~ ${regex} ]]; then
        param_type="${BASH_REMATCH[1]}"
        param_name="${BASH_REMATCH[2]}"
        if [[ "${param_type}" =~  ${pointer_regex} ]]; then
            pointer_type="${BASH_REMATCH[1]}"
            if [[ "${param_type}" =~ vector ]]; then
                # pointer_vector
                echo "SET_POINTER_VECTOR(factory, ${pointer_type}, ${param_name}, config, context, exec, td_for_child);"
            else
                # pointer
                echo "SET_POINTER(factory, ${pointer_type}, ${param_name}, config, context, exec, td_for_child);"
            fi
        else
            # value
            echo "SET_VALUE(factory, ${param_type}, ${param_name}, config);"
        fi
        # todo: value_array
            
    else
        echo "// script does not handle ${line}"
    fi
done <<< "${param}"


rm dev_tools/config/temp.hpp