#!/bin/bash

extract_varname() {
    local str="$1"
    # local GET_PARAM=" *([^ ]*) *$"
    # Need to remove the  = ....
    # note. it only remove the simple one
    local GET_PARAM=" *([^ =\*]*) *(= *.*)* *$"
    local parameter=""
    local temp=""
    IFS=',' read -ra par <<< "$str"
    for var in "${par[@]}"; do
        if [ -n "${temp}" ]; then
            temp="${temp},"
        fi
        temp="${temp}${var}"
        # only handle one pair <> currently
        if [[ "${temp}" =~ "<" ]] && [[ ! "${temp}" =~ ">" ]]; then
            continue
        fi
        # If the string contains typename, do not extract it.
        # It should automatically be decided from argument
        # Also need to ignore = ...
        if [[ "${temp}" =~ "typename" ]]; then
            :
        elif [[  "${temp}" =~ $GET_PARAM ]]; then
            if [ -n "${parameter}" ]; then
                parameter="${parameter}, "
            fi
            parameter="${parameter}${BASH_REMATCH[1]}"
        fi
        temp=""
    done
    echo "$parameter"
}

check_closed() {
    local str="$1"
    str="${str//->}"
    str_start="${str//[^(<\[]}"
    str_end="${str//[^>)\]]}"
    if [[ "${#str_start}" -eq "${#str_end}" ]]; then
        echo "true"
    else
        echo "false"
    fi
}
CONFIG_SELECTION_SUFFIX="_CONFIG"
generate_config_selection() {
    local template="$1"
    local name="$2"
    local variable="$3"
    local var_input="$4"
    template=$(echo ${template} | sed -E 's/^ *template *<(.*)> *$/\1/g')
    local temp=""
    IFS=',' read -ra par <<< "$template"

    local id_list=("bool" "int" "size_type" "typename")
    # Bash 4+ required
    declare -A rev_id_list
    local rev_id_list=(["bool"]=0 ["int"]=1 ["size_type"]=2 ["typename"]=3 ["Config"]=4)
    declare -A list_list
    list_list=(["bool"]="bool" ["int"]="int" ["size_type"]="size_type" ["typename"]="")
    local max_num=0;
    local param_regex="([^ =\*]*) *([^ =\*]*) *(= *.*)* *$"
    local template_rm_config=""
    local default_config=""
    for var in "${par[@]}"; do
        # echo "varaible - ${var}"
        if [ -n "${temp}" ]; then
            temp="${temp},"
        fi
        temp="${temp}${var}"
        is_closed=$(check_closed "$temp")
        if [[ "${is_closed}" = "false" ]]; then
            continue
        fi
        # echo "temp - ${temp}"
        # If the string contains typename, do not extract it.
        # It should automatically be decided from argument
        local config_regex="Config *([a-zA-Z0-9_]*) *= *([a-zA-Z][a-zA-Z_:,\(\) ]*)"
        if [[ "${temp}" =~ ${config_regex} ]]; then
            max_num=${rev_id_list[Config]}
            default_config="${BASH_REMATCH[2]}"
            temp=""
            continue
        fi
        if [ -n "${template_rm_config}" ]; then
                template_rm_config="${template_rm_config}, "
        fi
        template_rm_config="${template_rm_config}${temp}"
        if [[  "${temp}" =~ $param_regex ]]; then
            identification=${BASH_REMATCH[1]}
            parameter_name=${BASH_REMATCH[2]}
            # echo "${identification} ${parameter_name}"
            if [ "${max_num}" -eq "${rev_id_list[Config]}" ]; then
                if [ "${identification}" != "typename" ]; then
                    echo "static_assert(false, \"Only inferred type are allowed after Config\");"
                fi
            else
                if [ "${rev_id_list[${identification}]}" -ge "${max_num}" ]; then
                    max_num=${rev_id_list[${identification}]}
                    if [ -n "${list_list[${identification}]}" ]; then
                        parameter_name=", ${parameter_name}"
                    fi
                    list_list[${identification}]="${list_list[${identification}]}${parameter_name}"
                else
                    echo "static_assert(false, \"Need to follow bool, int, size_type, type ordering\");"
                fi
            fi
        fi
        temp=""
    done
    # for item in "${!list_list[@]}"; do
    #     echo "//$item - ${list_list[$item]}"
    # done
    config_generation=""
    if [[ "${default_config}" = *"config_set"* ]]; then
        # When the config is set in template, use it directly.
        config_generation="::gko::syn::value_list<Config, ${default_config}>();"
    else
        # When config use the constexpr variable, use the variable name. The constexpr variable will be changed to Config_list afterwards.
        config_generation="${default_config}_list"
    fi
    echo "template <${template_rm_config}>
        void ${name}${CONFIG_SELECTION_SUFFIX}
        (dim3 grid, dim3 block, size_t dynamic_shared_memory, cudaStream_t stream, std::shared_ptr<const CudaExecutor> exec, ${variable}) {
        auto config_list = ${config_generation};
        auto exec_config = exec->get_config();
        ${name}${CONFIG_SELECTION_SUFFIX}(
            config_list,
            [&exec_config] (Config config) {
                return (get_warp_size(exec_config) == get_warp_size(config)) &&
                        (get_block_size(exec_config) >= get_block_size(config));
                },
            ::gko::syn::value_list<${list_list[bool]}>(),
            ::gko::syn::value_list<${list_list[int]}>(),
            ::gko::syn::value_list<${list_list[size_type]}>(),
            ::gko::syn::type_list<${list_list[typename]}>(),
            grid, block, dynamic_shared_memory, stream, ${var_input});
    }
    "
}


GLOBAL_KEYWORD="__global__"
TEMPLATE_REGEX="^ *template <*"
FUNCTION_START="^ *(template *<|${GLOBAL_KEYWORD}|void)"
FUNCTION_NAME_END=".*\{.*"
SCOPE_START="${FUNCTION_NAME_END}"
SCOPE_END=".*\}.*"
CHECK_GLOBAL_KEYWORD=".*${GLOBAL_KEYWORD}.*"
FUNCTION_HANDLE=""
DURING_FUNCNAME="false"
ANAYSIS_FUNC=" *(template *<(.*)>)?.* (.*)\((.*)\)"
START_BLOCK_REX="^( *\/\*| *\/\/)"
END_BLOCK_REX="\*\/$| *\/\/"
IN_BLOCK=0
IN_FUNC=0
STORE_LINE=""
STORE_REGEX="__ *$"
HOST_SUFFIX="_AUTOHOSTFUNC"
EXTRACT_KERNEL="false"
GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"
DURING_LICENSE="false"
SKIP="false"
MAP_FILE="map_list"
rm "${MAP_FILE}"
while IFS='' read -r line || [ -n "$line" ]; do
    if [ "${EXTRACT_KERNEL}" = "false" ] && ([ "${line}" = "/*${GINKGO_LICENSE_BEACON}" ] ||  [ "${DURING_LICENSE}" = "true" ]); then
        DURING_LICENSE="true"
        if [ "${line}" = "${GINKGO_LICENSE_BEACON}*/" ]; then
            DURING_LICENSE="false"
            SKIP="true"
        fi
        continue
    fi
    # When do not need the license, do not need the space between license and other codes, neither.
    if [ ${SKIP} = "true" ] && [ -z "${line}" ]; then
        continue
    fi
    SKIP="false"
    if [[ "$line" =~ ${STORE_REGEX} ]]; then
        STORE_LINE="${STORE_LINE} ${line}"
    elif [[ -n "${STORE_LINE}" ]]; then
        echo "${STORE_LINE} ${line}"
        STORE_LINE=""
    else
        echo "${line}"
    fi
    # echo "Handle___ ${line}"
    if [[ "$line" =~ ${START_BLOCK_REX} ]] || [[ "${IN_BLOCK}" -gt 0 ]]; then
        if [[ "$line" =~ ${START_BLOCK_REX} ]]; then
            IN_BLOCK=$((IN_BLOCK+1))
        fi
        if [[ "$line" =~ ${END_BLOCK_REX} ]]; then
            IN_BLOCK=$((IN_BLOCK-1))
        fi
        # echo ""
        # echo "IN BLOCK ${IN_BLOCK}"
        # output to new file
        continue
    fi
    # echo "Handle ${line}"
    # handle comments
    if [[ "${line}" =~ $FUNCTION_START ]] || [[ $DURING_FUNCNAME = "true" ]]; then
        # echo "line ${line}"
        # echo "${FUNCTION_NAME_END}"
        DURING_FUNCNAME="true"
        FUNCTION_HANDLE="${FUNCTION_HANDLE} $line"
        if [[ "${line}" =~ ${FUNCTION_NAME_END} ]]; then
            # echo "end"
            DURING_FUNCNAME="false"
        fi
        if [[ "${line}" =~ ${SCOPE_START} ]]; then
            IN_FUNC=$((IN_FUNC+1))
        fi
        if [[ "${line}" =~ ${SCOPE_END} ]]; then
            IN_FUNC=$((IN_FUNC-1))
        fi
        # output to new file
        continue
    fi
    # echo "Handle ${line}"
    if [ -n "${FUNCTION_HANDLE}" ] && [[ ${DURING_FUNCNAME} = "false" ]]; then
        if [[ "${line}" =~ ${SCOPE_START} ]]; then
            IN_FUNC=$((IN_FUNC+1))
        fi
        if [[ "${line}" =~ ${SCOPE_END} ]]; then
            IN_FUNC=$((IN_FUNC-1))
        fi
        # echo "IN FUNC ${IN_FUNC}"

        # make sure the function is end
        if [[ "${IN_FUNC}" -eq 0 ]]; then
            # echo "check ${FUNCTION_HANDLE}"

            if [[ "${FUNCTION_HANDLE}" =~ $CHECK_GLOBAL_KEYWORD ]]; then
                echo ""
                # echo "${FUNCTION_HANDLE}"
                # remove additional space
                FUNCTION_HANDLE=$(echo "${FUNCTION_HANDLE}" | sed -E 's/ +/ /g;')
                # echo "->"
                # echo "${FUNCTION_HANDLE}"
                # echo "->"

                if [[ "${FUNCTION_HANDLE}" =~ $ANAYSIS_FUNC ]]; then
                    TEMPLATE="${BASH_REMATCH[1]}"
                    TEMPLATE_CONTENT="${BASH_REMATCH[2]}"
                    NAME="${BASH_REMATCH[3]}"
                    VARIABLE="${BASH_REMATCH[4]}"
                    VARIABLE=$(echo ${VARIABLE} | sed 's/__restrict__ //g')
                    VAR_INPUT=$(extract_varname "${VARIABLE}")
                    TEMPLATE_INPUT=$(extract_varname "${TEMPLATE_CONTENT}")
                    if [ -n "${TEMPLATE_INPUT}" ]; then
                        TEMPLATE_INPUT="<${TEMPLATE_INPUT}>"
                    fi
                    echo "${TEMPLATE} void ${NAME}${HOST_SUFFIX} (dim3 grid, dim3 block, size_t dynamic_shared_memory, cudaStream_t stream, ${VARIABLE}) {
                        /*KEEP*/${NAME}${TEMPLATE_INPUT}<<<grid, block, dynamic_shared_memory, stream>>>(${VAR_INPUT});
                        }"
                    if [[ "${FUNCTION_HANDLE}" = *"Config"* ]]; then
                        # echo "/** DPCPP ONLY // RM_TAG"
                        echo ""
                        echo "GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(${NAME}${HOST_SUFFIX}${CONFIG_SELECTION_SUFFIX}, ${NAME}${HOST_SUFFIX})"
                        echo ""
                        echo ""
                        config_function=$(generate_config_selection "${TEMPLATE}" "${NAME}${HOST_SUFFIX}" "${VARIABLE}" "${VAR_INPUT}")
                        echo "${config_function}"
                        # echo "**/ // RM_TAG"
                        echo ""
                        echo ""
                        # echo "${TEMPLATE} void ${NAME}${HOST_SUFFIX}${CONFIG_SELECTION_SUFFIX} // RM_TAG
                        #     (dim3 grid, dim3 block, size_t dynamic_shared_memory, cudaStream_t stream, std::shared_ptr<const CudaExecutor> exec, ${VARIABLE}) { // RM_TAG
                        #     ${NAME}${TEMPLATE_INPUT}<<<grid, block, dynamic_shared_memory, stream>>>(${VAR_INPUT}); // RM_TAG
                        #     } // RM_TAG"
                        # write the map to the file ${NAME} -> ${NAME}${HOST_SUFFIX}${CONFIG_SELECTION_SUFFIX}
                        echo "${NAME} -> ${NAME}${HOST_SUFFIX}${CONFIG_SELECTION_SUFFIX}" >> ${MAP_FILE}
                    else
                        # else, write the map to the file ${NAME} -> ${NAME}${HOST_SUFFIX}
                        echo "${NAME} -> ${NAME}${HOST_SUFFIX}" >> ${MAP_FILE}
                    fi
                fi
                # echo ""
                # check the property
                # extract template
                # maybe remove any [[ ]]
                # extract function name
                # extract function variables

                # check Config
                # extract bool, int, size_type, typename before Config
                # add one function like original one
                # add selection_config macro
                # need to keep the map
            fi
            FUNCTION_HANDLE=""
        fi
    fi


done < "$1"

# Maybe it only works in Linux
sort "${MAP_FILE}" | uniq > "${MAP_FILE}_temp"
mv "${MAP_FILE}_temp"  "${MAP_FILE}"
