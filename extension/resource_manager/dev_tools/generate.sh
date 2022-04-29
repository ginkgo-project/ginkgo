#!/usr/bin/env bash

CLANG_FORMAT=${CLANG_FORMAT:="clang-format"}

# Checks if $1 is self contained code, that is it does not have an open and
# unclosed code portion (<>()[]), e.g. `my_struct->my_func(xxx,` should fail.
check_closed() {
    local str="$1"
    # remove -> to avoid the confusion
    str="${str//->}"
    # Replace everything except begin or end characters, resp. (<[ and )>]
    str_start="${str//[^(<\[]}"
    str_end="${str//[^>)\]]}"
    # Check that there are as many begin as end characters
    if [[ "${#str_start}" -eq "${#str_end}" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

convert_snake_to_camel() {
    local snake_case="$1"
    local regex="s/(^|_)([a-zA-Z])/\U\2\E/g"
    local output=$(echo "$snake_case" | sed -E "${regex}")
    echo "${output}"
}

convert_camel_to_snake() {
    local camel_case="$1"
    local regex="s/([^A-Z]+)([A-Z])/\1_\2/g"
    local output=$(echo "$camel_case" | sed -E "${regex}")
    echo "${output,,}"
}

extract_namespace_class() {
    local input_path=$1
    declare -n namespace_ref=$2
    declare -n class_ref=$3
    local regex=".*/ginkgo/core/(.*)/([^/]*)\.hpp"
    if [[ $input_path =~ ${regex} ]]; then
        namespace_ref="gko::${BASH_REMATCH[1]//\//::}"
        class_ref="${BASH_REMATCH[2]}"
    fi
}

contain_factory() {
    local namespace_str="$1"
    if [[ "${namespace_str}" =~ matrix|executor|log ]]; then
        echo "false"
    else
        echo "true"
    fi
}

prepare_rmfactory_param() {
    local local_type="$1"
    local local_name="$2"
    local vector_regex=" *(std::)?vector<(.*)>$"
    local shared_regex=" *(std::)?shared_ptr<(.*)>$"
    local function_regex=" *(std::)?function<(.*)>"
    local array_regex=" *(gko::)?Array<(.*)>"
    local strategy_regex="typename ([^ ]*)::strategy_type"
    local is_vector="false"
    local is_pointer="false"
    if [[ "$local_type" =~ $function_regex ]]; then
        echo "SET_FUNCTION($local_type, $local_name); /* TODO: please create a map ${local_name}_map to handle the function ${BASH_REMATCH[2]} */"
        return
    elif [[ "$local_type" =~ $array_regex ]]; then
        echo "SET_ARRAY(${BASH_REMATCH[2]}, ${local_name});"
        return
    elif [[ "$local_type" =~ ${strategy_regex} ]]; then
        echo "SET_CSR_STRATEGY(${BASH_REMATCH[1]}, ${local_name});"
        return
    fi
    if [[ "$local_type" =~ $vector_regex ]]; then
        is_vector="true"
        local_type="${BASH_REMATCH[2]}"
    fi
    if [[ "$local_type" =~ $shared_regex ]]; then
        is_pointer="true"
        local_type="${BASH_REMATCH[2]}"
    fi
    if [[ "$is_pointer" == "false" && "$is_vector" == "false" ]]; then
        echo "SET_VALUE($local_type, $local_name);"
    elif [[ "$is_pointer" == "false" && "$is_vector" == "false" ]]; then
        echo "assert(false); SET_VALUE_VECTOR($local_type, $local_name); /* not yet considered */"
    elif [[ "$is_pointer" == "true" && "$is_vector" == "false" ]]; then
        echo "SET_POINTER($local_type, $local_name);"
    else
        echo "SET_POINTER_VECTOR($local_type, $local_name);"
    fi
}

extract_template() {
    local local_input="$1"
    declare -n type_list_ref=$2
    declare -n type_name_ref=$3
    declare -n type_default_ref=$4
    local type_regex=" *([^=]*) ([^ =]*)( = ([^=]*))?"
    IFS=',' read -ra par <<< "$local_input"
    local temp=""
    for var in "${par[@]}"; do
        if [ -n "${temp}" ]; then
            temp="${temp},"
        fi
        temp="${temp}${var}"
        if [[ $(check_closed "$temp") == "false" ]]; then
            continue
        fi
        if [[ "$temp" =~ $type_regex ]]; then
        # always add ; to ensure the empty list has the same number of elements
            type_list_ref="$type_list_ref${BASH_REMATCH[1]};"
            type_name_ref="$type_name_ref${BASH_REMATCH[2]};"
            type_default_ref="$type_default_ref${BASH_REMATCH[4]};"
        else
            echo "error in anlysis ${temp}"
        fi
        temp=""
    done
}

get_base() {
    local input="$1"
    if [[ "${input}" =~ "Executor" ]]; then
        echo "Executor"
    elif [[ "${input}" =~ "stop" ]]; then
        echo "Criterion"
    elif [[ "${input}" =~ "log" ]]; then
        echo "Logger"
    else
        echo "LinOp"
    fi
}

pack() {
    local input="$1"
    local need_pack="$2"
    if [ "${need_pack}" -gt "1" ]; then
        echo "PACK($input)"
    else
        echo "$input"
    fi
}

check_exist() {
    local input="$1"
    local source_file="extension/resource_manager/include/resource_manager/base/type_default.hpp"
    local result=$(grep -q "${input}," "$source_file")
    if grep -q "${input}," "$source_file"; then
        echo "true"
    else
        echo "false"
    fi
}

generate_create_from_config() {
    local output_type="$1"
    local enum_type="$2"
    local enum_item="$3"
    local create_type_name_content="$4"
    local select_func="$5"
    local select_output_type="$6"
    local select_list="$7"
    echo "template <>
std::shared_ptr<${output_type}> create_from_config<
    ${enum_type}, ${enum_type}::${enum_item}, ${output_type}>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    // get the template from base
    std::string base_string;
    if (item.HasMember(\"base\")) {
        base_string = get_base_template(item[\"base\"].GetString());
    }
    // get the individual type
    auto type_string = create_type_name( // trick for clang-format
            ${create_type_name_content}
        );
    // combine them together, base_string has higher priority than type_string
    auto combined = combine_template(base_string, remove_space(type_string));
    auto ptr = ${select_func}<${select_output_type}>(
        ${select_list}, [=](std::string key) { return key == combined; }, item,
        exec, linop, manager);
    return std::move(ptr);
}"
}


generate_template_list_content() {
    declare -n template_type_array_ref="$1"
    declare -n template_type_name_array_ref="$2"
    declare -n template_type_default_array_ref="$3"
    local n=${#template_type_name_array_ref[@]}
    for (( idx = 0; idx < ${num}; idx++ )); do
        local sep=", "
        if [ "$idx" -eq $(( num - 1 )) ]; then
            sep=""
        fi
        # check whether it is exist
        local local_type="${template_type_array_ref[idx]}"
        local local_type_name="${template_type_name_array_ref[idx]}"
        local local_type_default="${template_type_default_array_ref[idx]}"
        local local_todo="Can not find ${local_type_name} in with TT_LIST_G_PARTIAL, please condider adding it into type_default.hpp if it reused for many times."
        if [[ "$(check_exist "TT_LIST_G_PARTIAL(${local_type_name}")" == "false" ]]; then
            if [[ "${local_type}" =~ typename|class ]]; then
                echo "tt_list<${local_type_default}>${sep} // TODO: ${local_todo}"
            else
                echo "tt_list<${local_type_default}>${sep} // TODO: The type is ${local_type}, which should be wrapped in integral_constant. ${local_todo}"
            fi
        else
            echo "tt_list_g_t<handle_type::${local_type_name}>${sep}"
        fi
    done
}


generate_type_value_set() {
    declare -n template_type_array_ref="$1"
    declare -n template_type_name_array_ref="$2"
    declare -n template_type_default_array_ref="$3"
    local n=${#template_type_name_array_ref[@]}
    for (( idx = 0; idx < ${num}; idx++ )); do
        local sep=", "
        if [ "$idx" -eq $(( num - 1 )) ]; then
            sep=""
        fi
        # check whether it is exist
        # TODO: it shuold depend on the template list or check exist
        local local_type_name="${template_type_name_array_ref[idx]}"
        local local_type_default="${template_type_default_array_ref[idx]}"
        if [[ "$(check_exist "GET_DEFAULT_STRING_PARTIAL(${local_type_name}")" == "false" ]]; then
            echo "/* TODO: can not find ${local_type_name} with GET_DEFAULT_STRING_PARTIAL, please condider adding it into type_default.hpp if it reused for many times. */"
            if [[ -n "${local_type_default}" && "${template_type_array_ref[idx]}" =~ typename|class ]]; then
                # When it is type and contains the default type, use get_string to resolve alias
                echo "get_value_with_default(item, \"${local_type_name}\", get_string<${local_type_default}>())${sep}"
            else
                # no type or no default
                echo "get_value_with_default(item, \"${local_type_name}\", \"${local_type_default}\"s)${sep}"
            fi
            # If it does not contain default value, mark it required
            # It's disabled now because we can set it from the base
            # echo "get_required_value<std::string>(item, \"${local_type_name}\")"${sep};
        else
            echo "get_value_with_default(item, \"${local_type_name}\", get_default_string<handle_type::${local_type_name}>())${sep}"
        fi
    done
}

generate_actual_type_hint() {
    local local_template_type="$1"
    local local_enum_type="$2"
    local local_enum_item="$3"
    local local_class="$4"
    declare -n template_type_array_ref="$5"
    declare -n template_type_name_array_ref="$6"
    local n=${#template_type_array_ref[@]}
    local actual_template=""
    local all_type_template=""
    for (( idx = 0; idx < ${n}; idx++ )); do
        if [ -n "${actual_template}" ]; then
            actual_template="${actual_template}, "
            all_type_template="${all_type_template}, "
        fi
        actual_template="${actual_template}${template_type_name_array_ref[idx]}"
        if [[ ! "${template_type_array_ref[idx]}" =~ typename && ! "${template_type_array_ref[idx]}" =~ class ]]; then
            all_type_template="${all_type_template}std::integral_constant<${template_type_array_ref[idx]}, ${template_type_name_array_ref[idx]}>"
        else
            all_type_template="${all_type_template}${template_type_name_array_ref[idx]}"
        fi
    done
    echo "// TODO: the class contain non type template, please create corresponding actual_type like following into type_resolving.hpp"
    echo "// and the corresponding binding of integral_constant except the first one into types.hpp with its string name in type_string.hpp"
    echo "/*
template <${local_template_type}>
struct actual_type<type_list<
    std::integral_constant<${local_enum_type}, ${local_enum_type}::${local_enum_item}>,
    ${all_type_template}>> {
    using type = ${local_class}<${actual_template}>;
};
*/"
}

generate_factory_generic() {
    local local_template_type="$1"
    local local_class_type="$2"
    local local_num="$3"
    local local_factory_set="$4"
    echo "template <${local_template_type}>
struct Generic<typename ${local_class_type}::Factory, ${local_class_type}> {
    using type = std::shared_ptr<typename ${local_class_type}::Factory>;
    static type build(rapidjson::Value &item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager *manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY($(pack "${local_class_type}" ${local_num}), manager, item, exec, linop);
            ${local_factory_set}
            SET_EXECUTOR;
        }();
        add_logger(ptr, item, exec, linop, manager);
        return std::move(ptr);
    }
};"
}

generate_linop_generic() {
    local local_template_type="$1"
    local local_class_type="$2"
    local local_readable="$3"

    echo "template <${local_template_type}>
        struct Generic<${local_class_type}> {
        using type = std::shared_ptr<${local_class_type}>;
        static type build(rapidjson::Value &item,
                        std::shared_ptr<const Executor> exec,
                        std::shared_ptr<const LinOp> linop,
                        ResourceManager *manager)
        {
            auto exec_ptr =
            get_pointer_check<Executor>(item, \"exec\", exec, linop, manager);
            auto size = get_value_with_default(item, \"dim\", gko::dim<2>{});
            // TODO: consider other thing from constructor
            auto ptr = share(${local_class_type}::create(exec_ptr, size));
        "
    if [[ "${local_readable}" == "true" ]]; then
        echo "if (item.HasMember(\"read\")) {
            std::ifstream mtx_fd(item[\"read\"].GetString());
            auto data = gko::read_raw<typename ${local_class_type}::value_type, typename ${local_class_type}::index_type>(mtx_fd);
            ptr->read(data);
            }
        "
    fi
    echo "  add_logger(ptr, item, exec, linop, manager);
            return std::move(ptr);
        }
    };"
}

generate_logger_generic() {
    local local_template_type="$1"
    local local_class_type="$2"

    echo "template <${local_template_type}>
        struct Generic<${local_class_type}> {
        using type = std::shared_ptr<${local_class_type}>;
        static type build(rapidjson::Value &item,
                        std::shared_ptr<const Executor> exec,
                        std::shared_ptr<const LinOp> linop,
                        ResourceManager *manager)
        {
            auto exec_ptr =
            get_pointer_check<Executor>(item, \"exec\", exec, linop, manager);
            auto mask_value = get_mask_value_with_default(item, \"enabled_events\", gko::log::Logger::all_events_mask);
            // TODO: consider other thing from constructor
            auto ptr = ${local_class_type}::create(exec_ptr, mask_value);
            return std::move(ptr);
        }
    };"
}


# copy the input file to current folder with clang-format allowing long line to avoid the concate the line issue
input_file="$1"
# TODO: do more safe here
cp $input_file temp.hpp
rm_file=$(echo "$input_file" | sed -E "s~.*/ginkgo/core/(.*)~extension/resource_manager/include/resource_manager/\1~g")
mv .clang-format .clang-format.tmp
cp extension/resource_manager/dev_tools/.clang-format .
"${CLANG_FORMAT}" -i -style=file temp.hpp

namespace=""
class_snake=""
extract_namespace_class "$input_file" namespace class_snake
class=$(convert_snake_to_camel "${class_snake}")
base=$(get_base "${namespace}::${class}")
rm_class=${RM_CLASS:="${class}"}
rm_class_factory=${RM_CLASS_FACTORY:="${rm_class}Factory"}
base_namespace="gko::"
if [[ "${base}" == "Criterion" ]]; then
    base_namespace="gko::stop::"
elif [[ "${base}" == "Logger" ]]; then
    base_namespace="gko::log::"
fi
echo "Handle ${namespace}::${class}"
echo "Base ${base}"
echo "Resource Manager Class Enum Name (RM_CLASS): ${rm_class}"
echo "Resource Manager Factory Class Enum Name (RM_CLASS_FACTORY): ${rm_class_factory}"
# handle template
template_regex="template <(.*)> class ${class} *:.*(public|\{)"
no_template_regex="class ${class} *:.*(public|\{)"

# handle factory if it contains factory
handle_factory=$(contain_factory "${namespace}")
factory_param_regex="^ *(.*) GKO_FACTORY_PARAMETER_(VECTOR|SCALAR)\(([^,]*),.*"
in_factory="false"
readable="false"
factory_set=""
template_type_str=""
template_type_name_str=""
template_type_default_str=""
template_type_array=""
template_type_name_array=""
template_type_default_array=""
extracted_template="false"
while IFS='' read -r line; do
    if [[ "$line" =~ ReadableFromMatrixData ]]; then
        readable="true"
    fi
    if [[ "$line" =~ ${template_regex} ]]; then
        template_content="${BASH_REMATCH[1]}"
        # use the template typename and the default
        extract_template "${template_content}" template_type_str template_type_name_str template_type_default_str
        IFS=';' read -ra template_type_array <<< "$template_type_str"
        IFS=';' read -ra template_type_name_array <<< "$template_type_name_str"
        IFS=';' read -ra template_type_default_array <<< "$template_type_default_str"
        extracted_template="true"
    elif [[ "$line" =~ ${no_template_regex} ]]; then
        extracted_template="true"
        IFS=';' read -ra template_type_array <<< "$template_type_str"
        IFS=';' read -ra template_type_name_array <<< "$template_type_name_str"
        IFS=';' read -ra template_type_default_array <<< "$template_type_default_str"
    fi
    if [[ "$handle_factory" == "false" && "${extracted_template}" == "true" && "${line}" =~ public: ]]; then
        # Early break
        break
    fi
    if [[ "$handle_factory" == "true" && "${extracted_template}" == "true" ]]; then
        if [[ $in_factory == "false" && "$line" =~ "GKO_CREATE_FACTORY_PARAMETERS" ]]; then
            in_factory="true"
        fi
        if [[ $in_factory == "true" && "$line" =~ $factory_param_regex ]]; then
            param_type="${BASH_REMATCH[1]}"
            param_name="${BASH_REMATCH[3]}"
            factory_set="${factory_set}$(prepare_rmfactory_param "$param_type" "$param_name")"
        fi
        if [[ $in_factory == "true" && "$line" =~ "GKO_ENABLE_BUILD_METHOD" ]]; then
            in_facotry="false"
            break
        fi
    fi
done < "temp.hpp"

# Get template_type and class_type
template_type=""
num="${#template_type_array[@]}"
for (( idx = 0; idx < ${num}; idx++ )); do
    if [ -n "$template_type" ]; then
        template_type="$template_type, "
    fi
    template_type="${template_type}${template_type_array[idx]} ${template_type_name_array[idx]}"
done
used_type="${template_type_name_array[*]}"
used_type="${used_type// /, }"
class_type="${namespace}::${class}"
if [[ "$num" -gt "0" ]]; then
    class_type="${class_type}<${used_type}>"
fi


# Generate the file
mkdir -p "$(dirname "${rm_file}")"
GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"
echo "/*${GINKGO_LICENSE_BEACON}" > ${rm_file}
echo "${GINKGO_LICENSE_BEACON}*/" >> ${rm_file}
echo "">>${rm_file}
echo "#ifndef GKO" >> ${rm_file}
echo "#define GKO" >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}
if [[ "${readable}" == "true" ]]; then
    echo "#include <fstream>"  >> ${rm_file}
    echo "" >> ${rm_file}
    echo "">> ${rm_file}
fi
echo "#include <${input_file/include\//}>" >> ${rm_file}
if [[ "${readable}" == "true" ]]; then
    echo "#include <ginkgo/core/base/mtx_io.hpp>" >> ${rm_file}
fi
echo "" >> ${rm_file}
echo "" >> ${rm_file}
echo "#include \"resource_manager/base/generic_constructor.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/helper.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/macro_helper.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/rapidjson_helper.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/resource_manager.hpp\"" >> ${rm_file}
if [[ "${num}" != "0" ]]; then
    # the class contains template
    echo "#include \"resource_manager/base/template_helper.hpp\"" >> ${rm_file}
fi
echo "#include \"resource_manager/base/type_default.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/type_pack.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/type_resolving.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/type_string.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/types.hpp\"" >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}
echo "namespace gko {
namespace extension {
namespace resource_manager {" >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}
# Generate some hint
echo "// TODO: Please add this header file into resource_manager/resource_manager.hpp" >> ${rm_file}
echo "// TODO: Please add the corresponding to the resource_manager/base/types.hpp" >> ${rm_file}
if [[ "$handle_factory" == "true" ]]; then
    echo "// Add _expand(${rm_class_factory}) to ENUM_${base^^}FACTORY" >> ${rm_file}
fi
if [[ "$base" == "LinOp" ]] || [[ "$base" == "Logger" ]]; then
    echo "// Add _expand(${rm_class}) to ENUM_${base^^}" >> ${rm_file}
fi
echo "// If need to override the generated enum for RM, use RM_CLASS or RM_CLASS_FACTORY env and rerun the generated script." >> ${rm_file}
echo "// Or replace the (RM_${base}Factory::)${rm_class_factory} and (RM_${base}::)${rm_class} and their snake case in IMPLEMENT_BRIDGE, ENABLE_SELECTION, *_select, ..." >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}

# Factory Generic part
if [[ "$handle_factory" == "true" ]]; then
    echo "$(generate_factory_generic "${template_type}" "${class_type}" "${num}" "${factory_set}")" >> ${rm_file}
    echo "" >> ${rm_file}
fi
# Simple implementation or another Generic build for non-class
if [[ "${base}" == "LinOp" ]]; then
    if [[ "$handle_factory" == "true" ]]; then
        if [[ "${num}" == "0" ]]; then
            echo "SIMPLE_LINOP_WITH_FACTORY_IMPL_BASE(${namespace}::${class});" >> ${rm_file}
        else
            echo "SIMPLE_LINOP_WITH_FACTORY_IMPL(${namespace}::${class}, $(pack "${template_type}" ${num}), $(pack "${used_type}" ${num}));" >> ${rm_file}
        fi
    else
        echo "$(generate_linop_generic "${template_type}" "${class_type}" "${readable}")" >> ${rm_file}
    fi
    echo "" >> ${rm_file}
elif [[ "${base}" == "Logger" ]]; then
    echo "$(generate_logger_generic "${template_type}" "${class_type}")" >> ${rm_file}
    echo "" >> ${rm_file}
fi
echo "" >> "${rm_file}"

# identify whether the all type in template are type
template_alltype="true"
for (( idx = 0; idx < ${num}; idx++ )); do
    if [[ ! "${template_type_array[idx]}" =~ typename && ! "${template_type_array[idx]}" =~ class ]]; then
        template_alltype="false"
        break
    fi
done

# selection hint and list
selection_suffix=""
selection_addition=""
if [[ "$template_alltype" == "false" ]]; then
    selection_suffix="_ID"
    selection_addition=", RM_${base}, ${rm_class}"
    #
    echo "$(generate_actual_type_hint "${template_type}" "RM_${base}" "${rm_class}" "${namespace}::${class}" template_type_array template_type_name_array)" >> ${rm_file}
fi
rm_class_factory_snake="$(convert_camel_to_snake "${rm_class_factory}")"
rm_class_snake="$(convert_camel_to_snake "${rm_class}")"
if [ "${num}" -gt "0" ]; then
    if [[ "$handle_factory" == "true" ]]; then
        echo "ENABLE_SELECTION${selection_suffix}(${rm_class_factory_snake}_select, call, std::shared_ptr<${base_namespace}${base}Factory>, get_actual_factory_type${selection_addition});" >> ${rm_file}
    fi
    if [[ "$base" == "LinOp" ]] || [[ "$base" == "Logger" ]]; then
        echo "ENABLE_SELECTION${selection_suffix}(${rm_class_snake}_select, call, std::shared_ptr<${base_namespace}${base}>, get_actual_type${selection_addition});" >> ${rm_file}
    fi
    echo "" >> ${rm_file}
    echo "" >> ${rm_file}

    # template list
    echo "constexpr auto ${rm_class_snake}_list = typename span_list<
        $(generate_template_list_content template_type_array template_type_name_array template_type_default_array)
        >::type();" >> ${rm_file}
    echo "" >> ${rm_file}
    echo "" >> ${rm_file}
fi

select_type="${namespace}::${class}"
if [[ "$template_alltype" == "false" ]]; then
    select_type="type_list"
fi
# Create create_from_config
if [[ "${num}" == "0" ]]; then
    if [[ "$handle_factory" == "true" ]]; then
        echo "IMPLEMENT_BRIDGE(RM_${base}Factory, ${rm_class_factory}, ${namespace}::${class}::Factory);" >> "${rm_file}"
    fi
    if [[ "$base" == "LinOp" ]] || [[ "$base" == "Logger" ]]; then
        echo "IMPLEMENT_BRIDGE(RM_${base}, ${rm_class}, ${namespace}::${class});" >> "${rm_file}"
    fi
    echo "" >> "${rm_file}"
else
    type_value_with_default="$(generate_type_value_set template_type_array template_type_name_array template_type_default_array)"
    if [[ "$handle_factory" == "true" ]]; then
    # create_from_config for factory
        echo "$(generate_create_from_config "${base_namespace}${base}Factory" "RM_${base}Factory" \
                "${rm_class_factory}" "${type_value_with_default}" \
                "${rm_class_factory_snake}_select" "${select_type}" "${rm_class_snake}_list")" >> "${rm_file}"
        echo "" >> "${rm_file}"
    fi
    # create_from_config for itself
    if [[ "$base" == "LinOp" ]] || [[ "$base" == "Logger" ]]; then
        echo "$(generate_create_from_config "${base_namespace}${base}" "RM_$base" \
                "${rm_class}" "${type_value_with_default}" \
                "${rm_class_snake}_select" "${select_type}" "${rm_class_snake}_list")" >> "${rm_file}"
        echo "" >> "${rm_file}"
    fi
fi
echo "
}  // namespace resource_manager
}  // namespace extension
}  // namespace gko

" >> "${rm_file}"
echo "#endif  // GKO" >> "${rm_file}"

# Clean
mv .clang-format.tmp .clang-format
./dev_tools/scripts/format_header.sh "${rm_file}"
rm temp.hpp
