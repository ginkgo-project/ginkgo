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
    declare -n namespace_str=$2
    declare -n class_str=$3
    local regex=".*/ginkgo/core/(.*)/([^/]*)\.hpp"
    if [[ $input_path =~ ${regex} ]]; then
        namespace_str="gko::${BASH_REMATCH[1]//\//::}"
        class_str="${BASH_REMATCH[2]}"
    fi
}

contain_factory() {
    local namespace_str="$1"
    if [[ "${namespace_str}" =~ matrix|executor ]]; then
        echo "false"
    else
        echo "true"
    fi
}

prepare_rmfactory_param() {
    local input_type="$1"
    local input_name="$2"
    local vector_regex=" *(std::)?vector<(.*)>$"
    local shared_regex=" *(std::)?shared_ptr<(.*)>$"
    local function_regex=" *(std::)?function<(.*)>"
    local is_vector="false"
    local is_pointer="false"
    if [[ "$input_type" =~ $function_regex ]]; then
        echo "/* TODO: please create a map ${input_name}_map to handle the function ${BASH_REMATCH[2]}*/"
        echo "SET_FUNCTION($input_type, $input_name);"
    fi
    if [[ "$input_type" =~ $vector_regex ]]; then
        is_vector="true"
        input_type="${BASH_REMATCH[2]}"
    fi
    if [[ "$input_type" =~ $shared_regex ]]; then
        is_pointer="true"
        input_type="${BASH_REMATCH[2]}"
    fi
    if [[ "$is_pointer" == "false" && "$is_vector" == "false" ]]; then
        echo "SET_VALUE($input_type, $input_name);"
    elif [[ "$is_pointer" == "false" && "$is_vector" == "false" ]]; then
        echo "assert(false); SET_VALUE_VECTOR($input_type, $input_name); // not yet considered"
    elif [[ "$is_pointer" == "true" && "$is_vector" == "false" ]]; then
        echo "SET_POINTER($input_type, $input_name);"
    else
        echo "SET_POINTER_VECTOR($input_type, $input_name);"
    fi
}

extract_template() {
    local input="$1"
    declare -n type_list=$2
    declare -n type_name=$3
    declare -n type_default=$4
    local type_regex=" *([^=]*) ([^ =]*)( = ([^=]*))?"
    IFS=',' read -ra par <<< "$input"
    temp=""
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
            type_list="$type_list${BASH_REMATCH[1]};"
            type_name="$type_name${BASH_REMATCH[2]};"
            type_default="$type_default${BASH_REMATCH[4]};"
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
    else
        echo "LinOp"
    fi
}

pack() {
    local input="$1"
    local need_pack="$2"
    if [ ${need_pack} -gt "1" ]; then
        echo "PACK($input)"
    else
        echo "$input"
    fi
}

check_exist() {
    local input="$1"
    local source_file="extension/resource_manager/include/resource_manager/base/type_list.hpp"
    local result=$(grep -q "${input}," "$source_file")
    if grep -q "${input}," "$source_file"; then
        echo "true"
    else
        echo "false"
    fi
}

finish() {
    local output="$1"
    mv .clang-format.tmp .clang-format
    ./dev_tools/scripts/format_header.sh "${output}"
    rm temp.hpp
    exit 0
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
rm_class_factory="${rm_class}Factory"
if [[ "${base}" == "Criterion" ]]; then
    rm_class_factory="${rm_class}"
fi
rm_class_factory=${RM_CLASS_FACTORY:="${rm_class_factory}"}
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
template_type_list=""
template_type_name=""
template_type_default=""
template_type_list_array=""
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
        extract_template "${template_content}" template_type_list template_type_name template_type_default
        IFS=';' read -ra template_type_list_array <<< "$template_type_list"
        IFS=';' read -ra template_type_name_array <<< "$template_type_name"
        IFS=';' read -ra template_type_default_array <<< "$template_type_default"
        extracted_template="true"
    elif [[ "$line" =~ ${no_template_regex} ]]; then
        extracted_template="true"
        IFS=';' read -ra template_type_list_array <<< "$template_type_list"
        IFS=';' read -ra template_type_name_array <<< "$template_type_name"
        IFS=';' read -ra template_type_default_array <<< "$template_type_default"
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

mkdir -p "$(dirname "${rm_file}")"
GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"
echo "/*${GINKGO_LICENSE_BEACON}" > ${rm_file}
echo "${GINKGO_LICENSE_BEACON}*/" >> ${rm_file}
echo "">>${rm_file}
echo "#ifndef GKO" >> ${rm_file}
echo "#define GKO" >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}
echo "#include \"resource_manager/base/element_types.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/type_list.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/helper.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/macro_helper.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/rapidjson_helper.hpp\"" >> ${rm_file}
echo "#include \"resource_manager/base/resource_manager.hpp\"" >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}
echo "#include <type_traits>" >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}
echo "namespace gko {
namespace extension {
namespace resource_manager {" >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}
echo "// TODO: Please add the corresponding to the resource_manager/base/types.hpp" >> ${rm_file}
if [[ "$handle_factory" == "true" && "${base}" == "LinOp" ]]; then
    echo "// Add _expand(${rm_class_factory}) to ENUM_${base^^}FACTORY" >> ${rm_file}
fi
if [[ "$handle_factory" == "true" && "${base}" == "Criterion" ]]; then
# check whether we should preseve Factory
    echo "// Add _expand(${rm_class_factory}) to ENUM_${base^^}FACTORY" >> ${rm_file}
fi
if [[ "${base}" == "LinOp" ]]; then
    echo "// Add _expand(${rm_class}) to ENUM_${base^^}" >> ${rm_file} 
fi 
echo "// If need to override the generated enum for RM, use RM_CLASS or RM_CLASS_FACTORY env and rerun the generated script." >> ${rm_file} 
echo "// Or replace the (RM_${base}Factory::)${rm_class_factory} and (RM_${base}::)${rm_class} and their snake case in IMPLEMENT_BRIDGE, ENABLE_SELECTION, *_select, ..." >> ${rm_file} 
echo "" >> ${rm_file}
echo "" >> ${rm_file}
# build Generic function
template_type=""
num="${#template_type_list_array[@]}"
for (( idx = 0; idx < ${num}; idx++ )); do
    if [ -n "$template_type" ]; then
        template_type="$template_type, "
    fi
    template_type="${template_type}${template_type_list_array[idx]} ${template_type_name_array[idx]}"
done
used_type="${template_type_name_array[*]}"
used_type="${used_type// /, }"
class_type="${namespace}::${class}"
if [[ "$num" -gt "0" ]]; then
    class_type="${class_type}<${used_type}>"
fi
if [[ "$handle_factory" == "true" ]]; then
    echo "template <${template_type}>
struct Generic<typename ${class_type}::Factory, ${class_type}> {
    using type = std::shared_ptr<typename ${class_type}::Factory>;
    static type build(rapidjson::Value &item,
                      std::shared_ptr<const Executor> exec,
                      std::shared_ptr<const LinOp> linop,
                      ResourceManager *manager)
    {
        auto ptr = [&]() {
            BUILD_FACTORY($(pack "${class_type}" ${num}), manager, item, exec, linop);
            ${factory_set}
            SET_EXECUTOR;
        }();
        return std::move(ptr);
    }
};
" >> ${rm_file}
fi
if [[ "${num}" == "0" ]]; then
    echo "" >> "${rm_file}"
    echo "" >> "${rm_file}"
    if [[ "${base}" == "Criterion" ]]; then
        echo "IMPLEMENT_BRIDGE(RM_${base}Factory, ${rm_class}, ${namespace}::${class}::Factory);" >> "${rm_file}"
    else
        if [[ "$handle_factory" == "true" ]]; then 
            echo "IMPLEMENT_BRIDGE(RM_${base}Factory, ${rm_class_factory}, ${namespace}::${class}::Factory);" >> "${rm_file}"
        fi
        echo "IMPLEMENT_BRIDGE(RM_${base}, ${rm_class}, ${namespace}::${class});" >> "${rm_file}"
    fi
    echo "

    }  // namespace resource_manager
    }  // namespace extension
    }  // namespace gko

    " >> "${rm_file}"
    echo "#endif  // GKO" >> "${rm_file}"
    finish "${rm_file}"
fi  
if [[ "${base}" == "LinOp" ]]; then
    if [[ "$handle_factory" == "true" ]]; then
        # simple part
        echo "" >> ${rm_file}
        echo "SIMPLE_LINOP_WITH_FACTORY_IMPL(${namespace}::${class}, $(pack "${template_type}" ${num}), $(pack "${used_type}" ${num}));" >> ${rm_file}
    else
        echo "template <${template_type}>
        struct Generic<${class_type}> {
            using type = std::shared_ptr<${class_type}>;
            static type build(rapidjson::Value &item,
                            std::shared_ptr<const Executor> exec,
                            std::shared_ptr<const LinOp> linop,
                            ResourceManager *manager)
            {
                auto exec_ptr =
                get_pointer_check<Executor>(item, \"exec\", exec, linop, manager);
                auto size = get_value_with_default(item, \"dim\", gko::dim<2>{});
                // TODO: consider other thing from constructor
                auto ptr = share(${namespace}::${class}<${used_type}>::create(exec_ptr, size));
                " >> ${rm_file}
        if [[ "${readable}" == "true" ]]; then
            echo "if (item.HasMember(\"read\")) {
            std::ifstream mtx_fd(item[\"read\"].GetString());
            auto data = gko::read_raw<typename ${class_type}::value_type, typename ${class_type}::index_type>(mtx_fd);
            ptr->read(data);
            }
            " >> ${rm_file}
        fi
        echo "return std::move(ptr);
            }
        };" >> ${rm_file}
    fi
fi
echo "" >> ${rm_file}
echo "" >> ${rm_file}
base_namespace=""
if [[ "${base}" == "Criterion" ]]; then
    base_namespace="stop::"
fi
# selection
# adapt with type
template_alltype="true"
for (( idx = 0; idx < ${num}; idx++ )); do
    if [[ ! "${template_type_list_array[idx]}" =~ typename && ! "${template_type_list_array[idx]}" =~ class ]]; then
        template_alltype="false"
        break
    fi
done
selection_suffix=""
selection_addition=""
if [[ "$template_alltype" == "false" ]]; then
    selection_suffix="_ID"
    selection_addition=", RM_${base}, ${rm_class}"
    echo "// TODO: the class contain non type template, please create corresponding actual_type like following" >> ${rm_file}
    actual_template=""
    all_type_template=""
    for (( idx = 0; idx < ${num}; idx++ )); do
        if [ -n "${actual_template}" ]; then
            actual_template="${actual_template}, "
            all_type_template="${all_type_template}, "
        fi
        actual_template="${actual_template}${template_type_name_array[idx]}"
        if [[ ! "${template_type_list_array[idx]}" =~ typename && ! "${template_type_list_array[idx]}" =~ class ]]; then
            all_type_template="${all_type_template}std::integral_constant<${template_type_list_array[idx]}, ${template_type_name_array[idx]}>"
        else
            all_type_template="${all_type_template}${template_type_name_array[idx]}"
        fi
    done
    echo "/*
template <${template_type}>
struct actual_type<type_list<
    std::integral_constant<RM_${base}, RM_${base}::${rm_class}>, ${all_type_template}>> {
    using type = ${namespace}::${class}<${actual_template}>;
};
*/" >> ${rm_file}
fi
rm_class_factory_snake="$(convert_camel_to_snake "${rm_class_factory}")"
rm_class_snake="$(convert_camel_to_snake "${rm_class}")"
if [[ "$handle_factory" == "true" ]]; then
    echo "ENABLE_SELECTION${selection_suffix}(${rm_class_factory_snake}_select, call, std::shared_ptr<gko::${base_namespace}${base}Factory>, get_actual_factory_type${selection_addition});" >> ${rm_file}
fi
if [[ "$base" == "LinOp" ]]; then
    echo "ENABLE_SELECTION${selection_suffix}(${rm_class_snake}_select, call, std::shared_ptr<gko::${base_namespace}${base}>, get_actual_type${selection_addition});" >> ${rm_file}
fi
echo "" >> ${rm_file}
echo "" >> ${rm_file}
# template list
tt_list=""
for (( idx = 0; idx < ${num}; idx++ )); do
    if [ -n "${tt_list}" ]; then
        tt_list="${tt_list}, "
    fi
    # check whether it is exist
    if [[ "$(check_exist "TT_LIST_G_PARTIAL(${template_type_name_array[idx]}")" == "false" ]]; then
        tt_list="${tt_list}/* TODO: can not find ${template_type_name_array[idx]} in tt_list_g, please condider add it if it reused for many times*/"
        # add a empty to easy maintain ,
        tt_list="${tt_list}tt_list<>"
    else
        tt_list="${tt_list}tt_list_g_t<handle_type::${template_type_name_array[idx]}>"
    fi
done
echo "constexpr auto ${rm_class_snake}_list =
    typename span_list<$tt_list>::type();" >> ${rm_file}
echo "" >> ${rm_file}
echo "" >> ${rm_file}
type_value_with_default=""
for (( idx = 0; idx < ${num}; idx++ )); do
    if [ -n "${type_value_with_default}" ]; then
        type_value_with_default="${type_value_with_default}, "
    fi
    # check whether it is exist
    # TODO: it shuold depend on the template list or check exist
    if [[ "$(check_exist "GET_DEFAULT_STRING_PARTIAL(${template_type_name_array[idx]}")" == "false" ]]; then
        type_value_with_default="${type_value_with_default}/*TODO: can not find ${template_type_name_array[idx]} in get_default_string, please condider add it if it reused for many times*/"
        if [ -n "${template_type_default_array[idx]}" ]; then
            # If it contains default value, use it
            type_value_with_default="${type_value_with_default}get_value_with_default(item, \"${template_type_name_array[idx]}\", \"${template_type_default_array[idx]}\")"
        else
            # If it does not contain default value, mark it required
            type_value_with_default="${type_value_with_default}get_required_value<std::string>(item, \"${template_type_name_array[idx]}\")";
        fi
    else
        type_value_with_default="${type_value_with_default}get_value_with_default(item, \"${template_type_name_array[idx]}\", get_default_string<handle_type::${template_type_name_array[idx]}>())"
    fi
done
select_type="${namespace}::${class}"
if [[ "$template_alltype" == "false" ]]; then
    select_type="type_list"
fi
if [[ "$handle_factory" == "true" ]]; then
# create_from_config for factory
echo "template <>
std::shared_ptr<gko::${base_namespace}${base}Factory> create_from_config<
    RM_${base}Factory, RM_${base}Factory::${rm_class_factory}, gko::${base_namespace}${base}Factory>(
    rapidjson::Value &item, std::shared_ptr<const Executor> exec,
    std::shared_ptr<const LinOp> linop, ResourceManager *manager)
{
    // go though the type
    auto type_string = create_type_name( // trick for clang-format
            ${type_value_with_default}
        );
    auto ptr = ${rm_class_factory_snake}_select<${select_type}>(
        ${rm_class_snake}_list, [=](std::string key) { return key == type_string; }, item,
        exec, linop, manager);
    return std::move(ptr);
}" >> ${rm_file}
echo "" >> ${rm_file}
fi
# create_from_config for itself
if [[ "$base" == "LinOp" ]]; then
    echo "template <>
    std::shared_ptr<gko::$base>
    create_from_config<RM_$base, RM_$base::${rm_class}, gko::$base>(
        rapidjson::Value &item, std::shared_ptr<const Executor> exec,
        std::shared_ptr<const LinOp> linop, ResourceManager *manager)
    {
        // go though the type
    auto type_string = create_type_name(  // trick for clang-format
            ${type_value_with_default}
        );
        auto ptr = ${rm_class_snake}_select<${select_type}>(
            ${rm_class_snake}_list, [=](std::string key) { return key == type_string; }, item,
            exec, linop, manager);
        return std::move(ptr);
    }" >> ${rm_file}
    echo "" >> ${rm_file}
fi
echo "
}  // namespace resource_manager
}  // namespace extension
}  // namespace gko

" >> "${rm_file}"
echo "#endif  // GKO" >> "${rm_file}"

finish "${rm_file}"