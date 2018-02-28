#!/usr/bin/env bash

# Important script variable
execute=1
TMPDIR="./tmp_$(date +%s)"
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
GINKGO_ROOT_DIR="${THIS_DIR}/../.."

function print_help {
    echo -e "Usage: $0 [options] solvername sourcetype sourcename"
    echo -e "\tOptions:"
    echo -e "\t--dry-run: does a dry run"
    echo -e "\t--help: prints this help"
    echo -e "\t--list: prints the list of possible sources"
    echo -e ""
    echo -e "This script generates a new solver or preconditioner from an"
    echo -e "existing one. --list can be use to list existing sources."
    echo -e "After the correct execution of this script, it is expected that"
    echo -e "solvername is integrated into Ginkgo and that you finish all todos."
    exit
}

function list_sources {
    for type in solver preconditioner
    do
        for i in $(ls $GINKGO_ROOT_DIR/core/$type/*.cpp)
        do
            echo "$type "$(basename "$i" | cut -d"." -f1)
        done
    done
    exit
}

if [ $# -lt 1 ]; then
    print_help
    exit 1
fi

while test $# -gt 0
do
    case "$1" in
        --help)
            print_help
            ;;
        --dry-run)
            execute=0
            echo -e "Doing a dry run."
            ;;
        --list)
            list_sources
            exit 1
            ;;
        --*)
            echo -e "bad option $1"
            exit 1
            ;;
        *)
            name=$1
            source_type=$2
            source_name=$3
            shift
            shift
            ;;
    esac
    shift
done

source_name=${source_name,,}
name=${name,,}
Name=${name^}
NAME=${name^^}

if [[ "$name" == "" ]] || ( [[ "$source_type" != "preconditioner" ]] && [[ "$source_type" != "solver" ]] ); then
    print_help
    exit 1
fi

if [ -f ${GINKGO_ROOT_DIR}/core/${source_type}/${name}.cpp ]; then
    echo "Error: a ${source_type} with this name exists."
    echo "Choose a different name."
    exit 1
fi

if [ ! -f ${GINKGO_ROOT_DIR}/core/${source_type}/${source_name}.cpp ]; then
    echo "Error: a ${source_type} named ${source_name} does not exist."
    echo "Use --list to list possible sources."
    exit 1
fi

TEMPLATE_FILES=(
    "${name}.cpp"
    "${name}.hpp"
    "${name}_kernels.hpp"
    "${name}_kernels.cpp"
    "${name}_kernels.cpp"
    "${name}_kernels.cu"
    "${name}.cpp"
    "${name}_kernels.cpp"
#    "${name}_kernels.cpp"
    "${name}_kernels.cpp"
)
TEMPLATE_FILES_LOCATIONS=(
    "core/$source_type"
    "core/$source_type"
    "core/$source_type"
    "reference/$source_type"
    "cpu/$source_type"
    "gpu/$source_type"
    "core/test/$source_type"
    "reference/test/$source_type"
#    "cpu/test/$source_type"
    "gpu/test/$source_type"
)
TEMPLATE_FILES_TYPES=(
    "$source_type file"
    "class header"
    "kernel header"
    "kernel file"
    "kernel file"
    "kernel file"
    "unit tests for ${name} $type"
    "unit tests for ${name} reference kernels"
#    "unit tests for ${name} CPU kernels"
    "unit tests for ${name} GPU kernels"
)
TEMPLATE_FILES_DESCRIPTIONS=(
    "This is where the ${name} algorithm needs to be implemented."
    "This is where the ${name} class functions need to be implemented."
    "This is where the algorithm-specific kernels need to be added."
    "Reference kernels for ${name} need to be implemented here."
    "CPU kernels for ${name} need to be implemented here."
    "GPU kernels for ${name} need to be implemented here."
    ""
    ""
#    ""
    ""
)

mkdir ${TMPDIR}

# create folder for temporary files

# copy files needed into temporary folder
for (( i=1; i<${#TEMPLATE_FILES[@]}+1; i++ ))
do
    sourcename=$(echo ${TEMPLATE_FILES[$i-1]} | sed "s/${name}/${source_name}/")
    sourcepath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${sourcename}
    destpath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${TEMPLATE_FILES[$i-1]}
    mkdir -p ${TMPDIR}/${TEMPLATE_FILES_LOCATIONS[$i-1]}
    cp ${GINKGO_ROOT_DIR}/$sourcepath ${TMPDIR}/$destpath
done

# search and replace keywords with new solver name
echo -e "\nCreating temporary files:"
for (( i=1; i<${#TEMPLATE_FILES[@]}+1; i++ ))
do
    destpath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${TEMPLATE_FILES[$i-1]}
    perl -pi -e "s/${source_name}/$name/g" ${TMPDIR}/$destpath
    perl -pi -e "s/${source_name^}/$Name/g" ${TMPDIR}/$destpath
    perl -pi -e "s/${source_name^^}/$NAME/g" ${TMPDIR}/$destpath
    ls ${TMPDIR}/$destpath
done

if [ $execute == 1 ]
then
    echo -e "\nRenaming and distributing files"
    # rename and distribute the files to the right location
    # for each file, make sure it does not exist yet
    for (( i=1; i<${#TEMPLATE_FILES[@]}+1; i++ ))
    do
        sourcepath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${TEMPLATE_FILES[$i-1]}
        destpath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${TEMPLATE_FILES[$i-1]}
        if [ ! -f ${GINKGO_ROOT_DIR}/$destpath ]; then
            cp ${TMPDIR}/${sourcepath} ${GINKGO_ROOT_DIR}/${destpath}
        else
            echo -e "Error: file ${GINKGO_ROOT_DIR}/$destpath exists"
            echo -e "Remove file first if you want to replace it."
            read -p ""
        fi
    done


    echo -e "cleaning up temporary files."
    rm -rf ${TMPDIR}
else
    echo -e "\nNo file was copied because --dry-run was used"
    echo -e "You can inspect the generated solver files in ${TMPDIR}."
fi

if [ -f todo_${name}.txt ]; then
    rm todo_${name}.txt
fi

echo -e "\nSummary:"                                                                 | tee -a todo_${name}.txt
for (( i=1; i<${#TEMPLATE_FILES[@]}+1; i++ ))
do
    destpath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${TEMPLATE_FILES[$i-1]}

    echo "Created ${TEMPLATE_FILES_TYPES[$i-1]}"         | tee -a todo_${name}.txt
    echo "$destpath"                                     | tee -a todo_${name}.txt
    if [ "${TEMPLATE_FILES_DESCRIPTIONS[$i-1]}" != "" ]
    then
        echo -e "\t${TEMPLATE_FILES_DESCRIPTIONS[$i-1]}" | tee -a todo_${name}.txt
    fi
    echo ""                                              | tee -a todo_${name}.txt
done

echo "All tests have to be modified to the specific ${source_type} characteristics."    | tee -a todo_${name}.txt
echo ""                                                                         | tee -a todo_${name}.txt
echo ""                                                                         | tee -a todo_${name}.txt
echo "The following CMakeLists have to be modified manually:"                   | tee -a todo_${name}.txt
echo "core/CMakeLists.txt"                                                      | tee -a todo_${name}.txt
echo "core/test/${source_type}/CMakeLists.txt"                                          | tee -a todo_${name}.txt
echo ""                                                                         | tee -a todo_${name}.txt
echo "reference/CMakeLists.txt"                                                 | tee -a todo_${name}.txt
echo "reference/test/${source_type}/CMakeLists.txt"                                     | tee -a todo_${name}.txt
echo ""                                                                         | tee -a todo_${name}.txt
echo "cpu/CMakeLists.txt"                                                       | tee -a todo_${name}.txt
echo "cpu/test/${source_type}/CMakeLists.txt"                                           | tee -a todo_${name}.txt
echo ""                                                                         | tee -a todo_${name}.txt
echo "gpu/CMakeLists.txt"                                                       | tee -a todo_${name}.txt
echo "gpu/test/${source_type}/CMakeLists.txt"                                           | tee -a todo_${name}.txt
echo ""                                                                         | tee -a todo_${name}.txt
echo ""                                                                         | tee -a todo_${name}.txt
echo "The following header file has to modified:"                               | tee -a todo_${name}.txt
echo "core/device_hooks/common_kernels.inc.cpp"                                 | tee -a todo_${name}.txt
echo "Equivalent to the other solvers, the following part has to be appended:"  | tee -a todo_${name}.txt
echo "##################################################" | tee -a todo_${name}.txt
echo "#include #include \"core/solver/test_kernels.hpp\"" | tee -a todo_${name}.txt
echo "// ..."                                             | tee -a todo_${name}.txt
echo "namespace  ${name} {"                               | tee -a todo_${name}.txt
echo ""                                                   | tee -a todo_${name}.txt
echo ""                                                   | tee -a todo_${name}.txt
echo "// template <typename ValueType>"                   | tee -a todo_${name}.txt
echo "// GKO_DECLARE_${NAME}_INITIALIZE_KERNEL(ValueType)"| tee -a todo_${name}.txt
echo "// NOT_COMPILED(GKO_HOOK_MODULE);"                  | tee -a todo_${name}.txt
echo "// GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_${NAME}_INITIALIZE_KERNEL);" | tee -a todo_${name}.txt
echo ""                                                   | tee -a todo_${name}.txt
echo "// ..."                                             | tee -a todo_${name}.txt
echo ""                                                   | tee -a todo_${name}.txt
echo ""                                                   | tee -a todo_${name}.txt
echo "}  // namespace ${name}"                            | tee -a todo_${name}.txt
echo "##################################################" | tee -a todo_${name}.txt
echo ""                                                   | tee -a todo_${name}.txt
echo ""                                                   | tee -a todo_${name}.txt
echo "A summary of the required next steps has been written to:"
echo "todo_${name}.txt"
