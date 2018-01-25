#!/usr/bin/env bash

function print_help {
    echo -e "Usage: $0 [options] solvername"
    echo -e "\tOptions:"
    echo -e "\t--dry-run: does a dry run"
    echo -e "\t--help: prints this help"
    echo -e ""
    echo -e "This script uses template files to generate a new solver."
    echo -e "After the correct execution of this script, it is expected that"
    echo -e "solvername is integrated into Ginkgo and that you finish all todos."
    exit
}

execute=1

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
        --*)
            echo -e "bad option $1"
            exit 1
            ;;
        *)
            solvername=$1
            ;;
    esac
    shift
done

if [ "$solvername" == "" ]; then
    print_help
    exit 1
fi

# Important script variable
TMPDIR="./tmp_$(date +%s)"
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
GINKGO_ROOT_DIR="${THIS_DIR}/../.."
solvername=${solvername,,}
Solvername=${solvername^}
SOLVERNAME=${solvername^^}

TEMPLATE_FILES=(
    "solver.cpp"
    "solver.hpp"
    "solver_kernels.hpp"
    "solver_reference_kernels.cpp"
    "solver_cpu_kernels.cpp"
    "solver_gpu_kernels.cu"
    "solver_core_test.cpp"
    "solver_reference_test.cpp"
    "solver_cpu_test.cpp"
    "solver_gpu_test.cpp"
)
TEMPLATE_FILES_LOCATIONS=(
    "core/solver"
    "core/solver"
    "core/solver"
    "reference/solver"
    "cpu/solver"
    "gpu/solver"
    "core/test/solver"
    "reference/test/solver"
    "cpu/test/solver"
    "gpu/test/solver"
)
TEMPLATE_FILES_TYPES=(
    "solver file"
    "class header"
    "kernel header"
    "kernel file"
    "kernel file"
    "kernel file"
    "unit tests for ${solvername} solver"
    "unit tests for ${solvername} reference kernels"
    "unit tests for ${solvername} CPU kernels"
    "unit tests for ${solvername} GPU kernels"
)
TEMPLATE_FILES_DESCRIPTIONS=(
    "This is where the ${solvername} algorithm needs to be implemented."
    "This is where the ${solvername} class functions need to be implemented."
    "This is where the algorithm-specific kernels need to be added."
    "Reference kernels for ${solvername} need to be implemented here."
    "CPU kernels for ${solvername} need to be implemented here."
    "GPU kernels for ${solvername} need to be implemented here."
    ""
    ""
    ""
    ""
)

mkdir ${TMPDIR}

if [ -f ${GINKGO_ROOT_DIR}/core/solver/${solvername}.cpp ]; then
    echo "Error: a solver with this name exists."
    echo "Choose a different name."
    exit 1
fi

# create folder for temporary files

# copy files needed into temporary folder
for i in "${TEMPLATE_FILES[@]}"
do
    cp ${THIS_DIR}/../templates/$i ${TMPDIR}/.
done

# search and replace keywords with new solver name
perl -pi -e "s/XXSOLVERXX/$SOLVERNAME/g" ${TMPDIR}/*
perl -pi -e "s/Xxsolverxx/$Solvername/g" ${TMPDIR}/*
perl -pi -e "s/xxsolverxx/$solvername/g" ${TMPDIR}/*
echo -e "\nCreating temporary files:"
for i in "${TEMPLATE_FILES[@]}"
do
    ls ${TMPDIR}/$i
done

if [ $execute == 1 ]
then
    echo -e "\nRenaming and distributing files"
    # rename and distribute the files to the right location
    # for each file, make sure it does not exist yet
    for (( i=1; i<${#TEMPLATE_FILES[@]}+1; i++ ))
    do
        filename=$(echo ${TEMPLATE_FILES[$i-1]} | sed "s/solver/${solvername}/")
        destpath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/$filename
        if [ ! -f ${GINKGO_ROOT_DIR}/$destpath ]; then
            cp ${TMPDIR}/${TEMPLATE_FILES[$i-1]} ${GINKGO_ROOT_DIR}/$destpath
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

if [ -f todo_${solvername}.txt ]; then
    rm todo_${solvername}.txt
fi

echo -e "\nSummary:"                                                                 | tee -a todo_${solvername}.txt
for (( i=1; i<${#TEMPLATE_FILES[@]}+1; i++ ))
do
    filename=$(echo ${TEMPLATE_FILES[$i-1]} | sed "s/solver/${solvername}/")
    destpath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/$filename

    echo "Created ${TEMPLATE_FILES_TYPES[$i-1]}"         | tee -a todo_${solvername}.txt
    echo "$destpath"                                     | tee -a todo_${solvername}.txt
    if [ "${TEMPLATE_FILES_DESCRIPTIONS[$i-1]}" != "" ]
    then
        echo -e "\t${TEMPLATE_FILES_DESCRIPTIONS[$i-1]}" | tee -a todo_${solvername}.txt
    fi
    echo ""                                              | tee -a todo_${solvername}.txt
done

echo "All tests have to be modified to the specific solver characteristics."    | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "The following CMakeLists have to be modified manually:"                   | tee -a todo_${solvername}.txt
echo "core/CMakeLists.txt"                                                      | tee -a todo_${solvername}.txt
echo "core/test/solver/CMakeLists.txt"                                          | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "reference/CMakeLists.txt"                                                 | tee -a todo_${solvername}.txt
echo "reference/test/solver/CMakeLists.txt"                                     | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "cpu/CMakeLists.txt"                                                       | tee -a todo_${solvername}.txt
echo "cpu/test/solver/CMakeLists.txt"                                           | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "gpu/CMakeLists.txt"                                                       | tee -a todo_${solvername}.txt
echo "gpu/test/solver/CMakeLists.txt"                                           | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "The following header file has to modified:"                               | tee -a todo_${solvername}.txt
echo "core/device_hooks/common_kernels.inc.cpp"                                 | tee -a todo_${solvername}.txt
echo "Equivalent to the other solvers, the following part has to be appended:"  | tee -a todo_${solvername}.txt
echo "#######################################################################"  | tee -a todo_${solvername}.txt
echo "namespace  ${solvername} {"                                                  | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "// template <typename ValueType>"                                         | tee -a todo_${solvername}.txt
echo "// GKO_DECLARE_${SOLVERNAME}_INITIALIZE_KERNEL(ValueType)"                  | tee -a todo_${solvername}.txt
echo "// NOT_COMPILED(GKO_HOOK_MODULE);"                                        | tee -a todo_${solvername}.txt
echo "// GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_${SOLVERNAME}_INITIALIZE_KERNEL);" | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "// ..."                                                                   | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "}  // namespace ${solvername}"                                               | tee -a todo_${solvername}.txt
echo "#######################################################################"  | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "A summary of the required next steps has been written to:"
echo "todo_${solvername}.txt"
