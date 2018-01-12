#!/bin/bash

function print_help {
    echo -e "usage: $0 SOLVER_NAME"
    echo -e ""
    echo -e "This script uses template files to generate a new solver."
    echo -e "After the correct execution of this script, it is expected that"
    echo -e "SOLVER_NAME is integrated into Ginkgo and that you finish all todos."
    exit
}

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


# Important script variable
execute=1
TMPDIR="./tmp_$(date +%s)"
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
GINKGO_ROOT_DIR="${THIS_DIR}/../.."
Solvername=${solvername^}
SOLVERNAME=${solvername^^}

mkdir ${TMPDIR}

if [ -f ${GINKGO_ROOT_DIR}/core/solver/${solvername}.cpp ]; then
    echo "Error: a solver with this name exists."
    echo "Choose a different name."
    exit 1
fi

# create folder for temporary files

# copy files needed into temporary folder
cp ${THIS_DIR}/../templates/solver.cpp ${TMPDIR}/.
cp ${THIS_DIR}/../templates/solver.hpp ${TMPDIR}/.
cp ${THIS_DIR}/../templates/solver_kernels.hpp ${TMPDIR}/.
cp ${THIS_DIR}/../templates/solver_reference_kernels.cpp ${TMPDIR}/.
cp ${THIS_DIR}/../templates/solver_cpu_kernels.cpp ${TMPDIR}/.
cp ${THIS_DIR}/../templates/solver_gpu_kernels.cu ${TMPDIR}/.

cp ${THIS_DIR}/../templates/solver_core_test.cpp ${TMPDIR}/.
cp ${THIS_DIR}/../templates/solver_reference_test.cpp ${TMPDIR}/.
cp ${THIS_DIR}/../templates/solver_cpu_test.cpp ${TMPDIR}/.
cp ${THIS_DIR}/../templates/solver_gpu_test.cpp ${TMPDIR}/.

# search and replace keywords with new solver name
perl -pi -e "s/XXSOLVERXX/$SOLVERNAME/g" ${TMPDIR}/*
perl -pi -e "s/Xxsolverxx/$Solvername/g" ${TMPDIR}/*
perl -pi -e "s/xxsolverxx/$solvername/g" ${TMPDIR}/*
echo ""
echo "Create temporary files:"
echo ""
ls ${TMPDIR}/solver.cpp
ls ${TMPDIR}/solver.hpp
ls ${TMPDIR}/solver_kernels.hpp
ls ${TMPDIR}/solver_reference_kernels.cpp
ls ${TMPDIR}/solver_cpu_kernels.cpp
ls ${TMPDIR}/solver_gpu_kernels.cu
echo ""
ls ${TMPDIR}/solver_core_test.cpp
ls ${TMPDIR}/solver_reference_test.cpp
ls ${TMPDIR}/solver_cpu_test.cpp
ls ${TMPDIR}/solver_gpu_test.cpp
echo ""

if [ $execute == 1 ]
then
    echo "renaming and distributing files"
    # rename and distribute the files to the right location
    # for each file, make sure it does not exist yet
    
    # first the kernel and header files
    if [ ! -f ${GINKGO_ROOT_DIR}/core/solver/${solvername}.cpp ]; then
        cp ${TMPDIR}/solver.cpp ${GINKGO_ROOT_DIR}/core/solver/${solvername}.cpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/core/solver/${solvername}.cpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ${GINKGO_ROOT_DIR}/core/solver/${solvername}.hpp ]; then
        cp ${TMPDIR}/solver.hpp ${GINKGO_ROOT_DIR}/core/solver/${solvername}.hpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/core/solver/${solvername}.hpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ${GINKGO_ROOT_DIR}/core/solver/${solvername}_kernels.hpp ]; then
        cp ${TMPDIR}/solver_kernels.hpp ${GINKGO_ROOT_DIR}/core/solver/${solvername}_kernels.hpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/core/solver/${solvername}_kernels.hpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ${GINKGO_ROOT_DIR}/reference/solver/${solvername}_kernels.cpp ]; then
        cp ${TMPDIR}/solver_reference_kernels.cpp ${GINKGO_ROOT_DIR}/reference/solver/${solvername}_kernels.cpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/reference/solver/${solvername}_kernels.cpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ${GINKGO_ROOT_DIR}/cpu/solver/${solvername}_kernels.cpp ]; then
        cp ${TMPDIR}/solver_cpu_kernels.cpp ${GINKGO_ROOT_DIR}/cpu/solver/${solvername}_kernels.cpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/cpu/solver/${solvername}_kernels.cpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ${GINKGO_ROOT_DIR}/gpu/solver/${solvername}_kernels.cu ]; then
        cp ${TMPDIR}/solver_gpu_kernels.cu ${GINKGO_ROOT_DIR}/gpu/solver/${solvername}_kernels.cu
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/gpu/solver/${solvername}_kernels.cu exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
#    cp -n ${TMPDIR}/solver.cpp ${GINKGO_ROOT_DIR}/core/solver/${solvername}.cpp
#    cp -n ${TMPDIR}/solver.hpp ${GINKGO_ROOT_DIR}/core/solver/${solvername}.hpp
#    cp -n ${TMPDIR}/solver_kernels.hpp ${GINKGO_ROOT_DIR}/core/solver/${solvername}_kernels.hpp
#    cp -n ${TMPDIR}/solver_reference_kernels.cpp ${GINKGO_ROOT_DIR}/reference/solver/${solvername}_kernels.cpp
#    cp -n ${TMPDIR}/solver_cpu_kernels.cpp ${GINKGO_ROOT_DIR}/cpu/solver/${solvername}_kernels.cpp
#    cp -n ${TMPDIR}/solver_gpu_kernels.cu ${GINKGO_ROOT_DIR}/gpu/solver/${solvername}_kernels.cu
    
    # now the unit tests
    if [ ! -f ${GINKGO_ROOT_DIR}/core/test/solver/${solvername}.cpp ]; then
        cp ${TMPDIR}/solver_core_test.cpp ${GINKGO_ROOT_DIR}/core/test/solver/${solvername}.cpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/core/test/solver/${solvername}.cpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ${GINKGO_ROOT_DIR}/reference/test/solver/${solvername}_kernels.cpp ]; then
        cp ${TMPDIR}/solver_reference_test.cpp ${GINKGO_ROOT_DIR}/reference/test/solver/${solvername}_kernels.cpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/reference/test/solver/${solvername}_kernels.cpp"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ${GINKGO_ROOT_DIR}/cpu/test/solver/${solvername}_kernels.cpp ]; then
        cp ${TMPDIR}/solver_cpu_test.cpp ${GINKGO_ROOT_DIR}/cpu/test/solver/${solvername}_kernels.cpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/cpu/test/solver/${solvername}_kernels.cpp"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ${GINKGO_ROOT_DIR}/gpu/test/solver/${solvername}_kernels.cpp ]; then
        cp ${TMPDIR}/solver_gpu_test.cpp ${GINKGO_ROOT_DIR}/gpu/test/solver/${solvername}_kernels.cpp
    else
        echo "Error: file ${GINKGO_ROOT_DIR}/gpu/test/solver/${solvername}_kernels.cpp"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
#    cp -n ${TMPDIR}/solver_core_test.cpp ${GINKGO_ROOT_DIR}/core/test/solver/${solvername}.cpp
#    cp -n ${TMPDIR}/solver_reference_test.cpp ${GINKGO_ROOT_DIR}/reference/test/solver/${solvername}_kernels.cpp
#    cp -n ${TMPDIR}/solver_cpu_test.cpp ${GINKGO_ROOT_DIR}/cpu/test/solver/${solvername}_kernels.cpp
#    cp -n ${TMPDIR}/solver_gpu_test.cpp ${GINKGO_ROOT_DIR}/gpu/test/solver/${solvername}_kernels.cpp
    
    
    # clean up temporary folder
else
    echo ""
fi



if [ -f todo_${solvername}.txt ]; then
    rm todo_${solvername}.txt
fi

echo "Summary:"                                                                 | tee -a todo_${solvername}.txt
echo "Created solver file"                                                      | tee -a todo_${solvername}.txt
echo "ginkgo/core/solver/${solvername}.cpp"                                        | tee -a todo_${solvername}.txt
echo "    This is where the ${solvername} algorithm needs to be implemented."      | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created class header"                                                     | tee -a todo_${solvername}.txt
echo "ginkgo/core/solver/${solvername}.hpp"                                        | tee -a todo_${solvername}.txt
echo "    This is where the ${solvername} class functions need to be implemented." | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created kernel header"                                                    | tee -a todo_${solvername}.txt
echo "ginkgo/core/solver/${solvername}_kernels.hpp"                                | tee -a todo_${solvername}.txt
echo "    This is where the algorithm-specific kernels need to be added."       | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created kernel file"                                                      | tee -a todo_${solvername}.txt
echo "ginkgo/reference/solver/${solvername}_kernels.cpp"                           | tee -a todo_${solvername}.txt
echo "    Reference kernels for ${solvername} need to be implemented here."        | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created kernel file"                                                      | tee -a todo_${solvername}.txt
echo "ginkgo/cpu/solver/${solvername}_kernels.cpp"                                 | tee -a todo_${solvername}.txt
echo "    CPU kernels for ${solvername} need to be implemented here."              | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created kernel file"                                                      | tee -a todo_${solvername}.txt
echo "ginkgo/gpu/solver/${solvername}_kernels.cu"                                  | tee -a todo_${solvername}.txt
echo "    GPU kernels for ${solvername} need to be implemented here."              | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created unit tests for ${solvername} solver in:"                             | tee -a todo_${solvername}.txt
echo "ginkgo/core/test/solver/${solvername}.cpp"                                   | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created unit tests for ${solvername} reference kernels in"                   | tee -a todo_${solvername}.txt
echo "ginkgo/reference/test/solver/${solvername}_kernels.cpp"                      | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created unit tests for ${solvername} CPU kernels in"                         | tee -a todo_${solvername}.txt
echo "ginkgo/cpu/test/solver/${solvername}_kernels.cpp"                            | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
echo "Created unit tests for ${solvername} GPU kernels in"                         | tee -a todo_${solvername}.txt
echo "ginkgo/gpu/test/solver/${solvername}_kernels.cpp"                            | tee -a todo_${solvername}.txt
echo ""                                                                         | tee -a todo_${solvername}.txt
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

echo -e "cleaning up temporary files."
rm -rf ${TMPDIR}
