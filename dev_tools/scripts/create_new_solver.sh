#!/bin/bash

# Ask the developer for the solver name
echo 'What is the name of the new solver? (all lowercase)'
read varname
echo 'Selected solver name:' ${varname}
varname2=${varname^}
varname3=${varname^^}
echo 'Mixed letter solver name:' ${varname2}
echo 'Capital letter solver name:' ${varname3}

echo ""

if [ -f ../../core/solver/${varname}.cpp ]; then
    echo "Error: a solver with this name exists."
    echo "Choose a different name."
    exit 1
fi


echo 'Execute file generation? [yes/no]' 
read execute
if [ $execute == "yes" ]
then
    echo "Execute."
else
    echo "Dry run."
fi

# create folder for temporary files
mkdir tmp

# copy files needed into temporary folder
cp ../templates/solver.cpp tmp/.
cp ../templates/solver.hpp tmp/.
cp ../templates/solver_kernels.hpp tmp/.
cp ../templates/solver_reference_kernels.cpp tmp/.
cp ../templates/solver_cpu_kernels.cpp tmp/.
cp ../templates/solver_gpu_kernels.cu tmp/.

cp ../templates/solver_core_test.cpp tmp/.
cp ../templates/solver_reference_test.cpp tmp/.
cp ../templates/solver_cpu_test.cpp tmp/.
cp ../templates/solver_gpu_test.cpp tmp/.

# search and replace keywords with new solver name
perl -pi -e "s/XXSOLVERXX/$varname3/g" tmp/*
perl -pi -e "s/Xxsolverxx/$varname2/g" tmp/*
perl -pi -e "s/xxsolverxx/$varname/g" tmp/*
echo ""
echo "Create temporary files:"
echo ""
ls tmp/solver.cpp
ls tmp/solver.hpp
ls tmp/solver_kernels.hpp
ls tmp/solver_reference_kernels.cpp
ls tmp/solver_cpu_kernels.cpp
ls tmp/solver_gpu_kernels.cu
echo ""
ls tmp/solver_core_test.cpp
ls tmp/solver_reference_test.cpp
ls tmp/solver_cpu_test.cpp
ls tmp/solver_gpu_test.cpp
echo ""

if [ $execute == "yes" ]
then
    echo "renaming and distributing files"
    # rename and distribute the files to the right location
    # for each file, make sure it does not exist yet
    
    # first the kernel and header files
    if [ ! -f ../../core/solver/${varname}.cpp ]; then
        cp tmp/solver.cpp ../../core/solver/${varname}.cpp
    else
        echo "Error: file ../../core/solver/${varname}.cpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ../../core/solver/${varname}.hpp ]; then
        cp tmp/solver.hpp ../../core/solver/${varname}.hpp
    else
        echo "Error: file ../../core/solver/${varname}.hpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ../../core/solver/${varname}_kernels.hpp ]; then
        cp tmp/solver_kernels.hpp ../../core/solver/${varname}_kernels.hpp
    else
        echo "Error: file ../../core/solver/${varname}_kernels.hpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ../../reference/solver/${varname}_kernels.cpp ]; then
        cp tmp/solver_reference_kernels.cpp ../../reference/solver/${varname}_kernels.cpp
    else
        echo "Error: file ../../reference/solver/${varname}_kernels.cpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ../../cpu/solver/${varname}_kernels.cpp ]; then
        cp tmp/solver_cpu_kernels.cpp ../../cpu/solver/${varname}_kernels.cpp
    else
        echo "Error: file ../../cpu/solver/${varname}_kernels.cpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ../../gpu/solver/${varname}_kernels.cu ]; then
        cp tmp/solver_gpu_kernels.cu ../../gpu/solver/${varname}_kernels.cu
    else
        echo "Error: file ../../gpu/solver/${varname}_kernels.cu exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
#    cp -n tmp/solver.cpp ../../core/solver/${varname}.cpp
#    cp -n tmp/solver.hpp ../../core/solver/${varname}.hpp
#    cp -n tmp/solver_kernels.hpp ../../core/solver/${varname}_kernels.hpp
#    cp -n tmp/solver_reference_kernels.cpp ../../reference/solver/${varname}_kernels.cpp
#    cp -n tmp/solver_cpu_kernels.cpp ../../cpu/solver/${varname}_kernels.cpp
#    cp -n tmp/solver_gpu_kernels.cu ../../gpu/solver/${varname}_kernels.cu
    
    # now the unit tests
    if [ ! -f ../../core/test/solver/${varname}.cpp ]; then
        cp tmp/solver_core_test.cpp ../../core/test/solver/${varname}.cpp
    else
        echo "Error: file ../../core/test/solver/${varname}.cpp exists"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ../../reference/test/solver/${varname}_kernels.cpp ]; then
        cp tmp/solver_reference_test.cpp ../../reference/test/solver/${varname}_kernels.cpp
    else
        echo "Error: file ../../reference/test/solver/${varname}_kernels.cpp"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ../../cpu/test/solver/${varname}_kernels.cpp ]; then
        cp tmp/solver_cpu_test.cpp ../../cpu/test/solver/${varname}_kernels.cpp
    else
        echo "Error: file ../../cpu/test/solver/${varname}_kernels.cpp"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
    if [ ! -f ../../gpu/test/solver/${varname}_kernels.cpp ]; then
        cp tmp/solver_gpu_test.cpp ../../gpu/test/solver/${varname}_kernels.cpp
    else
        echo "Error: file ../../gpu/test/solver/${varname}_kernels.cpp"
        echo "Remove file first if you want to replace it."
        read -p ""
    fi
#    cp -n tmp/solver_core_test.cpp ../../core/test/solver/${varname}.cpp
#    cp -n tmp/solver_reference_test.cpp ../../reference/test/solver/${varname}_kernels.cpp
#    cp -n tmp/solver_cpu_test.cpp ../../cpu/test/solver/${varname}_kernels.cpp
#    cp -n tmp/solver_gpu_test.cpp ../../gpu/test/solver/${varname}_kernels.cpp
    
    
    # clean up temporary folder
    echo "cleaning up temporary files."
    rm -rf tmp
else
    echo ""
fi

echo ""
echo ""
echo "Summary:"
echo "Created solver file"
echo "ginkgo/core/solver/${varname}.cpp" 
echo "    This is where the ${varname} algorithm needs to be implemented."
echo ""
echo "Created class header"
echo "ginkgo/core/solver/${varname}.hpp" 
echo "    This is where the ${varname} class functions need to be implemented."
echo ""
echo "Created kernel header"
echo "ginkgo/core/solver/${varname}_kernels.hpp"
echo "    This is where the algorithm-specific kernels need to be added."
echo ""
echo "Created kernel file"
echo "ginkgo/reference/solver/${varname}_kernels.cpp"
echo "    Reference kernels for ${varname} need to be implemented here."
echo ""
echo "Created kernel file"
echo "ginkgo/cpu/solver/${varname}_kernels.cpp" 
echo "    CPU kernels for ${varname} need to be implemented here."
echo ""
echo "Created kernel file"
echo "ginkgo/gpu/solver/${varname}_kernels.cu" 
echo "    GPU kernels for ${varname} need to be implemented here."
echo ""
echo "Created unit tests for ${varname} solver in:"
echo "ginkgo/core/test/solver/${varname}.cpp"
echo ""
echo "Created unit tests for ${varname} reference kernels in"
echo "ginkgo/reference/test/solver/${varname}_kernels.cpp"
echo ""
echo "Created unit tests for ${varname} CPU kernels in"
echo "ginkgo/cpu/test/solver/${varname}_kernels.cpp"
echo ""
echo "Created unit tests for ${varname} GPU kernels in"
echo "ginkgo/gpu/test/solver/${varname}_kernels.cpp"
echo ""
echo "All tests have to be modified to the specific solver characteristics."
echo ""
echo ""
echo "The following CMakeLists have to be modified manually:"
echo "core/CMakeLists.txt"
echo "core/test/solver/CMakeLists.txt"
echo ""
echo "reference/CMakeLists.txt"
echo "reference/test/solver/CMakeLists.txt"
echo ""
echo "cpu/CMakeLists.txt"
echo "cpu/test/solver/CMakeLists.txt"
echo ""
echo "gpu/CMakeLists.txt"
echo "gpu/test/solver/CMakeLists.txt"
echo ""
echo ""
echo "The following header file has to modified:"
echo "core/device_hooks/common_kernels.inc.cpp"
echo "Similar to the other solvers, the following part needs to be appended:"
echo ""
echo "#######################################################################"
echo "namespace  ${varname} {"
echo ""
echo ""
echo "template <typename ValueType>"
echo "GKO_DECLARE_${varname3}_INITIALIZE_KERNEL(ValueType)"
echo "NOT_COMPILED(GKO_HOOK_MODULE);"
echo "GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_${varname3}_INITIALIZE_KERNEL);"
echo ""
echo "..."
echo ""
echo ""
echo "}  // namespace ${varname}"
echo "#######################################################################"
echo ""
echo ""
echo "A summary of the required next steps has been written to:"
echo "todo_${varname}.txt"


if [ -f todo_${varname}.txt ]; then
    rm todo_${varname}.txt
fi

echo "Summary:"                                                                 >> todo_${varname}.txt
echo "Created solver file"                                                      >> todo_${varname}.txt
echo "ginkgo/core/solver/${varname}.cpp"                                        >> todo_${varname}.txt
echo "    This is where the ${varname} algorithm needs to be implemented."      >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created class header"                                                     >> todo_${varname}.txt
echo "ginkgo/core/solver/${varname}.hpp"                                        >> todo_${varname}.txt
echo "    This is where the ${varname} class functions need to be implemented." >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created kernel header"                                                    >> todo_${varname}.txt
echo "ginkgo/core/solver/${varname}_kernels.hpp"                                >> todo_${varname}.txt
echo "    This is where the algorithm-specific kernels need to be added."       >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created kernel file"                                                      >> todo_${varname}.txt
echo "ginkgo/reference/solver/${varname}_kernels.cpp"                           >> todo_${varname}.txt
echo "    Reference kernels for ${varname} need to be implemented here."        >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created kernel file"                                                      >> todo_${varname}.txt
echo "ginkgo/cpu/solver/${varname}_kernels.cpp"                                 >> todo_${varname}.txt
echo "    CPU kernels for ${varname} need to be implemented here."              >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created kernel file"                                                      >> todo_${varname}.txt
echo "ginkgo/gpu/solver/${varname}_kernels.cu"                                  >> todo_${varname}.txt
echo "    GPU kernels for ${varname} need to be implemented here."              >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created unit tests for ${varname} solver in:"                             >> todo_${varname}.txt
echo "ginkgo/core/test/solver/${varname}.cpp"                                   >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created unit tests for ${varname} reference kernels in"                   >> todo_${varname}.txt
echo "ginkgo/reference/test/solver/${varname}_kernels.cpp"                      >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created unit tests for ${varname} CPU kernels in"                         >> todo_${varname}.txt
echo "ginkgo/cpu/test/solver/${varname}_kernels.cpp"                            >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "Created unit tests for ${varname} GPU kernels in"                         >> todo_${varname}.txt
echo "ginkgo/gpu/test/solver/${varname}_kernels.cpp"                            >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "All tests have to be modified to the specific solver characteristics."    >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "The following CMakeLists have to be modified manually:"                   >> todo_${varname}.txt
echo "core/CMakeLists.txt"                                                      >> todo_${varname}.txt
echo "core/test/solver/CMakeLists.txt"                                          >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "reference/CMakeLists.txt"                                                 >> todo_${varname}.txt
echo "reference/test/solver/CMakeLists.txt"                                     >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "cpu/CMakeLists.txt"                                                       >> todo_${varname}.txt
echo "cpu/test/solver/CMakeLists.txt"                                           >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "gpu/CMakeLists.txt"                                                       >> todo_${varname}.txt
echo "gpu/test/solver/CMakeLists.txt"                                           >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "The following header file has to modified:"                               >> todo_${varname}.txt
echo "core/device_hooks/common_kernels.inc.cpp"                                 >> todo_${varname}.txt
echo "Equivalent to the other solvers, the following part has to be appended:"  >> todo_${varname}.txt
echo "#######################################################################"  >> todo_${varname}.txt
echo "namespace  ${varname} {"                                                  >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "// template <typename ValueType>"                                         >> todo_${varname}.txt
echo "// GKO_DECLARE_${varname3}_INITIALIZE_KERNEL(ValueType)"                  >> todo_${varname}.txt
echo "// NOT_COMPILED(GKO_HOOK_MODULE);"                                        >> todo_${varname}.txt
echo "// GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_${varname3}_INITIALIZE_KERNEL);" >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "// ..."                                                                   >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo "}  // namespace ${varname}"                                               >> todo_${varname}.txt
echo "#######################################################################"  >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
echo ""                                                                         >> todo_${varname}.txt
