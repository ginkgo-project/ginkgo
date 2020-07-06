#!/usr/bin/env bash

# Important script variable
execute=1
automatic_additions=1
TMPDIR="./tmp_$(date +%s)"
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )
GINKGO_ROOT_DIR="${THIS_DIR}/../.."

function print_help {
    echo -e "Usage: $0 [options] NewAlgorithm ModelType ModelName"
    echo -e "\tOptions:"
    echo -e "\t--dry-run: does a dry run"
    echo -e "\t--help: prints this help"
    echo -e "\t--list: prints the list of possible models"
    echo -e "\t--no-automatic-additions: do not automatically add generated files to CMakeFiles or common_kernels.inc.cpp file"
    echo -e ""
    echo -e "This script generates a new solver or preconditioner from an"
    echo -e "existing model identified thanks to the --list argument."
    echo -e "After the correct execution of this script, it is expected that"
    echo -e "NewAlgorithm is integrated into Ginkgo and that you finish all todos."
}

function list_sources {
    for type in solver preconditioner matrix factorization
    do
        for i in $(ls $GINKGO_ROOT_DIR/core/$type/*.cpp)
        do
            echo "$type "$(basename "$i" | cut -d"." -f1)
        done
    done
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
            exit 1
            ;;
        --dry-run)
            execute=0
            echo -e "Doing a dry run."
            ;;
        --list)
            list_sources
            exit 1
            ;;
        --no-automatic-additions)
            automatic_additions=0
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

if [[ "$name" == "" ]] || ( [[ "$source_type" != "preconditioner" ]] && [[ "$source_type" != "matrix" ]] && [[ "$source_type" != "solver" ]] && [[ "$source_type" != "factorization" ]] ) || [[ "$source_name" == "" ]]; then
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
    "${name}_*.[ch]*"
    "${name}_kernels.hip.cpp"
    "${name}.cpp"
    "${name}_kernels.cpp"
    "${name}_kernels.cpp"
    "${name}_kernels.cpp"
    "${name}_kernels.*"
)
CMAKE_FILES=(
    "core/CMakeLists.txt"
    ""
    ""
    "reference/CMakeLists.txt"
    "omp/CMakeLists.txt"
    "cuda/CMakeLists.txt"
    "hip/CMakeLists.txt"
    "core/test/$source_type/CMakeLists.txt"
    "reference/test/$source_type/CMakeLists.txt"
    "omp/test/$source_type/CMakeLists.txt"
    "cuda/test/$source_type/CMakeLists.txt"
    "hip/test/$source_type/CMakeLists.txt"
)
TEMPLATE_FILES_LOCATIONS=(
    "core/$source_type"
    "include/ginkgo/core/$source_type"
    "core/$source_type"
    "reference/$source_type"
    "omp/$source_type"
    "cuda/$source_type"
    "hip/$source_type"
    "core/test/$source_type"
    "reference/test/$source_type"
    "omp/test/$source_type"
    "cuda/test/$source_type"
    "hip/test/$source_type"
)
TEMPLATE_FILES_TYPES=(
    "$source_type file"
    "class header"
    "kernel header"
    "Reference kernel file"
    "OpenMP kernel file"
    "CUDA kernel file"
    "HIP kernel file"
    "unit tests for ${name} $source_type"
    "unit tests for ${name} reference kernels"
    "unit tests for ${name} OMP kernels"
    "unit tests for ${name} CUDA kernels"
    "unit tests for ${name} HIP kernels"
)
TEMPLATE_FILES_DESCRIPTIONS=(
    "This is where the ${name} algorithm needs to be implemented."
    "This is where the ${name} class functions need to be implemented."
    "This is where the algorithm-specific kernels need to be added."
    "Reference kernels for ${name} need to be implemented here."
    "OMP kernels for ${name} need to be implemented here."
    "CUDA kernels for ${name} need to be implemented here."
    "HIP kernels for ${name} need to be implemented here."
    "This is where core related unit tests should be implemented, i.e. relating to the interface without executor usage."
    "This is where tests with the Reference executor should be implemented. Usually, this means comparing against previously known values."
    "This is where tests with the OpenMP executor should be implemented. Usually, this means comparing against a Reference execution."
    "This is where tests with the CUDA executor should be implemented. Usually, this means comparing against a Reference execution."
    "This is where tests with the HIP executor should be implemented. Usually, this means comparing against a Reference execution."
)

mkdir ${TMPDIR}

for (( i=1; i<${#TEMPLATE_FILES[@]}+1; i++ ))
do
    sourcename=$(echo ${TEMPLATE_FILES[$i-1]} | sed "s/${name}/${source_name}/" )
    sourcepath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${sourcename}

    # create folder for temporary files
    mkdir -p ${TMPDIR}/${TEMPLATE_FILES_LOCATIONS[$i-1]}

    # Evaluate the extension and try to find the matching files
    for j in $(ls ${GINKGO_ROOT_DIR}/${sourcepath})
    do
        if [ -f "$j" ]
        then
            filename=$(basename -- ${j})
            source_path=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${filename}
            destname=$(echo "${filename}" | sed "s/${source_name}/${name}/")
            destpath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/$destname

            cp ${GINKGO_ROOT_DIR}/$source_path ${TMPDIR}/$destpath

            # Replace all instances of source_name by the user's requested name
            perl -n -i -e "print unless m/.*common.*${source_name}_kernels.hpp.inc.*/" ${TMPDIR}/$destpath
            perl -pi -e "s/${source_name}/$name/g" ${TMPDIR}/$destpath
            perl -pi -e "s/${source_name^}/$Name/g" ${TMPDIR}/$destpath
            perl -pi -e "s/${source_name^^}/$NAME/g" ${TMPDIR}/$destpath

            # Comment all code
            awk -v name=${name} '/^{$/,/^}$/ { if ($0 == "{"){ print "GKO_NOT_IMPLEMENTED;"; print "//" $0; print "// TODO (script:" name "): change the code imported from '${source_type}'/'${source_name}' if needed"; next} else { print "//" $0; next }} 1' ${TMPDIR}/$destpath > tmp
            mv tmp ${TMPDIR}/$destpath

            ls ${TMPDIR}/$destpath

            if [ $execute == 1 ]
            then
                if [ ! -f ${GINKGO_ROOT_DIR}/$destpath ]; then
                    cp ${TMPDIR}/${destpath} ${GINKGO_ROOT_DIR}/${destpath}
                else
                    echo -e "Error: file ${GINKGO_ROOT_DIR}/$destpath exists"
                    echo -e "Remove file first if you want to replace it."
                    read -p ""
                fi
            fi
        else
            echo "Warning: Source file $sourcepath was not found."
        fi
    done
done

if [ $execute == 1 ]
then
    if [ $automatic_additions -eq 1 ]
    then
        ## Try to automatically add the files to CMakeLists
        echo -e "Modifying CMakeLists.txt and common_kernels.inc.cpp"
        for ((i=1; i<=${#CMAKE_FILES[@]}; i++))
        do
            sourcepath=${TEMPLATE_FILES_LOCATIONS[$i-1]}/${TEMPLATE_FILES[$i-1]}
            for j in $(ls ${GINKGO_ROOT_DIR}/${sourcepath})
            do
                filename=$(basename -- $j)
                shortname=$(echo $filename | cut -d"." -f1)
                sourcename=$(echo ${shortname} | sed "s/${name}/${source_name}/" )
                if [[ ! -f ${j} || "${j}" == *".hpp" || "${j}" == *".cuh" ]];
                then
                    continue
                fi

                cmake_file="${GINKGO_ROOT_DIR}/${CMAKE_FILES[$i-1]}"
                if [[ $cmake_file == *"test/"* ]]
                then
                    insert=$(grep -E "\(${sourcename}[_\)]{1}" $cmake_file | sed "s/$source_name/$name/")
                    echo "$insert" >> $cmake_file
                    cat $cmake_file | sort > tmp
                    mv tmp $cmake_file
                elif [[ $cmake_file != "${GINKGO_ROOT_DIR}/" ]]
                then
                    ## For most directories this works with something of the form:
                    ##target_sources(
                    ##     PRIVATE
                    ##     <lib1>
                    ##     ...
                    ##     <libn>)
                    ## For HIP:
                    ##set(GINKGO_HIP_SOURCES
                    ##    <lib1>
                    ##    ...
                    ##    <libn>)
                    if [[ $cmake_file == *"hip/"* ]]
                    then
                        list=( $(awk '/^set\(GINKGO_HIP_SOURCES/,/    .*\)/ {if ( match($0, "GINKGO_HIP_SOURCES") == 0 ) { print $0 }}' $cmake_file) )
                    else
                        list=( $(awk '/^target_sources/,/    .*\)/ {if ( match($0, "target_sources") == 0 && match($0, "PRIVATE") == 0 ) { print $0 }}' $cmake_file) )
                    fi

                    last_elem=$((${#list[@]}-1))
                    list[$last_elem]=$(echo ${list[$last_elem]} | tr -d ')')
                    list+=( "$source_type/${filename}" )
                    IFS=$'\n' sorted=($(sort <<<"${list[*]}"))
                    unset IFS
                    last_elem=$((${#sorted[@]}-1))
                    sorted[$last_elem]=$(echo ${sorted[$last_elem]}")")

                    ## find the correct position and clear up the CMakeList.txt
                    if [[ $cmake_file == *"hip/"* ]]
                    then
                        insert_to=$(grep -n -m 1 "GINKGO_HIP_SOURCES" $cmake_file | sed 's/:.*//')
                        awk '/^set\(GINKGO_HIP_SOURCES/,/    .*\)/ {if (match($0, "GINKGO_HIP_SOURCES") != 0 ){ print $0 }; next}1'  $cmake_file > tmp
                    else
                        insert_to=$(grep -n -m 1 "target_sources" $cmake_file | sed 's/:.*//')
                        insert_to=$((insert_to + 1)) # account for the "PRIVATE"
                        awk '/^target_sources/,/    .*\)/ {if (match($0, "target_sources") != 0 || match($0, "PRIVATE") != 0){ print $0 }; next}1'  $cmake_file > tmp
                    fi

                    mytmp=`mktemp`
                    head -n$insert_to tmp > $mytmp
                    for line in "${sorted[@]}"
                    do
                        echo "    $line" >> $mytmp
                    done
                    tail -n +$((insert_to+1)) tmp >> $mytmp
                    mv $mytmp tmp
                    mv tmp $cmake_file
                fi
            done
        done


        ## Automatically duplicate the entry of common_kernels.inc.cpp
        common_kernels_file="${GINKGO_ROOT_DIR}/core/device_hooks/common_kernels.inc.cpp"
        # add the new kernel file to headers
        headers=( "$(grep '#include \"' $common_kernels_file)" )
        headers+=( "#include \"core/${source_type}/${name}_kernels.hpp\"")
        IFS=$'\n' headers_sorted=($(sort <<<"${headers[*]}"))
        unset IFS
        header_block_begin=$(grep -n "#include \"" $common_kernels_file | head -n1 | sed 's/:.*//')
        grep -v '#include "' $common_kernels_file > tmp

        mytmp=`mktemp`
        head -n$((header_block_begin-1)) tmp > $mytmp
        for line in "${headers_sorted[@]}"
        do
            echo "$line" >> $mytmp
        done
        tail -n +$((header_block_begin)) tmp >> $mytmp
        mv $mytmp tmp
        mv tmp $common_kernels_file

        ## Automatically duplicate common_kernels.inc.cpp code block
        IFS=$'\n'
        GLOB_IGNORE='*'
        old_code_block=( "$(awk '/namespace '${source_name}' {/,/}  \/\/ namespace '${source_name}'/ {print $0}' $common_kernels_file)" )
        # code_block=( $(echo "${code_block[@]}" ) )
        unset IFS
        unset GLOB_IGNORE
        old_code_block_end=$(grep -ne "}  // namespace ${source_name}$" $common_kernels_file | sed 's/:.*//')

        mytmp=`mktemp`
        head -n$old_code_block_end $common_kernels_file > $mytmp
        echo -e "\n\n// TODO (script:${name}): adapt this block as needed" >> $mytmp
        for line in "${old_code_block[@]}"
        do
            echo -e "$line" | sed "s/${source_name^^}/$NAME/g" | sed "s/${source_name}/$name/g" >> $mytmp
        done
        tail -n +$((old_code_block_end+1)) $common_kernels_file >> $mytmp
        mv $mytmp $common_kernels_file
    fi

    echo -e "cleaning up temporary files."
    rm -rf ${TMPDIR}
else
    echo -e "\nNo file was copied because --dry-run was used"
    echo -e "You can inspect the generated solver files in ${TMPDIR}."
fi

if [ -f todo_${name}.txt ]; then
    rm todo_${name}.txt
fi

echo -e "\n###Summary:"                                                                 | tee -a todo_${name}.txt
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

if [ $automatic_additions -eq 1 ]
then
    for (( i=1; i<${#CMAKE_FILES[@]}+1; i++ ))
    do
        if [[ "${CMAKE_FILES[$i-1]}" != "" ]]
        then
            echo "Modified ${CMAKE_FILES[$i-1]}"              | tee -a todo_${name}.txt
        fi
    done
    echo "Modified core/device_hooks/common_kernels.inc.cpp"  | tee -a todo_${name}.txt
fi

if [ $automatic_additions -eq 0 ]
then
    echo ""                                                   | tee -a todo_${name}.txt
    echo "The following CMakeLists have to be modified manually:"| tee -a todo_${name}.txt
    echo "core/CMakeLists.txt"                                | tee -a todo_${name}.txt
    echo "core/test/${source_type}/CMakeLists.txt"            | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo "reference/CMakeLists.txt"                           | tee -a todo_${name}.txt
    echo "reference/test/${source_type}/CMakeLists.txt"       | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo "omp/CMakeLists.txt"                                 | tee -a todo_${name}.txt
    echo "omp/test/${source_type}/CMakeLists.txt"             | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo "cuda/CMakeLists.txt"                                | tee -a todo_${name}.txt
    echo "cuda/test/${source_type}/CMakeLists.txt"            | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo "hip/CMakeLists.txt"                                 | tee -a todo_${name}.txt
    echo "hip/test/${source_type}/CMakeLists.txt"             | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo "The following header file has to be modified:"      | tee -a todo_${name}.txt
    echo "core/device_hooks/common_kernels.inc.cpp"           | tee -a todo_${name}.txt
    echo "Equivalent to the other solvers, the following part has to be appended:"  | tee -a todo_${name}.txt
    echo "##################################################" | tee -a todo_${name}.txt
    echo "#include #include \"core/solver/test_kernels.hpp\"" | tee -a todo_${name}.txt
    echo "// ..."                                             | tee -a todo_${name}.txt
    echo "namespace  ${name} {"                               | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo "// template <typename ValueType>"                   | tee -a todo_${name}.txt
    echo "// GKO_DECLARE_${NAME}_INITIALIZE_KERNEL(ValueType)"| tee -a todo_${name}.txt
    echo "// GKO_NOT_COMPILED(GKO_HOOK_MODULE);"                  | tee -a todo_${name}.txt
    echo "// GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_${NAME}_INITIALIZE_KERNEL);" | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo "// ..."                                             | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo "}  // namespace ${name}"                            | tee -a todo_${name}.txt
    echo "##################################################" | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
    echo ""                                                   | tee -a todo_${name}.txt
fi

echo -e "\n\n\n### TODO:"                                         | tee -a todo_${name}.txt
echo -e "In all of the previous files ${source_name} was automatically replaced into ${name}. Ensure there is no inconsistency."                              | tee -a todo_${name}.txt
echo -e ""                                                    | tee -a todo_${name}.txt
echo -e "All the imported code was commented and TODO items were generated in the new files." | tee -a todo_${name}.txt
echo -e "Check all the modified files for \"// TODO (script:${name}):\" items"| tee -a todo_${name}.txt
echo -e "e.g. by using  grep -nR \"// TODO (script:${name}):\" ${GINKGO_ROOT_DIR} | grep -v \"create_new_algorithm.sh\" | grep -v \"todo_${name}.txt\"." | tee -a todo_${name}.txt
echo ""                                                       | tee -a todo_${name}.txt
echo "A tentative list of relevant TODO items follows:"       | tee -a todo_${name}.txt
grep -nR "// TODO (script:${name}):" ${GINKGO_ROOT_DIR} | grep -v "create_new_algorithm.sh" | grep -v "todo_${name}.txt" | tee -a todo_${name}.txt


echo "A summary of the required next steps has been written to:"
echo "todo_${name}.txt"
