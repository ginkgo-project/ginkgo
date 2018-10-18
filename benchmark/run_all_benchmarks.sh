EXECUTOR=$1


SSGET=ssget
NUM_PROBLEMS=$(${SSGET} -n)
NUM_PROBLEMS=10


# Checks the creation dates of a set of files $@, replaces file $1
# with the newest file from the set, and deletes the other
# files.
keep_latest() {
    RESULT=$1
    for file in $@; do
        if [ ${file} -nt ${RESULT} ]; then
            RESULT=${file}
        fi
    done
    if [ "${RESULT}" != "$1" ]; then
        cp ${RESULT} $1
    fi
    for file in ${@:2}; do
        rm -f ${file}
    done
}


# Creates an input file for $1-th problem in the SuiteSparse collection
generate_suite_sparse_input() {
    INPUT=$(${SSGET} -i $i -e)
    cat << EOT
[{
    "filename": "${INPUT}",
    "problem": $(${SSGET} -i $i -j)
}]
EOT
}


# Creates an input file for a block diagonal matrix with block size $1 and
# number of blocks $2. The location of the matrix is given by $3.
generate_block_diagonal_input() {
    cat << EOT
[{
    "filename": "$3",
    "problem": {
        "type": "block-diagonal",
        "num_blocks": $2,
        "block_size": $1
    }
}]
EOT
}


# Computes matrix statistics of the problems described in file $1, and updates
# the file with results. Backups are created after each processed problem to
# prevent data loss in case of a crash. Once the extraction is completed, the
# backups and the results are combined, and the newest file is taken as the
# final result.
compute_matrix_statistics() {
    cp $1 "$1.imd" # make sure we're not loosing the original input
    ./matrix_statistics/matrix_statistics \
        --backup="$1.bkp" --double_buffer="$1.bkp2" \
        <"$1.imd" 2>&1 >$1
    keep_latest $1 "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the SpMV benchmarks for all SpMV formats by using file $1 as the input,
# and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_spmv_benchmarks() {
    cp $1 "$1.imd" # make sure we're not loosing the original input
    ./spmv/spmv --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor=${EXECUTOR} --formats="csr,coo,hybrid,sellp,ell" \
                <"$1.imd" 2>&1 >$1
    keep_latest $1 "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the solver benchmarks for all supported solvers by using file $1 as the
# input, and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_solver_benchmarks() {
    cp $1 "$1.imd" # make sure we're not loosing the original input
    ./solver/solver --backup="$1.bkp" --double_buffer="$1.bkp2" \
                    --executor=${EXECUTOR} --solvers="cg,bicgstab,cgs,fcg" \
                    --max_iters=10000 --rel_res_goal=1e-6 \
                    <"$1.imd" 2>&1 >$1
    keep_latest $1 "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the preconditioner benchmarks for all supported preconditioners by using
# file $1 as the input, and updating it with the results. Backups are created
# after each benchmark run, to prevent data loss in case of a crash. Once the
# benchmarking is completed, the backups and the results are combined, and the
# newest file is taken as the final result.
run_preconditioner_benchmarks() {
    local bsize
    for bsize in {1..32}; do
        echo -e "\t\t running jacobi for block size ${bsize}/32" 1>&2
        cp $1 "$1.imd" # make sure we're not loosing the original input
        ./preconditioner/preconditioner \
            --backup="$1.bkp" --double_buffer="$1.bkp2" \
            --executor=${EXECUTOR} --preconditioners=jacobi \
            --max_block_size=${bsize} \
            <"$1.imd" 2>&1 >$1
        keep_latest $1 "$1.bkp" "$1.bkp2" "$1.imd"
    done
}


# SuiteSparse matrices
for (( i=1; i <= ${NUM_PROBLEMS}; ++i )); do
    if [ "${BENCHMARK}" == "preconditioner" ]; then
        break
    fi
    if [ $(${SSGET} -i $i -preal) = "0" ]; then
        ${SSGET} -i $i -c >/dev/null
        continue
    fi
    RESULT_DIR="results/${SYSTEM_NAME}/${EXECUTOR}/SuiteSparse"
    GROUP=$(${SSGET} -i $i -pgroup)
    NAME=$(${SSGET} -i $i -pname)
    RESULT_FILE="${RESULT_DIR}/${GROUP}/${NAME}.json"
    PREFIX="($i/${NUM_PROBLEMS}):\t"
    mkdir -p $(dirname ${RESULT_FILE})
    generate_suite_sparse_input $i >${RESULT_FILE}

    echo -e "${PREFIX}Extracting statistics for ${GROUP}/${NAME}" 1>&2
    compute_matrix_statistics ${RESULT_FILE}

    echo -e "${PREFIX}Running SpMV for ${GROUP}/${NAME}" 1>&2
    run_spmv_benchmarks ${RESULT_FILE}

    if [ "${BENCHMARK}" != "solver" -o \
         "$(${SSGET} -i $i -prows)" != "$(${SSGET} -i $i -pcols)" ]; then
        ${SSGET} -i $i -c >/dev/null
        continue
    fi

    echo -e "${PREFIX}Running solvers for ${GROUP}/${NAME}" 1>&2
    run_solver_benchmarks ${RESULT_FILE}

    echo -e "${PREFIX}Cleaning up problem ${GROUP}/${NAME}" 1>&2
    ${SSGET} -i $i -c >/dev/null
done


if [ "${BENCHMARK}" != "preconditioner" ]; then
    exit
fi


# block diagonal matrices
BLOCK_SIZES="$(seq 1 32)"
NUM_BLOCKS="$(seq 10000 2000 50000)"

count_tokens() { echo $#; }
NUM_PROBLEMS=$((
    $(count_tokens ${BLOCK_SIZES}) * $(count_tokens ${NUM_BLOCKS}) ))
ID=1

for bsize in ${BLOCK_SIZES}; do
    for nblocks in ${NUM_BLOCKS}; do
        RESULT_DIR="results/${SYSTEM_NAME}/${EXECUTOR}/Generated"
        GROUP="block-diagonal"
        NAME="${nblocks}-${bsize}"
        RESULT_FILE="${RESULT_DIR}/${GROUP}/${NAME}.json"
        PREFIX="(${ID}/${NUM_PROBLEMS}):\t"
        mkdir -p $(dirname ${RESULT_FILE})
        mkdir -p "/tmp/${GROUP}"
        generate_block_diagonal_input \
            ${bsize} ${nblocks} "/tmp/${GROUP}/${NAME}.mtx" >${RESULT_FILE}

        echo -e "${PREFIX}Generating problem ${GROUP}/${NAME}" 1>&2
        cp "${RESULT_FILE}" "${RESULT_FILE}.tmp"
        ./matrix_generator/matrix_generator <"${RESULT_FILE}.tmp" 2>&1 \
            >${RESULT_FILE}
        rm "${RESULT_FILE}.tmp"

        echo -e "${PREFIX}Extracting statistics for ${GROUP}/${NAME}" 1>&2
        compute_matrix_statistics ${RESULT_FILE}

        echo -e "${PREFIX}Running preconditioners for ${GROUP}/${NAME}" 1>&2
        run_preconditioner_benchmarks ${RESULT_FILE}

        echo -e "${PREFIX}Cleaning up problem ${GROUP}/${NAME}" 1>&2
        rm -r "/tmp/${GROUP}/${NAME}.mtx"
        ID=$(( ${ID} + 1 ))
    done
done
