################################################################################
# Environment variable detection

if [ ! "${BENCHMARK}" ]; then
    echo "BENCHMARK   environment variable not set - assuming \"spmv\"" 1>&2
    BENCHMARK="spmv"
fi

if [ ! "${DRY_RUN}" ]; then
    echo "DRY_RUN     environment variable not set - assuming \"false\"" 1>&2
    DRY_RUN="false"
fi

if [ ! "${EXECUTOR}" ]; then
    echo "EXECUTOR    environment variable not set - assuming \"cuda\"" 1>&2
    EXECUTOR="cuda"
fi

if [ ! "${SEGMENTS}" ]; then
    echo "SEGMENTS    environment variable not set - running entire suite" 1>&2
    SEGMENTS=1
    SEGMENT_ID=1
elif [ ! "${SEGMENT_ID}" ]; then
    echo "SEGMENT_ID  environment variable not set - exiting" 1>&2
    exit 1
fi

if [ ! "${PRECONDS}" ]; then
    echo "PRECONDS    environment variable not set - assuming \"none\"" 1>&2
    PRECONDS="none"
fi

if [ ! "${SYSTEM_NAME}" ]; then
    echo "SYSTEM_MANE environment variable not set - assuming \"unknown\"" 1>&2
    SYSTEM_NAME="unknown"
fi

if [ ! "${DEVICE_ID}" ]; then
    echo "DEVICE_ID environment variable not set - assuming \"0\"" 1>&2
    DEVICE_ID="0"
fi


################################################################################
# Utilities

# Checks the creation dates of a set of files $@, replaces file $1
# with the newest file from the set, and deletes the other
# files.
keep_latest() {
    RESULT="$1"
    for file in $@; do
        if [ "${file}" -nt "${RESULT}" ]; then
            RESULT="${file}"
        fi
    done
    if [ "${RESULT}" != "$1" ]; then
        cp "${RESULT}" "$1"
    fi
    for file in ${@:2}; do
        rm -f "${file}"
    done
}


# Computes matrix statistics of the problems described in file $1, and updates
# the file with results. Backups are created after each processed problem to
# prevent data loss in case of a crash. Once the extraction is completed, the
# backups and the results are combined, and the newest file is taken as the
# final result.
compute_matrix_statistics() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.imd" # make sure we're not loosing the original input
    ./matrix_statistics/matrix_statistics \
        --backup="$1.bkp" --double_buffer="$1.bkp2" \
        <"$1.imd" 2>&1 >"$1"
    keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the conversion benchmarks for all matrix formats by using file $1 as the
# input, and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_conversion_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.imd" # make sure we're not loosing the original input
    ./conversions/conversions --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --formats="csr,coo,hybrid,sellp,ell" \
                --device_id="${DEVICE_ID}" \
                <"$1.imd" 2>&1 >"$1"
    keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the SpMV benchmarks for all SpMV formats by using file $1 as the input,
# and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_spmv_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.imd" # make sure we're not loosing the original input
    ./spmv/spmv --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --formats="csr,coo,hybrid,sellp,ell" \
                --device_id="${DEVICE_ID}" \
                <"$1.imd" 2>&1 >"$1"
    keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
}


# Runs the solver benchmarks for all supported solvers by using file $1 as the
# input, and updating it with the results. Backups are created after each
# benchmark run, to prevent data loss in case of a crash. Once the benchmarking
# is completed, the backups and the results are combined, and the newest file is
# taken as the final result.
run_solver_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.imd" # make sure we're not loosing the original input
    ./solver/solver --backup="$1.bkp" --double_buffer="$1.bkp2" \
                    --executor="${EXECUTOR}" --solvers="$2" \
                    --preconditioners="${PRECONDS}" \
                    --max_iters=10000 --rel_res_goal=1e-10 \
                    --detailed=false \
                    --device_id="${DEVICE_ID}" \
                    <"$1.imd" 2>&1 >"$1"
    keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
}

# A list of block sizes that should be run for the block-Jacobi preconditioner
BLOCK_SIZES="$(seq 1 32)"
# A lis of precision reductions to run the block-Jacobi preconditioner for
PRECISIONS="0,0 0,1 0,2 1,0 1,1 2,0 autodetect"
# Runs the preconditioner benchmarks for all supported preconditioners by using
# file $1 as the input, and updating it with the results. Backups are created
# after each benchmark run, to prevent data loss in case of a crash. Once the
# benchmarking is completed, the backups and the results are combined, and the
# newest file is taken as the final result.
run_preconditioner_benchmarks() {
    [ "${DRY_RUN}" == "true" ] && return
    local bsize
    for bsize in ${BLOCK_SIZES}; do
        for prec in ${PRECISIONS}; do
            echo -e "\t\t running jacobi ($prec) for block size ${bsize}" 1>&2
            cp "$1" "$1.imd" # make sure we're not loosing the original input
            ./preconditioner/preconditioner \
                --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --preconditioners="jacobi" \
                --max_block_size="${bsize}" \
                --storage_optimization="${prec}" \
                --device_id="${DEVICE_ID}" \
                <"$1.imd" 2>&1 >"$1"
            keep_latest "$1" "$1.bkp" "$1.bkp2" "$1.imd"
        done
    done
}


################################################################################
# SuiteSparse collection

SSGET=ssget
NUM_PROBLEMS="$(${SSGET} -n)"

# Creates an input file for $1-th problem in the SuiteSparse collection
generate_suite_sparse_input() {
    INPUT=$(${SSGET} -i "$1" -e)
    cat << EOT
[{
    "filename": "${INPUT}",
    "problem": $(${SSGET} -i "$1" -j)
}]
EOT
}

for i in $(seq 1 $(ssget -n))
do
    NNZ=$(${SSGET} -p nonzeros -i $i)
    COLS=$(${SSGET} -p cols -i $i)
    if [ "$COLS" -lt 10000000 -a "$NNZ" -lt 500000000 ]
    then
        echo "$i"
    fi
done > matrix_list.txt

NUM_PROBLEMS="$(cat matrix_list.txt | wc -l)"

LOOP_START=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID} - 1) / ${SEGMENTS}))
LOOP_END=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID}) / ${SEGMENTS}))
for (( p=${LOOP_START}; p < ${LOOP_END}; ++p )); do
    i=$(sed -n "${p},${p}p" matrix_list.txt)
    if [ "${BENCHMARK}" == "preconditioner" ]; then
        break
    fi
    if [ "$(${SSGET} -i "$i" -preal)" = "0" ]; then
        [ "${DRY_RUN}" != "true" ] && ${SSGET} -i "$i" -c >/dev/null
        continue
    fi
    RESULT_DIR="results/${SYSTEM_NAME}/${EXECUTOR}/SuiteSparse"
    GROUP=$(${SSGET} -i "$i" -pgroup)
    NAME=$(${SSGET} -i "$i" -pname)
    RESULT_FILE="${RESULT_DIR}/${GROUP}/${NAME}.json"
    PREFIX="($i/${NUM_PROBLEMS}):\t"
    mkdir -p "$(dirname "${RESULT_FILE}")"
    generate_suite_sparse_input "$i" >"${RESULT_FILE}"

    echo -e "${PREFIX}Extracting statistics for ${GROUP}/${NAME}" 1>&2
    compute_matrix_statistics "${RESULT_FILE}"

    echo -e "${PREFIX}Running SpMV for ${GROUP}/${NAME}" 1>&2

    run_spmv_benchmarks "${RESULT_FILE}"

    if [ "${BENCHMARK}" == "conversions" ]; then
        echo -e "${PREFIX}Running Conversion for ${GROUP}/${NAME}" 1>&2
        run_conversion_benchmarks "${RESULT_FILE}"
    fi

    if [ "${BENCHMARK}" != "solver" -o \
         "$(${SSGET} -i "$i" -prows)" != "$(${SSGET} -i "$i" -pcols)" ]; then
        [ "${DRY_RUN}" != "true" ] && ${SSGET} -i "$i" -c >/dev/null
        continue
    fi

    POSDEF=$(${SSGET} -p posdef -i $i)
    NSYM=$(${SSGET} -p nsym -i $i)
    SOLVERS="bicgstab,cgs"
    if [ "$POSDEF" -eq 1 -a "${NSYM}" -eq 1 ]
    then
        SOLVERS="cg,${SOLVERS}"
    fi

    echo -e "${PREFIX}Running solvers for ${GROUP}/${NAME}" 1>&2
    run_solver_benchmarks "${RESULT_FILE}" "${SOLVERS}"

    echo -e "${PREFIX}Cleaning up problem ${GROUP}/${NAME}" 1>&2
    [ "${DRY_RUN}" != "true" ] && ${SSGET} -i "$i" -c >/dev/null
done

rm matrix_list.txt


if [ "${BENCHMARK}" != "preconditioner" ]; then
    exit
fi


################################################################################
# Generated collection

count_tokens() { echo "$#"; }

BLOCK_SIZES="$(seq 1 32)"
NUM_BLOCKS="$(seq 10000 2000 50000)"
NUM_PROBLEMS=$((
    $(count_tokens ${BLOCK_SIZES}) * $(count_tokens ${NUM_BLOCKS}) ))
ID=0


# Creates an input file for a block diagonal matrix with block size $1 and
# number of blocks $2. The location of the matrix is given by $3.
generate_block_diagonal_input() {
    cat << EOT
[{
    "filename": "$3",
    "problem": {
        "collection": "generated",
        "group": "block-diagonal",
        "name": "$2-$1",
        "type": "block-diagonal",
        "real": true,
        "binary": false,
        "2d3d": false,
        "posdef": false,
        "psym": 1,
        "nsym": 0,
        "kind": "artificially generated problem",
        "num_blocks": $2,
        "block_size": $1
    }
}]
EOT
}


# Generates the problem data using the input file $1
generate_problem() {
    [ "${DRY_RUN}" == "true" ] && return
    cp "$1" "$1.tmp"
    ./matrix_generator/matrix_generator <"$1.tmp" 2>&1 >"$1"
    keep_latest "$1" "$1.tmp"
}


LOOP_START=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID} - 1) / ${SEGMENTS}))
LOOP_END=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID}) / ${SEGMENTS}))
for bsize in ${BLOCK_SIZES}; do
    for nblocks in ${NUM_BLOCKS}; do
        ID=$((${ID} + 1))
        if [ "${ID}" -ge "${LOOP_END}" ]; then
            break
        fi
        if [ "${ID}" -lt "${LOOP_START}" ]; then
            continue
        fi
        RESULT_DIR="results/${SYSTEM_NAME}/${EXECUTOR}/Generated"
        GROUP="block-diagonal"
        NAME="${nblocks}-${bsize}"
        RESULT_FILE="${RESULT_DIR}/${GROUP}/${NAME}.json"
        PREFIX="(${ID}/${NUM_PROBLEMS}):\t"
        mkdir -p "$(dirname "${RESULT_FILE}")"
        mkdir -p "/tmp/${GROUP}"
        generate_block_diagonal_input \
            "${bsize}" "${nblocks}" "/tmp/${GROUP}/${NAME}.mtx" \
                >"${RESULT_FILE}"

        echo -e "${PREFIX}Generating problem ${GROUP}/${NAME}" 1>&2
        generate_problem "${RESULT_FILE}"

        echo -e "${PREFIX}Extracting statistics for ${GROUP}/${NAME}" 1>&2
        compute_matrix_statistics "${RESULT_FILE}"

        echo -e "${PREFIX}Running preconditioners for ${GROUP}/${NAME}" 1>&2
        BLOCK_SIZES="${bsize}"
        run_preconditioner_benchmarks "${RESULT_FILE}"

        echo -e "${PREFIX}Cleaning up problem ${GROUP}/${NAME}" 1>&2
        [ "${DRY_RUN}" != "true" ] && rm -r "/tmp/${GROUP}/${NAME}.mtx"
    done
    if [ "${ID}" -ge "${LOOP_END}" ]; then
        break
    fi
done
