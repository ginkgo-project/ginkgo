################################################################################
# Environment variable detection

if [ ! "${BENCHMARK}" ]; then
    BENCHMARK="spmv"
    echo "BENCHMARK   environment variable not set - assuming \"${BENCHMARK}\"" 1>&2
fi

if [ ! "${DRY_RUN}" ]; then
    DRY_RUN="false"
    echo "DRY_RUN     environment variable not set - assuming \"${DRY_RUN}\"" 1>&2
fi

if [ ! "${EXECUTOR}" ]; then
    EXECUTOR="cuda"
    echo "EXECUTOR    environment variable not set - assuming \"${EXECUTOR}\"" 1>&2
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
    PRECONDS="none"
    echo "PRECONDS    environment variable not set - assuming \"${PRECONDS}\"" 1>&2
fi

if [ ! "${FORMATS}" ]; then
    echo "FORMATS    environment variable not set - assuming \"csr,coo,ell,hybrid,sellp\"" 1>&2
    FORMATS="csr,coo,ell,hybrid,sellp"
fi

if [ ! "${SOLVERS}" ]; then
    SOLVERS="bicgstab,cg,cgs,fcg,gmres,cb_gmres_reduce1,idr"
    echo "SOLVERS    environment variable not set - assuming \"${SOLVERS}\"" 1>&2
fi

if [ ! "${SOLVERS_PRECISION}" ]; then
    SOLVERS_PRECISION=1e-6
    echo "SOLVERS_PRECISION environment variable not set - assuming \"${SOLVERS_PRECISION}\"" 1>&2
fi

if [ ! "${SOLVERS_MAX_ITERATIONS}" ]; then
    SOLVERS_MAX_ITERATIONS=10000
    echo "SOLVERS_MAX_ITERATIONS environment variable not set - assuming \"${SOLVERS_MAX_ITERATIONS}\"" 1>&2
fi

if [ ! "${SOLVERS_GMRES_RESTART}" ]; then
    SOLVERS_GMRES_RESTART=100
    echo "SOLVERS_GMRES_RESTART environment variable not set - assuming \"${SOLVERS_GMRES_RESTART}\"" 1>&2
fi

if [ ! "${SYSTEM_NAME}" ]; then
    SYSTEM_NAME="unknown"
    echo "SYSTEM_MANE environment variable not set - assuming \"${SYSTEM_NAME}\"" 1>&2
fi

if [ ! "${DEVICE_ID}" ]; then
    DEVICE_ID="0"
    echo "DEVICE_ID environment variable not set - assuming \"${DEVICE_ID}\"" 1>&2
fi

if [ ! "${SOLVERS_JACOBI_MAX_BS}" ]; then
    SOLVERS_JACOBI_MAX_BS="32"
    "SOLVERS_JACOBI_MAX_BS environment variable not set - assuming \"${SOLVERS_JACOBI_MAX_BS}\"" 1>&2
fi

if [ ! "${BENCHMARK_PRECISION}" ]; then
    BENCHMARK_PRECISION="double"
    echo "BENCHMARK_PRECISION not set - assuming \"${BENCHMARK_PRECISION}\"" 1>&2
fi

if [ "${BENCHMARK_PRECISION}" == "double" ]; then
    BENCH_SUFFIX=""
elif [ "${BENCHMARK_PRECISION}" == "single" ]; then
    BENCH_SUFFIX="_single"
elif [ "${BENCHMARK_PRECISION}" == "dcomplex" ]; then
    BENCH_SUFFIX="_dcomplex"
elif [ "${BENCHMARK_PRECISION}" == "scomplex" ]; then
    BENCH_SUFFIX="_scomplex"
else
    echo "BENCHMARK_PRECISION is set to the not supported \"${BENCHMARK_PRECISION}\"." 1>&2
    echo "Currently supported values: \"double\", \"single\", \"dcomplex\" and \"scomplex\"" 1>&2
    exit 1
fi

if [ ! "${SOLVERS_RHS}" ]; then
    SOLVERS_RHS="1"
    echo "SOLVERS_RHS environment variable not set - assuming \"${SOLVERS_RHS}\"" 1>&2
fi

if [ "${SOLVERS_RHS}" == "random" ]; then
    SOLVERS_RHS_FLAG="--rhs_generation=random"
elif [ "${SOLVERS_RHS}" == "1" ]; then
    SOLVERS_RHS_FLAG="--rhs_generation=1"
elif [ "${SOLVERS_RHS}" == "sinus" ]; then
    SOLVERS_RHS_FLAG="--rhs_generation=sinus"
else
    echo "SOLVERS_RHS does not support the value \"${SOLVERS_RHS}\"." 1>&2
    echo "The following values are supported: \"1\", \"random\" and \"sinus\"" 1>&2
    exit 1
fi

if [ ! "${SOLVERS_INITIAL_GUESS}" ]; then
    SOLVERS_INITIAL_GUESS="rhs"
    echo "SOLVERS_RHS environment variable not set - assuming \"${SOLVERS_INITIAL_GUESS}\"" 1>&2
fi

if [ "${SOLVERS_INITIAL_GUESS}" == "random" ]; then
    SOLVERS_INITIAL_GUESS_FLAG="--initial_guess_generation=random"
elif [ "${SOLVERS_INITIAL_GUESS}" == "0" ]; then
    SOLVERS_INITIAL_GUESS_FLAG="--initial_guess_generation=0"
elif [ "${SOLVERS_INITIAL_GUESS}" == "rhs" ]; then
    SOLVERS_INITIAL_GUESS_FLAG="--initial_guess_generation=rhs"
else
    echo "SOLVERS_RHS does not support the value \"${SOLVERS_RHS}\"." 1>&2
    echo "The following values are supported: \"0\", \"random\" and \"rhs\"" 1>&2
    exit 1
fi

if [ ! "${GPU_TIMER}" ]; then
    GPU_TIMER="false"
    echo "GPU_TIMER    environment variable not set - assuming \"${GPU_TIMER}\"" 1>&2
fi

# Control whether to run detailed benchmarks or not.
# Default setting is detailed=false. To activate, set DETAILED=1.
if  [ ! "${DETAILED}" ] || [ "${DETAILED}" -eq 0 ]; then
    DETAILED_STR="--detailed=false"
else
    DETAILED_STR="--detailed=true"
fi

# This allows using a matrix list file for benchmarking.
# The file should contains a suitesparse matrix on each line.
# The allowed formats to target suitesparse matrix is:
#   id or group/name or name.
# Example:
# 1903
# Freescale/circuit5M
# thermal2
if [ ! "${MATRIX_LIST_FILE}" ]; then
    use_matrix_list_file=0
elif [ -f "${MATRIX_LIST_FILE}" ]; then
    use_matrix_list_file=1
else
    echo -e "A matrix list file was set to ${MATRIX_LIST_FILE} but it cannot be found."
    exit 1
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
    ./matrix_statistics/matrix_statistics${BENCH_SUFFIX} \
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
    ./conversions/conversions${BENCH_SUFFIX} --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --formats="${FORMATS}" \
                --device_id="${DEVICE_ID}" --gpu_timer=${GPU_TIMER} \
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
    ./spmv/spmv${BENCH_SUFFIX} --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --formats="${FORMATS}" \
                --device_id="${DEVICE_ID}" --gpu_timer=${GPU_TIMER} \
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
    ./solver/solver${BENCH_SUFFIX} --backup="$1.bkp" --double_buffer="$1.bkp2" \
                    --executor="${EXECUTOR}" --solvers="${SOLVERS}" \
                    --preconditioners="${PRECONDS}" \
                    --max_iters=${SOLVERS_MAX_ITERATIONS} --rel_res_goal=${SOLVERS_PRECISION} \
                    ${SOLVERS_RHS_FLAG} ${DETAILED_STR} ${SOLVERS_INITIAL_GUESS_FLAG} \
                    --gpu_timer=${GPU_TIMER} \
                    --jacobi_max_block_size=${SOLVERS_JACOBI_MAX_BS} --device_id="${DEVICE_ID}" \
                    --gmres_restart="${SOLVERS_GMRES_RESTART}" \
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
            ./preconditioner/preconditioner${BENCH_SUFFIX} \
                --backup="$1.bkp" --double_buffer="$1.bkp2" \
                --executor="${EXECUTOR}" --preconditioners="jacobi" \
                --jacobi_max_block_size="${bsize}" \
                --jacobi_storage="${prec}" \
                --device_id="${DEVICE_ID}" --gpu_timer=${GPU_TIMER} \
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

parse_matrix_list() {
    local source_list_file=$1
    local benchmark_list=""
    local id=0
    for mtx in $(cat ${source_list_file}); do
        if [[ ! "$mtx" =~ ^[0-9]+$ ]]; then
            if [[ "$mtx" =~ ^[a-zA-Z0-9_-]+$ ]]; then
                id=$(${SSGET} -s "[ @name == $mtx ]")
            elif [[ "$mtx" =~ ^([a-zA-Z0-9_-]+)\/([a-zA-Z0-9_-]+)$ ]]; then
                local group="${BASH_REMATCH[1]}"
                local name="${BASH_REMATCH[2]}"
                id=$(${SSGET} -s "[ @name == $name ] && [ @group == $group ]")
            else
                >&2 echo -e "Could not recognize entry $mtx."
            fi
        else
            id=$mtx
        fi
        benchmark_list="$benchmark_list $id"
    done
    echo "$benchmark_list"
}

if [ $use_matrix_list_file -eq 1 ]; then
    MATRIX_LIST=($(parse_matrix_list $MATRIX_LIST_FILE))
    NUM_PROBLEMS=${#MATRIX_LIST[@]}
fi

LOOP_START=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID} - 1) / ${SEGMENTS}))
LOOP_END=$((1 + (${NUM_PROBLEMS}) * (${SEGMENT_ID}) / ${SEGMENTS}))
for (( p=${LOOP_START}; p < ${LOOP_END}; ++p )); do
    if [ $use_matrix_list_file -eq 1 ]; then
        i=${MATRIX_LIST[$((p-1))]}
    else
        i=$p
    fi
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

    echo -e "${PREFIX}Running solvers for ${GROUP}/${NAME}" 1>&2
    run_solver_benchmarks "${RESULT_FILE}"

    echo -e "${PREFIX}Cleaning up problem ${GROUP}/${NAME}" 1>&2
    [ "${DRY_RUN}" != "true" ] && ${SSGET} -i "$i" -c >/dev/null
done


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
    ./matrix_generator/matrix_generator${BENCH_SUFFIX} <"$1.tmp" 2>&1 >"$1"
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
