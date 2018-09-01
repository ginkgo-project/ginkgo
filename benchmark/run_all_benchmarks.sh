EXECUTOR=$1


SSGET=ssget
NUM_PROBLEMS=$(${SSGET} -n)


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


# Creates an input file for SpMV benchmark for $1-th problem in the collection
generate_benchmark_input() {
    INPUT=$(${SSGET} -i $i -e)
    cat << EOT
[{
    "filename": "${INPUT}",
    "problem": $(${SSGET} -i $i -j)
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


for (( i=1; i <= ${NUM_PROBLEMS}; ++i )); do
    if [ $(${SSGET} -i $i -preal) = "0" ]; then
        ${SSGET} -i $i -c >/dev/null
        continue
    fi
    RESULT_DIR="results/${SYSTEM_NAME}/${EXECUTOR}/SuiteSparse"
    GROUP=$(${SSGET} -i $i -pgroup)
    NAME=$(${SSGET} -i $i -pname)
    RESULT_FILE="${RESULT_DIR}/${GROUP}/${NAME}.json"
    mkdir -p $(dirname ${RESULT_FILE})
    generate_benchmark_input $i >${RESULT_FILE}

    echo -e \
        "($i/${NUM_PROBLEMS}):\tExtracting statistics for ${GROUP}/${NAME}" 1>&2
    compute_matrix_statistics ${RESULT_FILE}

    echo -e "($i/${NUM_PROBLEMS}):\tRunning SpMV for ${GROUP}/${NAME}" 1>&2
    run_spmv_benchmarks ${RESULT_FILE}

    if [ "${BENCHMARK}" != "solver" -o \
         "$(${SSGET} -i $i -prows)" = "$(${SSGET} -i $i -pcols)" ]; then
        ${SSGET} -i $i -c >/dev/null
        continue
    fi

    echo -e "($i/${NUM_PROBLEMS}):\tRunning Solvers for ${GROUP}/${NAME}" 1>&2
    run_solver_benchmarks ${RESULT_FILE}

    ${SSGET} -i $i -c >/dev/null
done
