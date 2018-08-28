EXECUTOR=$1

SSGET=ssget
NUM_PROBLEMS=$(${SSGET} -n)

# remove in production mode
NUM_PROBLEMS=2


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
    INPUT=$(${SSGET} -i $i -e)
    echo "[{\"filename\": \"${INPUT}\", \"problem\": $(${SSGET} -i $i -j)}]" \
        >"${RESULT_FILE}.imd"
    echo -e "($i/${NUM_PROBLEMS}):\tRunning SpMV for ${GROUP}/${NAME}" 1>&2
    ./spmv/spmv --executor=${EXECUTOR} \
                --backup="${RESULT_FILE}.bkp" \
                --double_buffer="${RESULT_FILE}.bkp2" \
                --formats="csr,coo,hybrid,sellp,ell" \
                <"${RESULT_FILE}.imd" 2>&1 >${RESULT_FILE}
    keep_latest ${RESULT_FILE} "${RESULT_FILE}.bkp" "${RESULT_FILE}.bkp2" \
                "${RESULT_FILE}.imd"

    if [ "${BENCHMARK}" != "solver" -o \
         "$(${SSGET} -i $i -prows)" = "$(${SSGET} -i $i -pcols)" ]; then
        ${SSGET} -i $i -c >/dev/null
        continue
    fi

    cp ${RESULT_FILE} "${RESULT_FILE}.imd"
    echo -e "($i/${NUM_PROBLEMS}):\tRunning Solvers for ${GROUP}/${NAME}" 1>&2
    ./solver/solver --executor=${EXECUTOR} \
        --backup="${RESULT_FILE}.bkp" \
        --double_buffer="${RESULT_FILE}.bkp2" \
        --max_iters=10000 --rel_res_goal=1e-6 \
        --solvers="cg,bicgstab,cgs,fcg" \
        --detailed \
        <"${RESULT_FILE}.imd" 2>&1 >${RESULT_FILE}
    keep_latest ${RESULT_FILE} "${RESULT_FILE}.bkp" "${RESULT_FILE}.bkp2" \
                "${RESULT_FILE}.imd"

    ${SSGET} -i $i -c >/dev/null
done
