EXECUTOR=$1

SSGET=ssget
NUM_PROBLEMS=$(${SSGET} -n)

# remove in production mode
NUM_PROBLEMS=2

for (( i=1; i <= ${NUM_PROBLEMS}; ++i )); do
    if [ $(${SSGET} -i $i -preal) = "0" ]; then
        ${SSGET} -i $i -c >/dev/null
        continue
    fi
    RESULT_DIR="results/cuda"
    RESULT_FILE="$(${SSGET} -i $i -pgroup)/$(${SSGET} -i $i -pname).json"
    mkdir -p "${RESULT_DIR}/$(dirname ${RESULT_FILE})"
    INPUT=$(${SSGET} -i $i -e)
    cat >"${RESULT_DIR}/${RESULT_FILE}" << EOT
[
    {
        "filename": "${INPUT}",
        "problem": $(${SSGET} -i $i -j)
    }
]
EOT
    cp "${RESULT_DIR}/${RESULT_FILE}" "${RESULT_DIR}/${RESULT_FILE}.imd"
    ./spmv/spmv --executor=${EXECUTOR} \
                --backup="${RESULT_DIR}/${RESULT_FILE}.bkp" \
                --double_buffer="${RESULT_DIR}/${RESULT_FILE}.bkp2" \
                --formats="coo,csr,ell,sellp,hybrid" \
                <"${RESULT_DIR}/${RESULT_FILE}.imd" >"${RESULT_DIR}/${RESULT_FILE}"

    if [ "${BENCHMARK}" != "solver" -o \
         "$(${SSGET} -i $i -prows)" = "$(${SSGET} -i $i -pcols)" ]; then
        ${SSGET} -i $i -c >/dev/null
        continue
    fi

    cp "${RESULT_DIR}/${RESULT_FILE}" "${RESULT_DIR}/${RESULT_FILE}.imd"
    ./solver/solver --executor=${EXECUTOR} \
        --backup="${RESULT_DIR}/${RESULT_FILE}.bkp" \
        --double_buffer="${RESULT_DIR}/${RESULT_FILE}.bkp2" \
        --max_iters=10000 --rel_res_goal=1e-6 \
        --solvers="cg,bicgstab,cgs,fcg" \
        --detailed \
        <"${RESULT_DIR}/${RESULT_FILE}.imd" \
        >"${RESULT_DIR}/${RESULT_FILE}"

    ${SSGET} -i $i -c >/dev/null
done
