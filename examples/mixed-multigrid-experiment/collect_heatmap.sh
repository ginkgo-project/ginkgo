#!/bin/bash


RESULT_FOLDER=$1
MG_MODE=$2
COLLECT=$3
get_output() {
    local file="$1"
    local collect="$2"
    # residual_norm
    residual_norm=$(tail -n 5 ${file} | head -n 1)
    info=$(tail -n 4 ${file} | sed -E 's/[^0-9\.]//g' | tr '\n' ',' | sed -E 's/,$/\n/g')
    # iteration count, generation time[ms], total execution time[ms], executation time per iteration[ms]
    IFS=',' read -r -a array <<< "$info"
    if [[ "${collect}" == "iter" ]]; then
        echo "${array[0]}"
    elif [[ "${collect}" == "time" ]]; then
        echo "${array[2]}"
    elif [[ "${collect}" == "res" ]]; then
        echo "${residual_norm}"
    fi
}
output="${RESULT_FOLDER}/heatmap_${MG_MODE}_${COLLECT}.csv"
echo "${COLLECT} , single |" > "${output}"
echo "half -> , 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, never" >> "${output}"
for (( single=1; single <= 9; single++ )); do
    line=" , ${single}"
    for (( empty=1; empty <= ${single}; empty++ )); do
        line="${line}, "
    done
    half_start=$(( ${single} + 1 ))
    for (( half=${half_start}; half <= 9; half++ )); do
        file=${RESULT_FOLDER}/beh_${MG_MODE}_${single}_${half}.txt
        line="${line}, $(get_output ${file} ${COLLECT})"
    done
    file=${RESULT_FOLDER}/beh_${MG_MODE}_${single}.txt
    line="${line}, $(get_output ${file} ${COLLECT})"
    echo "${line}" >> "${output}"
done
line=" , never"
for (( half=1; half <= 9; half++ )); do
    file=${RESULT_FOLDER}/beh_${MG_MODE}_${half}_${half}.txt
    line="${line}, $(get_output ${file} ${COLLECT})"
done
file=${RESULT_FOLDER}/beh_${MG_MODE}.txt
line="${line}, $(get_output ${file} ${COLLECT})"
echo "${line}" >> "${output}"


