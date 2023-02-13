#!/bin/bash


RESULT_FOLDER=$1
MG_MODE=$2
output="${RESULT_FOLDER}/heatmap_${MG_MODE}.csv"
echo "0, 1, 2, 3, 4, 5, 6, 7, 8, 9" > "${output}"
for (( single=1; single <= 9; single++ )); do
    line="${single}"
    for (( empty=1; empty < ${single}; empty++ )); do
        line="${line}, "
    done
    for (( half=${single}; half <= 9; half++ )); do
        file=${RESULT_FOLDER}/beh_${MG_MODE}_${single}_${half}.txt
        # residual_norm
        residual_norm=$(tail -n 5 ${file} | head -n 1)
        info=$(tail -n 4 ${file} | sed -E 's/[^0-9\.]//g' | tr '\n' ',' | sed -E 's/,$/\n/g')
        # iteration count, generation time[ms], total execution time[ms], executation time per iteration[ms]
        IFS=',' read -r -a array <<< "$info"
        line="${line}, ${array[0]}"
    done
    echo "${line}" >> "${output}"
done


