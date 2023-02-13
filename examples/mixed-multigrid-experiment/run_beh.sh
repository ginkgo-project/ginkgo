#!/bin/bash


RESULT_FOLDER=$1
EXECUTOR=$2
MG_MODE=$3
MATRIX="$4"
B="$5"

for (( single=1; single <= 9; single++ )); do
    for (( half=${single}; half <= 9; half++ )); do
    ./mixed-multigrid-solver ${EXECUTOR} 4${single}${half} 10 v ${MG_MODE} ${MATRIX} ${B} > ${RESULT_FOLDER}/beh_${MG_MODE}_${single}_${half}.txt
    done
done


