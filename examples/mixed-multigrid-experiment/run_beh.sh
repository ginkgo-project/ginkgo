#!/bin/bash


RESULT_FOLDER=$1
EXECUTOR=$2
MG_MODE=$3
MATRIX="$4"
B="$5"
sm_mode="jacobi"
scale="0"
# all double
./mixed-multigrid-solver ${EXECUTOR} 0 10 v ${MG_MODE} ${sm_mode} ${scale} ${MATRIX} ${B} > ${RESULT_FOLDER}/beh_${MG_MODE}.txt
for (( single=1; single <= 9; single++ )); do
    # single 
    ./mixed-multigrid-solver ${EXECUTOR} 4${single} 10 v ${MG_MODE} ${MATRIX} ${B} > ${RESULT_FOLDER}/beh_${MG_MODE}_${single}.txt
    for (( half=${single}; half <= 9; half++ )); do
        # (single-)half
        ./mixed-multigrid-solver ${EXECUTOR} 4${single}${half} 10 v ${MG_MODE} ${MATRIX} ${B} > ${RESULT_FOLDER}/beh_${MG_MODE}_${single}_${half}.txt
    done
done


