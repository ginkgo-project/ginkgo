#!/bin/bash
FOLDER_PATH=.
FOLDER=`readlink -f ${FOLDER_PATH}`
URL=/to/workstation
EXECUTOR=$1
FORMATS=$2
SOLVERS=$3
SPMV_OUTPUT=$4
SOLVER_OUTPUT=$5
# spmv results are ${SPMV_OUTPUT}_${index}.json
# solver results are ${SOLVER_OUTPUT}_${index}.json
for i in {1..10}; do
    # Get data
    scp ${URL}:/data/suitesparse/realmtx_${i}.tar.gz ${FOLDER}
    tar zxvf ${FOLDER}/realmtx_${i}.tar.gz -C ${FOLDER}
    # Convert list to json file
    JSON=realmtx_${i}.json
    SRC=${FOLDER}/realmtx_${i}
    unset lastline
    echo "[" > ${JSON}
    while IFS='' read -r line || [[ -n "$line" ]]; do
        if ! [ -z "$lastline" ]; then
        echo "{ \"filename\": \"$SRC/$lastline\" }," >> ${JSON}
        fi
        lastline=$line
    done < ${SRC}/realmtx_${i}.list
    echo "{ \"filename\": \"$SRC/$lastline\" }" >> ${JSON}
    echo "]" >> ${JSON}
    # SPMV < JSON > SPMV_OUTPUT
    ./spmv/spmv -executor=$EXECUTOR -formats=$FORMATS < ${JSON} > ${SPMV_OUTPUT}_${i}.json
    # SOLVER < SPMV_OUTPUT > SOLVER_OUTPUT
    ./solver/solver -executor=$EXECUTOR -solvers=$SOLVERS < ${SPMV_OUTPUT}_${i}.json > ${SOLVER_OUTPUT}_${i}.json
    # Remove data
    rm ${FOLDER}/realmtx_${i}.tar.gz
    rm -rf ${FOLDER}/realmtx_${i}
done