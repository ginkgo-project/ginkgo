#!/bin/bash -l

#declare -a SOLVERS=("bicgstab" "cg" "direct" "gmres" "idr" "lower_trs" "upper_trs" "richardson")
#declare -a SOLVERS=("cg" "bicgstab" "gmres" "richardson")
declare -a SOLVERS=("cg")
#declare -a BATCH_SIZES=("50" "100" "500" "1000" "5000" "10000" "50000")
declare -a BATCH_SIZES=("1000" "5000" "10000" "50000" "100000")
declare -a MAT_SIZES=("16" "32" "64" "128" "256" "512" "768" "1024" "1280" "1536" "1792" "2048")

NUM_TILES=1
EXEC="dpcpp"
VER="opt"

BIN_PREFIX_PATH="${HOME}/ginkgo/build/examples/batched-solver"
OUTPUT_PATH="../performance"
DIR="${OUTPUT_PATH}/${VER}"
mkdir -p $DIR

if [ $NUM_TILES -eq 1 ];  then
    export ZE_AFFINITY_MASK=0.0
fi

for SOLVER in "${SOLVERS[@]}" 
do
    FNAME="$DIR/${EXEC}_${NUM_TILES}t_batch_${SOLVER}.txt"
    echo -e "Writing runtime data into $FNAME ..."
    echo -e "b\m ${MAT_SIZES[*]}" | tr " " "\t" > $FNAME
    for b in "${BATCH_SIZES[@]}" 
    do
        write_buffer=("${b}")
        for m in "${MAT_SIZES[@]}"
        do
            echo "Running ./batched-solver $EXEC $b $m $SOLVER time no_resid"
            runtime=$(${BIN_PREFIX_PATH}/batched-solver $EXEC $b $m $SOLVER time no_resid)
            write_buffer+=("$runtime")
        done
    echo -e "${write_buffer[*]}" | tr " " "\t" >> $FNAME
    done
done

