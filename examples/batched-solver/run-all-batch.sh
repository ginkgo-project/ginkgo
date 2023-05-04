#!/bin/bash -l

#declare -a SOLVERS=("bicgstab" "cg" "direct" "gmres" "idr" "lower_trs" "upper_trs" "richardson")
#declare -a SOLVERS=("cg" "bicgstab" "gmres" "richardson")
declare -a SOLVERS=("cg")
declare -a BATCH_SIZES=("128" "256" "512" "1024" "2048" "4096" "8192" "16384" "32768" "65536" "131072" "262144")
#declare -a BATCH_SIZES=("32768" "65536" "131072" "262144")
#declare -a BATCH_SIZES=("131072")
declare -a MAT_SIZES=("16" "32" "64" "128" "256" "512" "1024")

NUM_TILES=2
EXEC="dpcpp"
VER="opt3"

BIN_PREFIX_PATH="${HOME}/ginkgo/build/examples/batched-solver"
OUTPUT_PATH="../performance"
DIR="${OUTPUT_PATH}/${VER}"
mkdir -p $DIR

if [ $NUM_TILES -eq 1 ];  then
    export ZE_AFFINITY_MASK=0.0
else
    unset ZE_AFFINITY_MASK
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

