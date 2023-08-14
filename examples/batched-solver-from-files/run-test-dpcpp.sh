#!/bin/bash -l

declare -a SOLVERS=("bicgstab")
declare -a INPUT_CASES=("dodecane_lu" "drm19" "gri12" "gri30" "isooctane")
declare -a NUM_UNIQUE_BATCH=("78" "67" "73" "90" "72")
declare -a BATCH_SIZES=("8192" "16384" "32768" "65536" "131072" "262144")
declare -a NUM_TILES_LIST=("2" "1")

EXEC="dpcpp"
VER="pvc"

BIN_PREFIX_PATH="../build/examples/batched-solver-from-files"
OUTPUT_PATH="performance"
DIR="${OUTPUT_PATH}/${VER}"
mkdir -p $DIR

#Warm up if submitting as a job
tmp=$(${BIN_PREFIX_PATH}/batched-solver-from-files $EXEC gri12 73 10000 bicgstab time no_resid)


for NUM_TILES in "${NUM_TILES_LIST[@]}"
do
  if [ $NUM_TILES -eq 1 ];  then
    export ZE_AFFINITY_MASK=0.0
  else
    unset ZE_AFFINITY_MASK
  fi

  for SOLVER in "${SOLVERS[@]}"
  do
    FNAME="$DIR/${EXEC}_${NUM_TILES}t_applinput_${SOLVER}.txt"

    echo -e "Writing runtime data into $FNAME ..."
    echo -e "input_case ${BATCH_SIZES[*]}" | tr " " "\t" >> $FNAME

    for (( i=0; i<${#INPUT_CASES[@]}; i++ ))
    do
      INPUT="${INPUT_CASES[$i]}"
      NUM_UBATCH="${NUM_UNIQUE_BATCH[$i]}"
      write_buffer=("${INPUT}")
      for b in "${BATCH_SIZES[@]}"
      do
        NUM_REPL=$(( ${b} / ${NUM_UBATCH} ))
        echo "Running ./batched-solver-from-files $EXEC $INPUT $NUM_UBATCH $NUM_REPL $SOLVER time no_resid"
        runtime=$(${BIN_PREFIX_PATH}/batched-solver-from-files $EXEC $INPUT $NUM_UBATCH $NUM_REPL $SOLVER time no_resid)
        write_buffer+=("$runtime")
      done
      echo -e "${write_buffer[*]}" | tr " " "\t" >> $FNAME
    done
  done
done
