#!/bin/bash -l

declare -a BATCH_SIZES=("16" "32" "64" "128" "256" "512" "1024" "2048" "4096" "8192" "16384" "32768" "65536" "131072")
declare -a MAT_SIZES=("16" "32" "64" "128" "256" "512" "1024" "2048")

EXEC=$1
SOLVER=$2
FNAME="${EXEC}_batch_${SOLVER}.csv"

echo "batch_size,nrows,time" > $FNAME

for b in "${BATCH_SIZES[@]}" 
do
	for m in "${MAT_SIZES[@]}"
	do
		echo "Running ./batched-solver $EXEC $b $m $SOLVER time no_resid>> $FNAME"
		echo "$b,$m,$(./batched-solver $EXEC $b $m $SOLVER time no_resid)" >> $FNAME
	done
done

