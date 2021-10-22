#!/bin/bash
#git rm log/*.txt
mkdir log_s

PATHa=../matrix/_lebo/big/
FULL=$PATHa$'*'

for f in $FULL
do
  NAME=${f#$PATHa}
  TXT2=$'log_s/'$NAME$'_lu.txt' 
  echo "Processing $f"
  ../demos_time -i $f > $TXT2
  echo "Write to $TXT2"
done
#nvprof --log-file 1.txt ../lu_cmd -i ../matrix/_lebo/big/1020_circuit204.mtx > 2.txt
#../lu_cmd -i ../data/circuit/7602_rajat03.mtx > log/1.txt
