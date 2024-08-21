#!/usr/bin/env bash

source="$1"
start="$2"
end="$3"
dest="$4"
# A_mg_*, A_mg_r_*, F_* .mtx

mkdir ${dest}
for ((i = ${start}; i < ${end}; i++ )); do
    new_index=$(( ${i} - ${start} ))
    echo "ln -s (A_mg/A_mg_r/F)_${i} to new (A_mg/A_mg_r/F)_${new_index}"
    ln -s ${source}/A_mg_${i}.mtx ${dest}/A_mg_${new_index}.mtx
    ln -s ${source}/A_mg_r_${i}.mtx ${dest}/A_mg_r_${new_index}.mtx
    ln -s ${source}/F_${i}.mtx ${dest}/F_${new_index}.mtx
done
new_index=$(( ${end} - ${start} )) 
echo "ln -s coarsest A_mg_${end} to new A_mg_${new_index}"
ln -s ${source}/A_mg_${end}.mtx ${dest}/A_mg_${new_index}.mtx
