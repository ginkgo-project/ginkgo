#!/usr/bin/env bash


problem="$1"
levels="$2"
result_folder="$3"

export CUDA_VISIBLE_DEVICES=3

echo "Run experiments for ${problem} with ${levels} levels to ${result_folder}"
mkdir ${result_folder}
# run Jacobi(double)
echo "Run Jacobi(double)"
./multigrid-solver cuda ${problem}/A_mg_0.mtx ${problem}/F_0.mtx ${levels} direct 1e-4 1 double 3 0.66 10 false ${problem}/A_mg > ${result_folder}/jacobi_double.txt

for precision in double single half; do
    echo "Run IC(${precision})"
    ./multigrid-solver cuda ${problem}/A_mg_0.mtx ${problem}/F_0.mtx ${levels} direct 1e-4 2 ${precision} 3 2.0 10 false ${problem}/A_mg > ${result_folder}/ic_${precision}.txt
done

for precision in double single half; do
    echo "Run ParICT (generate on ref) (${precision})"
    ./multigrid-solver cuda ${problem}/A_mg_0.mtx ${problem}/F_0.mtx ${levels} direct 1e-4 3 ${precision} 3 2.0 10 false ${problem}/A_mg > ${result_folder}/ict_${precision}.txt
done


