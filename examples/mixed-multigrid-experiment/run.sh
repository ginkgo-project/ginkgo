#!/bin/bash
AMGX_LISTS="
2cubes_sphere
cage13
cage14
offshore
thermal2
tmt_sym
"

RESULT_FOLDER=$1
EXECUTOR=$2
DATA_FOLDER=$3
for matrix in ${AMGX_LISTS}; do
    mtx="${DATA_FOLDER}/amgx_data/${matrix}/${matrix}.mtx"
    echo "matrix ${matrix} in ${mtx}"
    for mixed_mode in 0 1 2 3; do
        echo "|_ mixed_mode ${mixed_mode}"
        for num_levels in 3 10; do
            echo "   |_ num_levels ${num_levels}"
            ./mixed-multigrid-solver ${EXECUTOR} ${mixed_mode} ${num_levels} ${mtx} > ${RESULT_FOLDER}/${matrix}_mixed${mixed_mode}_level${num_levels}.txt
        done
    done
done

MFEM_LISTS="
beam-pw-sv0.1-o-3-l-3
l-shape-const-o-3-l-7
"
for matrix in ${MFEM_LISTS}; do
    echo "matrix ${matrix}"
    mtx="${DATA_FOLDER}/mfem_data/A-${matrix}-diag1bc.dat"
    b="${DATA_FOLDER}/mfem_data/b-${matrix}.dat"
    for mixed_mode in 0 1 2 3; do
        echo "|_ mixed_mode ${mixed_mode}"
        for num_levels in 3 10; do
            echo "   |_ num_levels ${num_levels}"
            ./mixed-multigrid-solver ${EXECUTOR} ${mixed_mode} ${num_levels} ${mtx} ${b}> ${RESULT_FOLDER}/${matrix}_mixed${mixed_mode}_level${num_levels}.txt
        done
    done
done
