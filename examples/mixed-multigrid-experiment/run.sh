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
MIXED_MODE="$4"
NUM_LEVELS="$5"
CYCLE="$6"
MODE="$7"
if [[ "${MIXED_MODE}" == "" ]]; then
    MIXED_MODE="0 1 2 3"
fi
if [[ "${NUM_LEVELS}" == "" ]]; then
    NUM_LEVELS="3 10"
fi
if [[ "${CYCLE}" == "" ]]; then
    CYCLE="v w f"
fi
if [[ "${MODE}" == "" ]]; then
    MODE="solver cg"
fi
run() {
    local matrix="$1"
    local mtx="$2"
    local b="$3"
    echo "matrix ${matrix} in ${mtx}"
    echo "b is ${b}"
    for mixed_mode in ${MIXED_MODE}; do
        echo "|_ mixed_mode ${mixed_mode}"
        for num_levels in ${NUM_LEVELS}; do
            echo "   |_ num_levels ${num_levels}"
            for cycle in ${CYCLE}; do
                echo "      |_ cycle ${cycle}"
                for mg_mode in ${MODE}; do
                    echo "         |_ mg_mode ${mg_mode}"
                    ./mixed-multigrid-solver ${EXECUTOR} ${mixed_mode} ${num_levels} ${cycle} ${mg_mode} ${mtx} ${b} > ${RESULT_FOLDER}/${matrix}_mixed${mixed_mode}_level${num_levels}_cycle${cycle}_mode${mg_mode}.txt
                done
            done
        done
    done
}


for matrix in ${AMGX_LISTS}; do
    mtx="${DATA_FOLDER}/amgx_data/${matrix}/${matrix}.mtx"
    run ${matrix} ${mtx}
done

MFEM_LISTS="
beam-pw-sv0.1-o-3-l-3
l-shape-const-o-3-l-7
"
for matrix in ${MFEM_LISTS}; do
    mtx="${DATA_FOLDER}/mfem_data/A-${matrix}-diag1bc.dat"
    b="${DATA_FOLDER}/mfem_data/b-${matrix}.dat"
    run ${matrix} ${mtx} ${b}
done
