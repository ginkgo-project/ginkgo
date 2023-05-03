

AMGX_LISTS="
2cubes_sphere
cage13
cage14
offshore
thermal2
tmt_sym
"
MFEM_LISTS="
beam-pw-sv0.1-o-3-l-3
l-shape-const-o-3-l-7
"
RESULT_FOLDER=$1
EXECUTOR=$2
DATA_FOLDER=$3
MIXED_MODE="0 1 2 3 -1 -2 -21 -22 -23"
MG_MODE="$4"
#"cg preconditioner"
SM_MODE="$5"
#"jacobi bj l1cheyb"
run() {
    local matrix="$1"
    local num_levels="$2"
    local cycle="$3"
    local scale="$4"
    local mtx="$5"
    local b="$6"
    for mixed_mode in ${MIXED_MODE}; do
        echo "|_ mixed_mode ${mixed_mode}"
        for mg_mode in ${MG_MODE}; do
            echo "   |_ mg_mode ${mg_mode}"
            for sm_mode in ${SM_MODE}; do
                echo "      |_ sm_mode ${sm_mode}"
                ./mixed-multigrid-experiment ${EXECUTOR} ${mixed_mode} ${num_levels} ${cycle} ${mg_mode} ${sm_mode} ${scale} ${mtx} ${b} > ${RESULT_FOLDER}/${matrix}_mixed${mixed_mode}_level${num_levels}_cycle${cycle}_mode${mg_mode}_${sm_mode}.txt
                echo "./mixed-multigrid-experiment ${EXECUTOR} ${mixed_mode} ${num_levels} ${cycle} ${mg_mode} ${sm_mode} ${scale} ${mtx} ${b} > ${RESULT_FOLDER}/${matrix}_mixed${mixed_mode}_level${num_levels}_cycle${cycle}_mode${mg_mode}_${sm_mode}.txt"
            done
        done
    done
}

for matrix in ${AMGX_LISTS}; do
    mtx="${DATA_FOLDER}/amgx_data/${matrix}/${matrix}.mtx"
    scale="0"
    if [[ "${matrix}" == "2cubes_sphere" ]] || [[ "${matrix}" == "offshore" ]]; then
        scale="1"
    fi
    run ${matrix} 10 v ${scale} ${mtx}
    # run ${matrix} 3 w ${scale} ${mtx}
done


for matrix in ${MFEM_LISTS}; do
    mtx="${DATA_FOLDER}/mfem_data/A-${matrix}-diag1bc.dat"
    b="${DATA_FOLDER}/mfem_data/b-${matrix}.dat"
    run ${matrix} 10 v 0 ${mtx} ${b}
done