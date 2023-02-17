MATRIX_LISTS="
2cubes_sphere
cage13
cage14
offshore
thermal2
tmt_sym
beam
l-shape
"
RESULT_FOLDER=$1
EXECUTOR=$2
DATA_FOLDER=$3
for matrix in ${MATRIX_LISTS}; do
    command_args="--formats=csrc --executor=${EXECUTOR} --detailed=false < ${DATA_FOLDER}/mg_${matrix}.yml > ${RESULT_FOLDER}/mg_${matrix}.yml"
    ./spmv --formats=csrc --executor=${EXECUTOR} --detailed=false < ${DATA_FOLDER}/mg_${matrix}.json > ${RESULT_FOLDER}/mg_${matrix}.json
    for prec in single half; do
        ./spmv_${prec} --formats=csrc --executor=${EXECUTOR} --detailed=false < ${DATA_FOLDER}/mg_${matrix}.json > ${RESULT_FOLDER}/mg_${matrix}_${prec}.json
    done
done