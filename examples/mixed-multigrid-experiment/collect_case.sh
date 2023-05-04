#!/bin/bash

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
MG_MODE="$2"
#"cg preconditioner"
SM_MODE="$3"
#"jacobi bj l1cheyb"
for num_levels in 10; do
    for cycle in v; do
        for mg_mode in ${MG_MODE}; do
            for sm_mode in ${SM_MODE}; do
                output=${RESULT_FOLDER}/collect_case_${num_levels}_${cycle}_${mg_mode}_${sm_mode}.csv
                echo "matrix, final residual, iteration count, generation time[ms], total execution time[ms], executation time per iteration[ms]" > ${output}
                for list in AMGX_LISTS MFEM_LISTS; do
                    for matrix in ${!list}; do
                        echo "matrix ${matrix} in ${mtx}"
                        line=""
                        for mixed_mode in 0 1 2 3 -11 -12 -13 -21 -22 -23; do
                            echo "|_ mixed_mode ${mixed_mode}"
                            # if [[ "${mixed_mode}" != "0" ]]; then
                            #     line="${line}, "
                            # fi
                            label="${matrix}_mixed${mixed_mode}"
                            suffix="_level${num_levels}_cycle${cycle}_mode${mg_mode}_${sm_mode}"
                            file="${RESULT_FOLDER}/${label}${suffix}.txt"
                            residual_norm=$(tail -n 5 ${file} | head -n 1)
                            info=$(tail -n 4 ${file} | sed -E 's/[^0-9\.]//g' | tr '\n' ',' | sed -E 's/,$/\n/g')
                            # line="${line}${label}, ${residual_norm}, ${info}"
                            echo "${label}, ${residual_norm}, ${info}" >> "${output}"
                        done
                        # echo "${line}" >> "${output}"
                    done
                done
            done
        done
    done
done

