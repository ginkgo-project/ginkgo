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
for num_levels in 10; do
    for cycle in v w f; do
        for mg_mode in solver cg; do
            output=${RESULT_FOLDER}/collect_case_${num_levels}_${cycle}_${mg_mode}.csv
            echo "matrix, final residual, iteration count, generation time[ms], total execution time[ms], executation time per iteration[ms]" > ${output}
            for list in AMGX_LISTS MFEM_LISTS; do
                for matrix in ${!list}; do
                    echo "matrix ${matrix} in ${mtx}"
                    line=""
                    for mixed_mode in 0 1 2 3; do
                        echo "|_ mixed_mode ${mixed_mode}"
                        if [[ "${mixed_mode}" != "0" ]]; then
                            line="${line}, "
                        fi
                        label="${matrix}_mixed${mixed_mode}_level${num_levels}"
                        suffix="_cycle${cycle}_mode${mg_mode}"
                        file="${RESULT_FOLDER}/${label}${suffix}.txt"
                        residual_norm=$(tail -n 5 ${file} | head -n 1)
                        info=$(tail -n 4 ${file} | sed -E 's/[^0-9\.]//g' | tr '\n' ',' | sed -E 's/,$/\n/g')
                        line="${line}${label}, ${residual_norm}, ${info}"
                    done
                    echo "${line}" >> "${output}"
                done
            done
        done
    done
done

