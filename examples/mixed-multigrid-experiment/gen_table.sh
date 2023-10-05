#!/usr/bin/env bash

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
smoother=jacobi
for num_levels in 10; do
    for cycle in v; do
        for mg_mode in cg; do
            output=${RESULT_FOLDER}/latex_${num_levels}_${cycle}_${mg_mode}.tex
            echo "\begin{table*}[t]" > ${output}
            echo "\centering" >> ${output}
            echo "\scriptsize" >> ${output}
            echo "\begin{tabular}{c | r r r | r r r | r r r | r r r | r r r | r r r}" >> ${output}
            echo "\toprule" >> ${output}
            echo " & \multicolumn{3}{c|}{\gko's AMG (DP)} & \multicolumn{3}{c|}{\gko's AMG (DP SP)} & \multicolumn{3}{c|}{\gko's AMG (DP FP HP)} & \multicolumn{3}{c|}{\gko's AMG (DP HP)} & \multicolumn{3}{c|}{\gko's AMG (DP SP BF)}  & \multicolumn{3}{c}{\gko's AMG (DP BF)} \\\\" >> ${output}
            echo "problem & res. norm & \#iter & time[ms] & res. norm & \#iter & time[ms] & res. norm & \#iter & time[ms] & res. norm & \#iter & time[ms] & res. norm & \#iter & time[ms] & res. norm & \#iter & time[ms] \\\\" >> ${output}
            echo "\midrule" >> ${output}
            for list in AMGX_LISTS MFEM_LISTS; do
                for matrix in ${!list}; do
                    echo "matrix ${matrix} in ${mtx}"
                    line="${matrix}"
                    for mixed_mode in 0 1 2 3 7 8; do
                        echo "|_ mixed_mode ${mixed_mode}"
                        label="${matrix}_mixed${mixed_mode}_level${num_levels}"
                        suffix="_cycle${cycle}_mode${mg_mode}_${smoother}_scale0"
                        file="${RESULT_FOLDER}/${label}${suffix}.txt"
                        residual_norm=$(tail -n 7 ${file} | head -n 1)
                        info=$(tail -n 6 ${file} | head -n 5 | sed -E 's/[^0-9\.]//g' | tr '\n' ',' | sed -E 's/,$/\n/g')
                        # iteration count, generation time[ms], total execution time[ms], executation time per iteration[ms], total execution median time[ms]
                        IFS=',' read -r -a array <<< "$info"
                        line="${line} & ${residual_norm} & ${array[0]} & ${array[4]}"
                    done
                    echo "${line} \\\\" >> "${output}"
                done
            done
            echo "\bottomrule" >> "${output}"
            echo "\end{tabular}" >> "${output}"
            echo "\caption{}" >> "${output}"
            echo "\end{table*}" >> "${output}"
        done
    done
done


