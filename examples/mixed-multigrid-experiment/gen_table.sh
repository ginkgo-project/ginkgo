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
PRECISION="$2"
smoother=jacobi
table_format="c"
header=""
label="problem"
for prec in $PRECISION; do
    table_format="${table_format} | r r r"
    label="${label} & res. norm & \#iter & time[ms]"
    prec_text=""
    if [[ "${prec}" == "0" ]]; then
        prec_text="DP"
    elif [[ "${prec}" == "1" ]]; then
        prec_text="DP-SP"
    elif [[ "${prec}" == "2" ]]; then
        prec_text="DP-SP-HP"
    elif [[ "${prec}" == "3" ]]; then
        prec_text="DP-HP"
    elif [[ "${prec}" == "7" ]]; then
        prec_text="DP-SP-BF"
    elif [[ "${prec}" == "8" ]]; then
        prec_text="DP-BF"
    else
        >&2 echo "Unknown Precision number ${prec}"
        exit 1
    fi
    header="${header} & \multicolumn{3}{c|}{\gko's AMG (${prec_text})}"
done
for num_levels in 10; do
    for cycle in v; do
        for mg_mode in cg; do
            output=${RESULT_FOLDER}/latex_${num_levels}_${cycle}_${mg_mode}_${PRECISION// /}.tex
            echo "\begin{table*}[t]" > ${output}
            echo "\centering" >> ${output}
            echo "\scriptsize" >> ${output}
            echo "\begin{tabular}{${table_format}}" >> ${output}
            echo "\toprule" >> ${output}
            echo "${header} \\\\" >> ${output}
            echo "${label} \\\\" >> ${output}
            echo "\midrule" >> ${output}
            for list in AMGX_LISTS MFEM_LISTS; do
                for matrix in ${!list}; do
                    echo "matrix ${matrix} in ${mtx}"
                    line="${matrix}"
                    for mixed_mode in ${PRECISION}; do
                        echo "|_ mixed_mode ${mixed_mode}"
                        label="${matrix}_mixed${mixed_mode}_level${num_levels}"
                        suffix="_cycle${cycle}_mode${mg_mode}_${smoother}_scale0"
                        file="${RESULT_FOLDER}/${label}${suffix}.txt"
                        residual_norm=$(tail -n 8 ${file} | head -n 1)
                        info=$(tail -n 7 ${file} | head -n 6 | sed -E 's/[^0-9\.]//g' | tr '\n' ',' | sed -E 's/,$/\n/g')
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


