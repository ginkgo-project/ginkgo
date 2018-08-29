#!/bin/bash
set -e
SSGET="ssget"
ROW="0,-1"
COL="0,-1"
NONZEROS="0,-1"
PATTERN_SYM="0,1"
NUMERICAL_SYM="0,1"
POSITIVE="-1"
TYPE="srtw"
VALUE="brc"
TEMP_L=""
TEMP_R=""
REPLY=""

extract() {
    IFS="," read -ra INTERVAL <<< $1
    TEMP_L=${INTERVAL[0]}
    TEMP_R=${INTERVAL[1]}
}

check_integer() {
    # check_integer id property value
    extract $3
    RESULTS=$(${SSGET} -i $1 -p $2)
    if ([[ "$PASS" == "true" ]]) && ([[ $TEMP_L -eq -1 ]] || [[ $RESULTS -ge $TEMP_L ]]) && ([[ $TEMP_R -eq -1 ]] || [[ $RESULTS -le $TEMP_R ]]); then
        PASS="true"
    else
        PASS="false"
    fi
}

main() {
    NUM_PROBLEMS=$(${SSGET} -n)
    for (( i=1; i <= ${NUM_PROBLEMS}; ++i )); do
        PASS="true"
        check_integer $i "rows" $ROW
        check_integer $i "cols" $COL
        check_integer $i "nonzeros" $NONZEROS
        if [[ "$PASS" == "true" ]]; then
            echo "${i}"
        fi
    done
}
print_usage_and_exit() {
    cat 1>&2 << EOT
Usage: $0 [options]
Available options:
    -e           ssget executor (default:ssget)
    -h           show this help
    -r           the number of rows
    -c           the number of columns
    -n           the number of nonzeros
    -p           the pattern symmetry
    -s           the numerical symmetry
    -d           whether is positive definite
    -t           type of structure (square, rectangular, tallskinny, wideshort)
    -v           type of value (bool,real,complex)
EOT
    exit $1
}
while getopts ":e:r:c:n:" opt; do
    case ${opt} in
        :)
            echo 1>&2 "Option -${OPTARG} provided without an argument"
            print_usage_and_exit 2
            ;;
        \?)
            echo 1>&2 "Unknown option: -${OPTARG}"
            print_usage_and_exit 1
            ;;
        e)  
            SSGET=${OPTARG}
            ;;
        r)
            if [[ "${OPTARG}" =~ ^(-1|([0-9]+))$ ]]; then
                ROW=${OPTARG},${OPTARG}
            elif [[ "${OPTARG}" =~ ^(-1|([0-9]+)),(-1|([0-9]+))$ ]]; then
                ROW=${OPTARG}
            else
                echo 1>&2 "ROW has to be a number or region, got: ${OPTARG}"
                print_usage_and_exit 3
            fi
            ;;
        c)
            if [[ "${OPTARG}" =~ ^(-1|([0-9]+))$ ]]; then
                COL=${OPTARG},${OPTARG}
            elif [[ "${OPTARG}" =~ ^(-1|([0-9]+)),(-1|([0-9]+))$ ]]; then
                COL=${OPTARG}
            else
                echo 1>&2 "COL has to be a number or region, got: ${OPTARG}"
                print_usage_and_exit 4
            fi
            ;;
        n)
            if [[ "${OPTARG}" =~ ^(-1|([0-9]+))$ ]]; then
                NONZEROS=${OPTARG},${OPTARG}
            elif [[ "${OPTARG}" =~ ^(-1|([0-9]+)),(-1|([0-9]+))$ ]]; then
                NONZEROS=${OPTARG}
            else
                echo 1>&2 "NONZEROS has to be a number or region, got: ${OPTARG}"
                print_usage_and_exit 5
            fi
            ;;
        h)
            print_usage_and_exit 0
            ;;
    esac
done

main


