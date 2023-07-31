#!/usr/bin/awk -f

!/^#/ {
    TIMES[$4] += ($2 - $1)/1000
    COUNT[$4] += 1
}

END {
    for (TGT in TIMES)
        AVG[TGT]=TIMES[TGT]/COUNT[TGT]
    asorti(AVG, SORTED, "@val_num_desc")
    for (num in SORTED)
        print AVG[SORTED[num]] " " SORTED[num]
}