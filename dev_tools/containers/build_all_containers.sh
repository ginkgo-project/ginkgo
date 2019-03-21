#!/bin/bash
 
hpccm --recipe ginkgo-nocuda-base.py --userarg gnu=8 llvm=6.0 papi=True > gko-nocuda-gnu8-llvm70.baseimage
hpccm --recipe ginkgo-cuda-base.py --userarg cuda=10.0 gnu=7 llvm=6.0 > gko-cuda100-gnu7-llvm60.baseimage
hpccm --recipe ginkgo-cuda-base.py --userarg cuda=9.2 gnu=7 llvm=5.0 > gko-cuda92-gnu7-llvm50.baseimage
hpccm --recipe ginkgo-cuda-base.py --userarg cuda=9.1 gnu=6 llvm=4.0 > gko-cuda91-gnu6-llvm40.baseimage
hpccm --recipe ginkgo-cuda-base.py --userarg cuda=9.0 gnu=5 llvm=3.9 > gko-cuda90-gnu5-llvm39.baseimage
for i in $(ls *.baseimage)
do
	name=$(echo $i | cut -d"." -f1)
	docker build -t localhost:5000/$name -f $i .
	docker push localhost:5000/$name
done
