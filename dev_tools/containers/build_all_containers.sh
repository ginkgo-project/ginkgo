#!/bin/bash


hpccm --recipe ginkgo-nocuda-base.py --userarg gnu=8 llvm=7 papi=True metis=True > gko-nocuda-gnu8-llvm70.baseimage
list=('gko-nocuda-gnu8-llvm70.baseimage')
hpccm --recipe ginkgo-nocuda-base.py --userarg gnu=9 llvm=8 papi=True > gko-nocuda-gnu9-llvm8.baseimage
list+=('gko-nocuda-gnu9-llvm8.baseimage')
if [ "$HOSTNAME" = "amdci" ]; then
  list+=('gko-amd-gnu7-llvm60.baseimage')
else
  hpccm --recipe ginkgo-cuda-base.py --userarg cuda=10.1 gnu=8 llvm=7 > gko-cuda101-gnu8-llvm70.baseimage
  hpccm --recipe ginkgo-cuda-base.py --userarg cuda=10.0 gnu=7 llvm=6.0 > gko-cuda100-gnu7-llvm60.baseimage
  hpccm --recipe ginkgo-cuda-base.py --userarg cuda=9.2 gnu=7 llvm=5.0 > gko-cuda92-gnu7-llvm50.baseimage
  hpccm --recipe ginkgo-cuda-base.py --userarg cuda=9.1 gnu=6 llvm=4.0 > gko-cuda91-gnu6-llvm40.baseimage
  hpccm --recipe ginkgo-cuda-base.py --userarg cuda=9.0 gnu=5 llvm=3.9 > gko-cuda90-gnu5-llvm39.baseimage
  list+=(gko-cuda*.baseimage)
fi

for i in "${list[@]}"
do
  name=$(echo $i | cut -d"." -f1)
  docker build -t localhost:5000/$name -f $i .
  docker push localhost:5000/$name
done
