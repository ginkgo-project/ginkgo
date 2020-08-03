In this file we explain how to generate and analyze results with adaptive
precision block-Jacobi, in particular to reproduce the figure 8 from the
relevant paper. We assume that the code is benchmarked on the Summit machine. If
that is not the case, we cannot help with packages selection and other details
such as job submission. For any issue reproducing these experiments please send
a mail to <mailto:ginkgo.library@gmail.com>.

The main steps are as follows:
1. Installing ssget and prefetch the matrices from the suitesparse
   collection
2. Download and build Ginkgo
3. Prepare the experiment scripts
4. Run the experiments
5. Publish the experiments to github and tie to the information in the
   previous mail for generating the plots.


### 1: Fetching the matrices
First of all, a tool is required for benchmarking:
https://github.com/ginkgo-project/ssget

This tool is a bash script simplifying downloading matrices from the SuiteSparse
matrix collection. The script can be put anywhere in the `PATH`, but line 39
(ARCHIVE_LOCATION) has to be configured, this is where the downloaded matrices
will be stored. On summit, this would typically have to be somewhere in
`$MEMBERWORK/<project>/....`, since this has better access inside jobs.

The matrices used for the experiments can be pre-downloaded, as this saves some
node time:
```bash
for i in $(seq 0 $(ssget -n)); do
    posdef=$(ssget -p posdef -i $i)
    cols=$(ssget -p cols -i $i)
    nnz=$(ssget -p nonzeros -i $i)
    if [ "$posdef" -eq 1 -a "$cols" -lt 10000000 -a "$nnz" -lt 500000000 ]; then
        ssget -f -i $i
		fi
done
```


### 2 - Building Ginkgo
Afterwards, Ginkgo can be cloned, configured and built, here are the steps. All
paths can be adapted as needed. The <...> (project) part absolutely needs to be
replaced:

```bash
project=<project>
ginkgo_source=$HOME/TOMS-bj-reproduce/ginkgo
ginkgo_build=$MEMBERWORK/${project,,}/TOMS-bj-reproduce/ginkgo-build
module load gcc/6.4.0 cuda/9.2.148 cmake/3.15.2 git/2.20.1
# For every new session, the previous setup is required
git clone https://github.com/ginkgo-project/ginkgo.git ${ginkgo_source} --branch 2019toms-adaptive-bj-solver
mkdir -p ${ginkgo_build} && cd ${ginkgo_build}
cmake -DBUILD_CUDA=on -DBUILD_OMP=off -DBUILD_EXAMPLES=off -DBUILD_GTEST=on -DDEVEL_TOOLS=off -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) ${ginkgo_source}
bsub -P ${project^^} -W 2:00 -nnodes 1 jsrun -n 1 -c 10 -g 0 make -j10
# This is a good time to go do something else, compilation will take a
# while as there is a big CUDA compiler bug which makes it extremely slow to
# compile the block jacobi with all optimizations. A job submission is required
# since due to this bug the CUDA compiler exceeds the 16GB limit on Summit's
# login nodes specifically for the block jacobi compilation...
make -j10 # afterwards, ensure everything is compiled
make test
# Everything should run without failure. If cuda tests fail logging
# in again might solve some issue, this could be due to the hardware
# restrictions on summit after 4 hours of login time.
```

### 3 - Prepare the experiment scripts
This simply creates two files for launching the experiments. A
ginkgo_benchmark.lsf for `bsub`, and a `benchmark_one_node.sh` which runs jsrun
and populates some arguments in order to create segments to be benchmarked, all
of which can run in parallel.

```bash
cat > ${ginkgo_source}/benchmark_one_node.sh << EOF
#!/bin/bash -x

cd \${1}/benchmark
chmod +x run_all_benchmarks.sh

ADAPTIVE_JACOBI_ACCURACY=\${4:-1e-1}
export BENCHMARK=solver
export PRECONDS=none,jacobi,adaptive-jacobi
export SYSTEM_NAME=V100_summit
export SEGMENT_ID=\${2}
export SEGMENTS=\${3}
./run_all_benchmarks.sh >/dev/null
EOF

cat > $ginkgo_source/benchmark_ginkgo.lsf << EOF
#!/bin/bash
#BSUB -P ${project^^}
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -J Ginkgo_Benchmark
#BSUB -o Ginkgo_Benchmark.%J
#BSUB -e Ginkgo_Benchmark.%J

if [ -z \${segment_id+x} ]
then
        echo "Please set variable segment_id"
        exit
fi

if [ -z \${segments+x} ]
then
        echo "Please set variable segments"
        exit
fi

module load gcc/6.4.0 cuda/9.2.148 cmake/3.15.2 git/2.20.1

jsrun -n 1 -a 1 -c 1 -g 1 $ginkgo_source/benchmark_one_node.sh $ginkgo_build \$segment_id \$segments
EOF

chmod +x ${ginkgo_source}/benchmark_one_node.sh
```


### 4 - Run the benchmarks
To run the benchmarks there are two parameters to pick:
+ the parallelism desired,
+ the number of matrices we want to reproduce against (all of them or a
  portion).


These are controlled with the variables `segments` and `segment_id`. As an
example, the following will run 20 benchmarks in parallel and benchmark all
matrices since we use all `segment_id`.

```bash
for i in $(seq 1 20); do segments=20 segment_id=$i bsub $ginkgo_source/benchmark_ginkgo.lsf; done
```

To only benchmark the first half of the matrices, we could do the following:
```bash
# Note the different in the `seq` below
for i in $(seq 1 10); do segments=20 segment_id=$i bsub $ginkgo_source/benchmark_ginkgo.lsf; done
```

### 5 - Publish the results and generate the plots
For analyzing the results, any tool can be used. The previous experiments generated json files for each matrices, each containing timing and convergence results without preconditioner, with standard block-Jacobi preconditioner, and with adaptive precision block-Jacobi.

In this section, we describe how to generate the plots by using Ginkgo's
[GPE](https://ginkgo-project.github.io/gpe/) tool. First, we need to publish the
experiments into a Github repository which will be then linked as source input
to the GPE. For this, we can simply fork the ginkgo-data repository. To do so,
we can go to the github repository and use the forking interface:
https://github.com/ginkgo-project/ginkgo-data/tree/2019toms-adaptive-bj

Once it's done, we want to clone the 2019toms-adaptive-bj branch, put all
results online and access the GPE for plotting the results. Here are the
detailed steps:
```bash
git clone https://github.com/<username>/ginkgo-data.git ${ginkgo_build}/benchmark/ginkgo-data --branch 2019toms-adaptive-bj
rsync -rtv ${ginkgo_build}/benchmark/results/ ${ginkgo_build}/benchmark/ginkgo-data/data/
cd ${ginkgo_build}/benchmark/ginkgo-data/data/
# The following updates the main `.json` files with the list of data
module load python/3.7.0
./build-list . > list.json
./agregate < list.json > agregate.json
git config --local user.name "<Name>"
git config --local user.email "<email>"
git commit -am "Ginkgo Reproduced BJ data"
git push
```

For the generating the plots in the GPE, here are the steps to go through:
1. Access the GPE: https://ginkgo-project.github.io/gpe/
2. Update data root URL, from
   https://raw.githubusercontent.com/ginkgo-project/ginkgo-data/master/data to
   https://raw.githubusercontent.com/<username>/ginkgo-data/2019toms-adaptive-bj/data
3. Click on the arrow to load the data, select the `Result Summary` entry above.
   The first few entries under this should be V100(cuda).
4. Click on select an example to choose a plotting script, and update the url
   from https://raw.githubusercontent.com/ginkgo-project/ginkgo-data/master/plots to
   https://raw.githubusercontent.com/<username>/ginkgo-data/2019toms-adaptive-bj/plots
5. Again Click on the arrow next to the URL to load everything
6. Select the plot  "Preconditioned CG detailed comparison"
7. The results should be available in the tab "plot" on the right side


### 6 - Generate results and plots for precision 1e-2
The previous steps benchmarked and generated the plot with block jacobi
precision 1e-1, to generate the results with 1e-2, both steps 4-5 need to be
repeated. The only changes is:

Edit `$ginkgo_source/benchmark_ginkgo.lsf`, append to the end of the jsrun line:
"1e-2"

In GPE, plotting with the previous link will now show precision 1e-2 by default.
To get back to the 1e-1 precision, replace `2019toms-adaptive-bj` in the link by
the previous commit hash.
