# Purpose
The purpose of this file is to explain how to create or use containers for Ginkgo. 

Custom containers are used in Ginkgo in order to test the correct functionality
of the library. As Ginkgo is a C++ CUDA-enabled library, it is important to test
both a wide variety of compilers and CUDA versions as part of the development
process. This allows to ensure Ginkgo is and stays compatible with the specified
compilers and CUDA versions.
# Tools used
To create and deploy containers, we will use:
+ [NVIDIA's container registry](https://ngc.nvidia.com/registry/nvidia-cuda)
+ [NVIDIA HPC Container Maker (HPCCM)](https://github.com/NVIDIA/hpc-container-maker/)
+ [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) should be installed and available
+ A [local docker registry](https://docs.docker.com/registry/deploying/#run-a-local-registry) should be up and running
+ docker and gitlab-runner
# Ginkgo containers
Creating container images is a tedious task. The [usual
process](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
requires writing what is called a `Dockerfile` which contains all commands
needed to build an image.

To facilitate building new docker images, it is advised to start with an already
existing container image (such as an ubuntu image), and extend it with new
functionalities to generate a new container. In our context this is what we will
be doing. Nevertheless, to facilitate container generation we have decided to
rely on NVIDIA's HPCCM.

## Ginkgo HPCCM recipes
HPCCM facilitates the container creation process significantly through a
high-level interface. HPCCM uses 'recipes', python files containing base
instructions, similar to a cookbook, tailored to generate Dockerfiles. Recipes
can take in arguments which allows to reuse the same recipe for building a wide
variety of containers. By default, HPCCM supports multiple packages and Linux
distributions which increases the portability of the HPCCM recipes. 

### Description
Ginkgo provides two recipes for creating containers. They are :
+ ginkgo-cuda-base.py: based on [NVIDIA's docker images](https://ngc.nvidia.com/registry/nvidia-cuda)
+ ginkgo-nocuda-base.py: based on the basic ubuntu image

There is minor differences, but all of Ginkgo's recipes install the following
packages:
+ GNU compilers
+ LLVM/Clang
+ Intel Compilers
+ OpenMP
+ Python 2 and 3
+ cmake
+ git, openssh, doxygen, curl (these are required for some synchronization or
  documentation building jobs)
+ valgrind, graphviz, jq (documentation and debugging)

### CUDA recipes
Every container is tailored to have matching CUDA, GNU Compilers and LLVM/Clang
versions. The information for compatible versions can usually be found in
[NVIDIA's
documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

+ CUDA is provided by default from nvidia-cuda, and requires no particular setup.
+ GNU and Clang compilers should use the extra_packages argument in order to
  have access to a repository providing all compiler versions (otherwise the
  default limit is gcc 5.4).
+ Arguments can be provided for CUDA, GNU and LLVM version.
+ It is required to use `libomp-dev` library for Clang+OpenMP to work.
+ hwloc is built and the server's topology is added to the container.
+ Finally, `LIBRARY_PATH` and `LD_LIBRARY_PATH` are properly setup for the CUDA
  library. For proper CMake detection of the GPUs, this should maybe be
  extended.
  
  
The dockerfiles and container images already generated are:
+ CUDA 9.0, GNU 5.5, LLVM 3.9, no Intel
+ CUDA 9.1, GNU 6, LLVM 4.0, Intel 2017 update 4
+ CUDA 9.2, GNU 7, LLVM 5.0, Intel 2017 update 4
+ CUDA 10.0, GNU 7, LLVM 6.0, Intel 2018 update 1
+ CUDA 10.1, GNU 8, LLVM 7, Intel 2019 update 4

### No CUDA recipe
Because CUDA limits the versions of compilers it can work with, it is good
practice to provide non-CUDA containers, particularly for the more recent
compilers.

The base image for this recipe is the same Ubuntu version as NVIDIA's to keep
the systems as similar as possible. There is only one extra difference for this
recipe: the image is very light and does not include the `make` command by
default, so it is necessary to add the `build-essential` package to the
requirements.

In addition to the previous argument, an extra `papi` argument can be given.
This argument can be set to `True` to indicate that the image should be built
for papi support. In this case, if papi files can be found, the library perfmon
(`libpfm4`) is added do the docker container and papi files are copied to the
container from a folder named `papi/` with the following format:
+ `papi/include`: papi include files
+ `papi/lib`: papi pre-built library files
+ `papi/bin`: papi pre-built binary files

The dockerfiles and container images already generated are:
+ GNU 9, LLVM 8 , Intel 2019 update 4.
## Using HPCCM recipes and docker to create containers
The following explains how to use recipes and docker to create new containers.
### Generate the Dockerfile
This is done with the NVIDIA's HPCCM tool. A base recipe should be given and the
 output should be written to a file, the dockerfile.
```bash
 hpccm --recipe ginkgo-cuda-base.py --userarg cuda=10.0 gnu=8 llvm=6.0 > gko-cuda100-gnu8-llvm60.baseimage
```
### Using docker to build the container
The command simply uses `docker build` the standard command for container
generation. 
```bash
docker build -t localhost:5000/gko-cuda100-gnu8-llvm60 -f gko-cuda100-gnu8-llvm60.baseimage .
```
A name is given to the image through `-t tag`. It is required to append
`localhost:5000/` to designate our server's local container registry.
The base image (or dockerfile) is given through the `-f` argument.
The path given here is `.`. This is important if building an image from the
`gko-nocuda-base` base image. This indicates the path where the relevant papi
pre-built files (`papi/include/...`, etc) files to be put into the container can
be found.
### Test the generated container
The created container should be tested to ensure all supposed functionalities
are present and properly working. Here is a standard procedure for this:
```bash
# get interactive access to a container
docker run --rm --runtime=nvidia -ti localhost:5000/gko-cuda100-gnu8-llvm60 
nvidia-smi
g++ --version
clang++ --version
```

In addition, it can be useful to test the CUDA and OpenMP functionality, for
this purpose a short C-program can be created such as:
```c++
	 #include <cuda.h>
	 #include <omp.h>
	 #define SOME_LIMIT 10000

	 int main()
	 {
		 cuInit(0); //whatever other CUDA standard API call

		 int acc = 0;
	 #pragma omp parallel for
		 for (int i=0; i<SOME_LIMIT; i++)
			 acc+=i*4+1; //whatever
		 return 0;
	 }
```

This should be compiled and tested with
+ g++ test.cpp -lcuda -I/usr/local/cuda/include -fopenmp
+ clang++ test.cpp -lcuda -I/usr/local/cuda/include -fopenmp

### Push the container to the local registry
Assuming the local registry is up and running, then with the previous steps
properly done (i.e. appending localhost:5000/ to all docker image names), then
pushing the image to the registry is:
```bash
docker push localhost:5000/gko-cuda100-gnu8-llvm60
```
