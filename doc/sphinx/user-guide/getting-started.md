# Getting Started





## Backends

Ginkgo supports running on the following architectures:
- multi-core x86 CPUs
- NVIDIA GPUs
- AMD GPUs
- Intel GPUs

We denote the architecture specific implementations in Ginkgo as backends.
The following backends are provided in Ginkgo:

Reference
: A CPU backend that runs on a single core. This is not optimized and should only be used for verification.

OMP
: A CPU backend that runs on multiple cores using OpenMP.

CUDA
: A GPU backend that runs on a single NVIDIA GPU.

HIP
: A GPU backend that runs on a single AMD GPU.

SYCL
: A GPU backend that runs on a single Intel GPU.

Multiple backends can be used at the same time.
Which backends can be used only depends on the system, see [](build-install.system-requirements.md) for more details.

The particular backend Ginkgo code runs on is chosen at runtime through an implementation of the `Executor` interface.

:::{warning}
add links, whatever to the executor reference
:::



- support different backends
- general overview
- high level example

```{literalinclude} ../../../examples/getting-started/getting-started.cpp
:language: cpp
:lines: 5-
```

:::{toctree}
functionality
:::