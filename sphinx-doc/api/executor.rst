.. _api_executor:

Executor
========

Overview
^^^^^^^^

NeoFOAM uses the MPI+X approach for parallelism, where X is the execution space used for parallelism. The `Executor` class uses Kokkos and provides an interface for memory management and specifics were to execute the operations:

- `CPUExecutor`: run on the CPU with MPI
- `OMPExecutor`: run on the CPU with OpenMP and MPI
- `GPUExecutor`: run on the GPU with MPI

Design
^^^^^^

One of the design goals is the ability to quickly switch between the executor models at runtime.


.. code-block:: cpp

        NeoFOAM::GPUExecutor gpuExec {};
        NeoFOAM::CPUExecutor cpuExec {};

        NeoFOAM::Field<NeoFOAM::scalar> GPUField(gpuExec, 10);
        NeoFOAM::Field<NeoFOAM::scalar> CPUField(cpuExec, 10);


The `Executor` is a std::variant

.. code-block:: cpp

    using executor = std::variant<OMPExecutor, GPUExecutor, CPUExecutor>;

and allows to switch between the different strategies for memory allocation and execution at runtime. We use `std::visit` to switch between the different strategies:

.. code-block:: cpp

    NeoFOAM::CPUExecutor exec{};
    std::visit([&](const auto& exec)
               { Functor(exec); },
               exec);

that are provided by a functor

.. code-block:: cpp

    struct Functor
    {
        void operator()(const CPUExecutor& exec)
        {
            std::cout << "CPUExecutor" << std::endl;
        }

        void operator()(const OMPExecutor& exec)
        {
            std::cout << "OMPExecutor" << std::endl;
        }

        void operator()(const GPUExecutor& exec)
        {
            std::cout << "GPUExecutor" << std::endl;
        }
    };

The visit pattern with the above functor would print different messages depending on the executor type. To extend the library with the additional features the above functor design should be used for the different implementations.

One can check that two operators are 'of the same type', i.e. execute in the same execution space using the equality operators `==` and `!=`.