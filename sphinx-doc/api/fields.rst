.. _api_fields:

Fields
======

Overview
^^^^^^^^
The Field classes are the central elements for implementing a platform portable CFD framework. Fields should allow to perform basic algebraic operations such as binary operations like the addition or subtraction of two fields, or scalar operations like the multiplication of a field with a scalar.

In the following, we will explain the implementation details of the field operations using the additions operator as an example. The block of code below shows an example implementation of the addition operator.

.. code-block:: cpp

    [[nodiscard]] Field<T> operator+(const Field<T>& rhs)
    {
        Field<T> result(exec_, size_);
        result = *this;
        add(result, rhs);
        return result;
    }


Besides creating a temporary for the result it mainly calls the free standing ``add`` function which is implemented in ``FieldOperations.hpp``. This, in turn, dispatches to the ``addOp`` functor, that holds the actual addition kernels. In the case of addition this is implemented as a  ``Kokkos::parallel_for`` function, see `Kokkos documentation  <https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_for.html>`_ for more details.

.. code-block:: cpp

   using executor = typename Executor::exec;
   auto a_f = a.field();
   auto b_f = b.field();
   Kokkos::parallel_for(
      Kokkos::RangePolicy<executor>(0, a_f.size()),
      KOKKOS_CLASS_LAMBDA(const int i) { a_f[i] = a_f[i] + b_f[i]; }
   );

The code snippet also highlights another important aspect, the executor. The executor, here defines the ``Kokkos::RangePolicy``, see  `Kokkos Programming Model  <https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/ProgrammingGuide/ProgrammingModel.md>`_. Besides defining the RangePolicy, the executor also holds functions for allocating and deallocationg memory. A full example of using NeoFOAMs fields with a GPU executor could be implemented as

.. code-block:: cpp

    NeoFOAM::GPUExecutor GPUExec {};
    NeoFOAM::Field<NeoFOAM::scalar> GPUa(GPUExec, N);
    NeoFOAM::fill(GPUa, 1.0);
    NeoFOAM::Field<NeoFOAM::scalar> GPUb(GPUExec, N);
    NeoFOAM::fill(GPUb, 2.0);
    auto GPUc = GPUa + GPUb;

Interface
^^^^^^^^^

.. doxygenfile:: NeoFOAM/fields/Field.hpp
   :project: NeoFOAM
