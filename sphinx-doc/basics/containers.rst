.. _basics_containers:

containers
==========

Fields
^^^^^^

The Field classes are the central elements for implementing a platform portable CFD framework. Fields are able to perform basic algebraic operations such as addition or subtraction of two fields, or scalar operations like the multiplication of a field with a scalar. The Field classes store the data in a platform-independent way and the executor, which is used to dispatch the operations to the correct device. The Field classes are implemented in the ``Field.hpp`` header file and mainly store a pointer to the data, the size of the data, and the executor.

.. doxygenclass:: NeoFOAM::Field
    :members: size_, data_, exec_

To run a function on the GPU, the data and function need to be trivially copyable. This is not the case with the existing OpenFOAM Field class , and it can be viewed as a wrapper around the data. To solve this issue, the  NeoFOAM field class has a public member function called ``field()`` that returns a span that can be used to access the data on the CPU and GPU.

.. doxygenclass:: NeoFOAM::Field
    :members: field

The following example shows how to use the field function to access the data of a field and set all the values to 1.0. However, the for loop is only executed on the single CPU core and not on the GPU.

.. code-block:: cpp

     NeoFOAM::CPUExecutor cpuExec {};
     NeoFOAM::Field<NeoFOAM::scalar> a(cpuExec, size);
     std::span<double> sA = a.field();
     // for loop
     for (int i = 0; i < sA.size(); i++)
     {
          sA[i] = 1.0;
     }

To run the for loop on the GPU is a bit more complicated and is based on the Kokkos library that simplifies the process and support parallelization strategies for different GPU vendors and  OpenMP  support for CPU targets. The following example shows how to set all the values of a field to 1.0 on the GPU.

.. code-block:: cpp

     NeoFOAM::GPUExecutor gpuExec {};
     NeoFOAM::Field<NeoFOAM::scalar> b(gpuExec, size);
     std::span<double> sB = b.field();
     Kokkos::parallel_for(
          Kokkos::RangePolicy<gpuExec::exec>(0, sB.size()),
          KOKKOS_LAMBDA(const int i) { sB[i] = 1.0; }
     );

Kokkos requires the knowledge of where to run the code and the range of the loop. The range is defined by the size of the data and the executor. The `KOKKOS_LAMBDA` is required to mark the function so it is also compiled for the GPU. This approach however is not very user-friendly and requires the knowledge of the Kokkos library. To simplify the process, the Field class stores the executor and the field independent of the device can be set to 1.0 with the following code.

.. code-block:: cpp

     NeoFOAM::Field<NeoFOAM::scalar> c(gpuExec, size);
     NeoFOAM::fill(b, 10.0);

The fill function uses the `std::visit` function to call the correct function based on the executor as described in the previous section.

.. code-block:: cpp

    Operation op{};
    std::visit([&](const auto& exec)
               { op(exec); },
               exec);


.. note::

     TODO
     organize the FieldOperation so the can be easily shown here


FieldGraph
^^^^^^^^^^

The Field can now be used to compose  more complex data structures. To solve PDE's, information about the neighbors is required. This is usually done with the following approach:


.. code-block:: cpp

     int nCells = 3;
     std::vector<std::vector<int> > cellToCellStencil(nCells);

     cellToCellStencil.push_back({1, 2, 3});
     cellToCellStencil.push_back({4, 5, 6});
     cellToCellStencil.push_back({7, 8, 9});

     for (for auto& cell : cellToCellStencil)
     {
          for (auto& neibour : cell)
          {
               std::cout << neibour << " ";
          }
          std::cout << std::endl;
     }


Now we can loop over each cell and access the neighbors with a nested for loop. However, this approach is not suited for GPUs. Instead of the  vector of vector approach, the neighbor hood  is stored  with two fields (described with std::vector to simplify the example):

.. code-block:: cpp

     int nCells = 3;
     std::vector<int> value = {1, 2, 3, 4, 5, 6, 7, 8, 9};
     std::vector<int> offset_ = {0, 3, 6, 9};

     for (int i = 0; i < nCells ; i++)
     {
          int start = offset_[i];
          int end = offset_[i+1];
          for (int j = start; j < end; j++)
          {
               int neibour = value[j];
               std::cout << neibour << " ";
          }
          std::cout << std::endl;
     }

The same approach is used in the ``FieldGraph`` class (we had a better name for this but i forgot). That implements the above approach using the Field class.

.. note::

     TODO
     implement the FieldGraph class

BoundaryFields
^^^^^^^^^^^^^^

The BoundaryFields class is used to store the boundary conditions of a field. The BoundaryFields class is implemented in the ``BoundaryFields.hpp`` header file and store the boundary conditions in a general container that can be used to present different boundary conditions: Mixed, Dirichlet, Neumann. The class uses the same of set approach to loop over the boundary patches

.. note::

     TODO
     implement the boundaryFields see other commit

.. doxygenclass:: NeoFOAM::BoundaryFields
    :members:
        value_
        refValue_
        valueFraction_
        refGrad_
        boundaryTypes_
        offset_
        nBoundaries_
        nBoundaryFaces_


DomainField
^^^^^^^^^^^

The domainField stores the internalField and the boundaryFields in a single container and is used to represent all the relevant values of a fields for a given mesh.

.. note::

     TODO
     implement the DomainField see other commit

.. doxygenclass:: NeoFOAM::DomainField
    :members:
