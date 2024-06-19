# `gko::experimental::mpi::communicator`

- thin wrapper
- cheap to copy
- convert `MPI_Comm` to communicator
- wraps point-to-point and collective operations
  - refer to API for full list
- type alias `comm_index_type`

- mapping mpi rank to gpu device id

# MPI Environment

- scope guard for mpi environment