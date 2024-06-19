# `gko::experimental::distributed::Matrix`

- template parameters

## Distribution Pattern

- row-wise distributed
- local rows are split by column distribution
  - column distribution may be different than row distribution
- matrix entries with row and column index owned by the current process are called local
  - if column index owned by other processes non-local
- local and non-local entries are stored separately
- local in local matrix
- non-local in non-local matrix
- picture to show
- non-local matrix is compressed
  - no column is completely zero
  - requires reordering of columns
  - can't be inferred just by row and column partition


## Creating

- default create + read
  - changing matrix format 

## Applying

- apply to distr vectors
- collective communication
- has to be called on all processes in the communicator
- try to overlap comm with computation

## Accessing Local Data

- read only access to local + non-local

