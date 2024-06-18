# `gko::experimental::distributed::Partition`

- vectors and matrices are row-wise distributed
- each row has a owning process
- represented by partition
  - image to show partition
- stored as collection of half-open intervals called ranges
- each range belongs to a part, i.e. process


# Creating

- partition can't be changed after created

## Helper Functions

- create from local sizes/ranges

# Accessing 

- only const accessors
- scalar query functions
  - get_size
  - get_num_ranges
  - get_num_parts
  - get_num_empty_parts
- array query functions
  - compressed range bounds
  - range starting indices
  - part id per range
  - part size