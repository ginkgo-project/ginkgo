# Ginkgo for casopt generated matrices - Electrostatic field simulation

## Structure

- `electrostatic-abb.cpp`: to run ginkgo solver instance for a given matrix file

- `utils.hpp`: helper functions defined by user, handle input/output, display progres ...

- `/data`: sub directory for example matrix file
 
## Usage

After building, the executable `electrostatic-abb` should be generated in the build folder

the working directory should include a config file, config file name set to `electro.config` can be changed in the code.

config file format:

```
<executor>
<solver>
<problem name>
<input mode>
```
**Executor**: module to be used by ginkgo - default reference

**algorithm**: solver, supported so far are gmres and bicgstab - default gmres

**problem name**: name of problem to be solved, a matching file with name  *<problem_name>* with extension either or *".amtx"* for ASCII file format or *"bmtx"* for binary format. 

The matching file should be found in /data. Check field input mode on how to specify which reader to use - default sphere

**input mode**: ascii or binary, depending on type of matrix file to read - default ascii

## Further notes

- Todo: verify double and single precision. Not consistant in input files due to fortran print.




