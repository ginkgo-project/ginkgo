# Ginkgo for casopt generated matrices - Electrostatic field simulation

## Structure

- `electrostatic-abb.cpp`: to run ginkgo solver instance for a given matrix file

- `utils.hpp`: helper functions defined by user, handle input/output, display progres ...

- `/data`: sub directory for example matrix file
 
## Usage

### Building and running
After building, the executable `electrostatic-abb` should be generated in the corresponding build folder

the working directory should include a config file, config file name set to `electro.config` can be changed in the code - check config file format for further description of the config file.

- config file format: multiple lines with format `'option' 'value'`

available options can set from the config file, not all options should be included, but then the default value will be used

*possible options*
```
<executor> <executor module (default reference)>
<solver> <algebraic solver (default gmres)>
<problem name> <name of existing .bmtx/.amtx file in data/ folderm >
<input mode> <binary/ascii depending on type of method used>
<writeResult> <true/false, write output file>
<initialGuess> <zero/rhs initial vector guess used>
```

### Chanding ValueType

In this example, the valueType is defined as runtime. To switch between double and float, that should be done in the source, and then recompiling/rebuilding the electrostatic module.




