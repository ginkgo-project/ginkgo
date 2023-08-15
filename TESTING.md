<!--
SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors

SPDX-License-Identifier: BSD-3-Clause
-->

Testing Instructions                            {#testing_ginkgo}
-------------------------------------
### Running the unit tests
You need to compile ginkgo with `-DGINKGO_BUILD_TESTS=ON` option to be able to
run the tests. 

#### Using make test
After configuring Ginkgo, use the following command inside the build folder to run all tests:

```sh
make test
```

The output should contain several lines of the form:

```
     Start  1: path/to/test
 1/13 Test  #1: path/to/test .............................   Passed    0.01 sec
```

To run only a specific test and see more details results (e.g. if a test failed)
run the following from the build folder:

```sh
./path/to/test
```

where `path/to/test` is the path returned by `make test`.

#### Using make quick_test
After compiling Ginkgo, use the following command inside the build folder to run
a small subset of tests that should execute quickly:

```sh
make quick_test
```

These tests do not use GPU features except for a few device property queries, so
they may still fail if Ginkgo was compiled with GPU support, but no such GPU is
available. The output is equivalent to `make test`.

#### Using CTest 
The tests can also be ran through CTest from the command line, for example when
in a configured build directory:

```sh 
ctest -T start -T build -T test -T submit
```

Will start a new test campaign (usually in `Experimental` mode), build Ginkgo
with the set configuration, run the tests and submit the results to our CDash
dashboard.


Another option is to use Ginkgo's CTest script which is configured to build
Ginkgo with default settings, runs the tests and submits the test to our CDash
dashboard automatically.

To run the script, use the following command:

```sh
ctest -S cmake/CTestScript.cmake
```

The default settings are for our own CI system. Feel free to configure the
script before launching it through variables or by directly changing its values.
A documentation can be found in the script itself.
