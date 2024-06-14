# Third Party Libraries and Packages

Ginkgo relies on third party packages in different cases. These third party
packages can be turned off by disabling the relevant options.

+ `-DGINKGO_BUILD_TESTS=ON`: Our tests are implemented with [Google
  Test](https://github.com/google/googletest);
+ `-DGINKGO_BUILD_BENCHMARKS=ON`: For argument management we use
  [gflags](https://github.com/gflags/gflags) and for JSON parsing we use
  [nlohmann-json](https://github.com/nlohmann/json);
+ `-DGINKGO_BUILD_HWLOC=ON`:
  [hwloc](https://www.open-mpi.org/projects/hwloc) to detect and control cores
  and devices.
+ `-DGINKGO_BUILD_HWLOC=ON` and `-DGINKGO_BUILD_TESTS=ON`:
  [libnuma](https://www.man7.org/linux/man-pages/man3/numa.3.html) is required
  when testing the functions provided through MachineTopology.
+ `-DGINKGO_BUILD_EXAMPLES=ON`:
  [OpenCV](https://opencv.org/) is required for some examples, they are disabled when OpenCV is not available.
+ `-DGINKGO_BUILD_DOC=ON`:
  [doxygen](https://www.doxygen.nl/) is required to build the documentation and
  additionally [graphviz](https://graphviz.org/) is required to build the class hierarchy graphs.
+ [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) is required
  when using the `NestedDissection` reordering functionality.
  If METIS is not found, the functionality is disabled.
+ [PAPI](https://icl.utk.edu/papi/) (>= 7.1.0) is required when using the `Papi` logger.
  If PAPI is not found, the functionality is disabled.

Ginkgo attempts to use pre-installed versions of these package if they match
version requirements using `find_package`. Otherwise, the configuration step
will download the files for each of the packages `GTest`, `gflags`, and
`nlohmann-json` and build them internally.

Note that, if the external packages were not installed to the default location,
the CMake option `-DCMAKE_PREFIX_PATH=<path-list>` needs to be set to the
semicolon (`;`) separated list of install paths of these external packages. For
more Information, see the [CMake documentation for
CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/v3.9/variable/CMAKE_PREFIX_PATH.html)
for details.

For convenience, the options `GINKGO_INSTALL_RPATH[_.*]` can be used
to bind the installed Ginkgo shared libraries to the path of its dependencies.
