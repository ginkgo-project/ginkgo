# Building Ginkgo in Windows

Depending on the configuration settings, some manual work might be required:
* Build Ginkgo with Debug mode:
  Some Debug build specific issues can appear depending on the machine and environment:
  When you encounter the error message `ld: error: export ordinal too large`, add the compilation flag `-O1`
  by adding `-DCMAKE_CXX_FLAGS=-O1` to the CMake invocation.
* Build Ginkgo in _MinGW_:\
  If encountering the issue `cc1plus.exe: out of memory allocating 65536 bytes`, please follow the workaround in
  [reference](https://www.intel.com/content/www/us/en/programmable/support/support-resources/knowledge-base/embedded/2016/cc1plus-exe--out-of-memory-allocating-65536-bytes.html),
  or trying to compile ginkgo again might work.
