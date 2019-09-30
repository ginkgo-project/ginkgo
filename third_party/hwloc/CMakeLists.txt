ginkgo_load_git_package(hwloc_external
  "https://github.com/open-mpi/hwloc.git"
  "2e3206f388efa7996b4df5b5876e5a6580098b3e")
ginkgo_add_external_target(hwloc hwloc_ext build/include build/lib STATIC "_debug" hwloc_external FALSE)
