ginkgo_load_git_package(gflags_external
    "https://github.com/gflags/gflags.git"
    "660603a3df1c400437260b51c55490a046a12e8a"
    "-DBUILD_TESTING=OFF" "-DBUILD_gflags_LIB=OFF" "-DBUILD_gflags_nothreads_LIB=ON" "-DBUILD_STATIC_LIBS=ON")
ginkgo_add_external_target(gflags_nothreads build/include build/lib STATIC "" gflags_external FALSE)
add_library(gflags ALIAS gflags_nothreads)
