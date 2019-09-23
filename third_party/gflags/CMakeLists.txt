ginkgo_load_git_package(gflags_external
    "https://github.com/gflags/gflags.git"
    "e171aa2d15ed9eb17054558e0b3a6a413bb01067"
    "-DGFLAGS_BUILD_TESTING=OFF" "-DGFLAGS_BUILD_gflags_LIB=OFF"
    "-DGFLAGS_BUILD_gflags_nothreads_LIB=ON" "-DGFLAGS_BUILD_STATIC_LIBS=ON"
    "-DGFLAGS_BUILD_PACKAGING=OFF")
if(WIN32)
    ginkgo_add_external_target(gflags gflags_nothreads_static build/include build/lib STATIC "_debug" gflags_external FALSE)
else()
    ginkgo_add_external_target(gflags gflags_nothreads build/include build/lib STATIC "_debug" gflags_external FALSE)
endif()
