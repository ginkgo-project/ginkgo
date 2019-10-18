if(MSVC)
    # cmake links dynamic runtime libraries by default in Visual Studio
    # use the ginkgo's flags to use the same runtime libraries as ginkgo
    ginkgo_load_git_package(gflags_external
        "https://github.com/gflags/gflags.git"
        "0b7f8db2c6b1b0b2451da0923a9ab09cc610e8d1"
        "-DGFLAGS_BUILD_TESTING=OFF" "-DGFLAGS_BUILD_gflags_LIB=OFF"
        "-DGFLAGS_BUILD_gflags_nothreads_LIB=ON" "-DGFLAGS_BUILD_STATIC_LIBS=ON"
        "-DGFLAGS_BUILD_PACKAGING=OFF" "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
        "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}" "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
        "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")
else()
    ginkgo_load_git_package(gflags_external
        "https://github.com/gflags/gflags.git"
        "0b7f8db2c6b1b0b2451da0923a9ab09cc610e8d1"
        "-DGFLAGS_BUILD_TESTING=OFF" "-DGFLAGS_BUILD_gflags_LIB=OFF"
        "-DGFLAGS_BUILD_gflags_nothreads_LIB=ON" "-DGFLAGS_BUILD_STATIC_LIBS=ON"
        "-DGFLAGS_BUILD_PACKAGING=OFF")
endif()
if(WIN32)
    # gflags uses gflags_nothreads_static not gflags_nothreads_static in Windows.
    ginkgo_add_external_target(gflags gflags_nothreads_static build/include build/lib STATIC "_debug" gflags_external FALSE)
else()
    ginkgo_add_external_target(gflags gflags_nothreads build/include build/lib STATIC "_debug" gflags_external FALSE)
endif()
