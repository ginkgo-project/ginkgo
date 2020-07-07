if(MSVC)
    # cmake links dynamic runtime libraries by default in Visual Studio
    # use the ginkgo's flags to use the same runtime libraries as ginkgo
    ginkgo_load_git_package(gflags_external
        "https://github.com/gflags/gflags.git"
        "f7388c6655e699f777a5a74a3c9880b9cfaabe59"
        "-DGFLAGS_BUILD_TESTING=OFF" "-DGFLAGS_BUILD_gflags_LIB=OFF"
        "-DGFLAGS_BUILD_gflags_nothreads_LIB=ON" "-DGFLAGS_BUILD_STATIC_LIBS=ON"
        "-DGFLAGS_BUILD_PACKAGING=OFF" "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
        "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}" "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
        "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")
else()
    # There is a weird issue with Intel 19 and c++17 causing a linking error.
    # Use c++11 instead.
    set(INTEL19_STD_FIX "")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel" AND CMAKE_CXX_COMPILER_VERSION MATCHES "19")
        set(INTEL19_STD_FIX "-DCMAKE_CXX_FLAGS=-std=c++11")
    endif()
    ginkgo_load_git_package(gflags_external
        "https://github.com/gflags/gflags.git"
        "f7388c6655e699f777a5a74a3c9880b9cfaabe59"
        "-DGFLAGS_BUILD_TESTING=OFF" "-DGFLAGS_BUILD_gflags_LIB=OFF"
        "-DGFLAGS_BUILD_gflags_nothreads_LIB=ON" "-DGFLAGS_BUILD_STATIC_LIBS=ON"
        "-DGFLAGS_BUILD_PACKAGING=OFF"
        "${INTEL19_STD_FIX}"
        )
endif()
if(WIN32)
    # gflags uses gflags_nothreads_static not gflags_nothreads_static in Windows.
    ginkgo_add_external_target(gflags gflags_nothreads_static build/include build/lib
        STATIC "_debug" gflags_external FALSE)
else()
    ginkgo_add_external_target(gflags gflags_nothreads build/include build/lib
        STATIC "_debug" gflags_external FALSE)
endif()
