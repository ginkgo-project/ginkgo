cmake_minimum_required(VERSION 3.9)
project(batched-band-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

add_executable(batched-band-solver batched-band-solver-1.cpp)
target_link_libraries(batched-band-solver Ginkgo::ginkgo)

foreach(ibatch RANGE 0 1)
    configure_file("data/banded_N_50_KL_6_KU_3/${ibatch}/A.mtx" "data/banded_N_50_KL_6_KU_3/${ibatch}/A.mtx" COPYONLY)
    configure_file("data/banded_N_100_KL_7_KU_8/${ibatch}/A.mtx" "data/banded_N_100_KL_7_KU_8/${ibatch}/A.mtx" COPYONLY)
    configure_file("data/banded_N_200_KL_25_KU_30/${ibatch}/A.mtx" "data/banded_N_200_KL_25_KU_30/${ibatch}/A.mtx" COPYONLY)
endforeach()


