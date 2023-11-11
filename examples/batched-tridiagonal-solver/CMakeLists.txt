cmake_minimum_required(VERSION 3.9)
project(batched-tridiagonal-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

add_executable(batched-tridiagonal-solver batched-tridiagonal-solver.cpp)
target_link_libraries(batched-tridiagonal-solver Ginkgo::ginkgo)

foreach(ibatch RANGE 0 1)
    configure_file("data/gallery_lesp_100/${ibatch}/A.mtx" "data/gallery_lesp_100/${ibatch}/A.mtx" COPYONLY)
    configure_file("data/gallery_lesp_200/${ibatch}/A.mtx" "data/gallery_lesp_200/${ibatch}/A.mtx" COPYONLY)
endforeach()


