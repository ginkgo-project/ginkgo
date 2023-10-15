cmake_minimum_required(VERSION 3.13)
project(batched-solver-from-files)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

add_executable(batched-solver-from-files batched-solver-from-files.cpp)
target_link_libraries(batched-solver-from-files Ginkgo::ginkgo)

foreach(ibatch RANGE 0 5)
    configure_file("data/isooctane/${ibatch}/A.mtx" "data/isooctane/${ibatch}/A.mtx" COPYONLY)
    configure_file("data/isooctane/${ibatch}/b.mtx" "data/isooctane/${ibatch}/b.mtx" COPYONLY)
    configure_file("data/isooctane/${ibatch}/S.mtx" "data/isooctane/${ibatch}/S.mtx" COPYONLY)
endforeach()
foreach(ibatch RANGE 0 3)
    configure_file("data/gri12/${ibatch}/A.mtx" "data/gri12/${ibatch}/A.mtx" COPYONLY)
    configure_file("data/gri12/${ibatch}/b.mtx" "data/gri12/${ibatch}/b.mtx" COPYONLY)
    configure_file("data/gri12/${ibatch}/S.mtx" "data/gri12/${ibatch}/S.mtx" COPYONLY)
endforeach()
