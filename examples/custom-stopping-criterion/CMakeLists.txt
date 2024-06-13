cmake_minimum_required(VERSION 3.16)
project(custom-stopping-criterion)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
endif()

add_executable(custom-stopping-criterion
	custom-stopping-criterion.cpp)

target_link_libraries(custom-stopping-criterion Ginkgo::ginkgo
	Threads::Threads)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
