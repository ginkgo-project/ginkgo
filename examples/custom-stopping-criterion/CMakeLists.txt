set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_executable(custom-stopping-criterion
	custom-stopping-criterion.cpp)
target_link_libraries(custom-stopping-criterion ginkgo
	Threads::Threads)
target_include_directories(custom-stopping-criterion
	PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
