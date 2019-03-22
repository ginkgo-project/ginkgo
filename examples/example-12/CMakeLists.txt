set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_executable(example-12
	example-12.cpp)
target_link_libraries(example-12 ginkgo
	Threads::Threads)
target_include_directories(example-12
	PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
