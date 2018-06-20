set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_executable(asynchronous_stopping_criterion
	asynchronous_stopping_criterion.cpp byinteraction.cpp)
target_link_libraries(asynchronous_stopping_criterion ginkgo
	Threads::Threads)
target_include_directories(asynchronous_stopping_criterion
	PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
