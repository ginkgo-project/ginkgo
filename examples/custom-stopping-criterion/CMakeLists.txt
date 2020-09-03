set(run_target "custom-stopping-criterion")
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_executable(${run_target}
	${run_target}.cpp)
target_link_libraries(${run_target} ginkgo
	Threads::Threads)
target_include_directories(${run_target}
	PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
