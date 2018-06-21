function(ginkgo_add_object_library name)
    add_library(${name} OBJECT ${ARGN})
    target_include_directories(${name} PUBLIC ${PROJECT_SOURCE_DIR})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

add_subdirectory(omp)
add_subdirectory(gpu)
add_subdirectory(reference)
