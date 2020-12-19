function(ginkgo_add_object_library name)
    add_library(${name} OBJECT ${ARGN})
    ginkgo_compile_features(${name})
    ginkgo_default_includes(${name})
    target_include_directories(${name} PUBLIC ${Ginkgo_SOURCE_DIR})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

function(ginkgo_add_normal_library name)
    add_library(${name} ${ARGN})
    ginkgo_compile_features(${name})
    ginkgo_default_includes(${name})
    if(GINKGO_HAVE_HWLOC)
      target_include_directories(${name}
        PUBLIC 
        $<BUILD_INTERFACE:${HWLOC_INCLUDE_DIRS}>
        )
    target_link_libraries(${name} PUBLIC ${HWLOC_LIBRARIES})
    endif()
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endfunction()
  
ginkgo_add_normal_library(ginkgo_core_device machine_topology.cpp)
ginkgo_install_library(ginkgo_core_device core/devices)

add_subdirectory(cuda)
add_subdirectory(dpcpp)
add_subdirectory(hip)
add_subdirectory(omp)
add_subdirectory(reference)
