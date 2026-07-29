function(ginkgo_add_object_library name)
    add_library(${name} OBJECT ${ARGN})
    ginkgo_compile_features(${name})
    ginkgo_default_includes(${name})
    target_include_directories(${name} PUBLIC ${Ginkgo_SOURCE_DIR})
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

function(ginkgo_add_library name)
    add_library(${name} ${ARGN})
    ginkgo_compile_features(${name})
    ginkgo_default_includes(${name})
    if(GINKGO_HAVE_HWLOC)
        target_link_libraries(${name} PUBLIC hwloc)
    endif()
    set_target_properties(${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

ginkgo_add_library(ginkgo_device machine_topology.cpp device.cpp)
ginkgo_install_library(ginkgo_device)
generate_export_header(
    ginkgo_device
    EXPORT_FILE_NAME "${Ginkgo_BINARY_DIR}/include/ginkgo/export_device.hpp"
    BASE_NAME "GKO_DEVICE"
    INCLUDE_GUARD_NAME "GKO_PUBLIC_API_EXPORT_DEVICE_HPP_"
)
set_target_properties(
    ginkgo_device
    PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
)

add_subdirectory(cuda)
add_subdirectory(dpcpp)
add_subdirectory(hip)
add_subdirectory(omp)
add_subdirectory(reference)
