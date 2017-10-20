# Download and unpack googletest at configure time
configure_file("${CMAKE_CURRENT_LIST_DIR}/DownloadCMakeLists.txt.in"
               download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download)
if(result)
  message(FATAL_ERROR "CMake step for gtest/download failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download)
if(result)
  message(FATAL_ERROR "Build step for gtest/download failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
                 ${CMAKE_CURRENT_BINARY_DIR}/build)

