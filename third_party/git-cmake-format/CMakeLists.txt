# Download and unpack git-cmake-format at configure time
configure_file("${CMAKE_CURRENT_LIST_DIR}/DownloadCMakeLists.txt.in"
               download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download)
if(result)
  message(FATAL_ERROR "CMake step for git-cmake-format/download failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download)
if(result)
  message(FATAL_ERROR "Build step for git-cmake-format/download failed: ${result}")
endif()


# Add git-cmake-format directly to our build. This defines
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
                 ${CMAKE_CURRENT_BINARY_DIR}/build)

