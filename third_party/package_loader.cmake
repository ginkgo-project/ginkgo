set(PACKAGE_DOWNLOADER_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/DownloadCMakeLists.txt.in")

function(_load_git_package_impl package_name package_url package_tag)
    configure_file(${PACKAGE_DOWNLOADER_SCRIPT}
                   download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download)
    if(result)
      message(FATAL_ERROR
        "CMake step for ${package_name}/download failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download)
    if(result)
      message(FATAL_ERROR
        "Build step for ${package_name}/download failed: ${result}")
    endif()
endfunction()

# needs to be a macro, so parent settings from the package's CMakeLists.txt get
# propagated to the scope where this is called
macro(load_git_package package_name package_url package_tag)
    # Download and unpack package at configure time
    _load_git_package_impl("${package_name}" "${package_url}" "${package_tag}")
    # Add package to the build
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/src
                     ${CMAKE_CURRENT_BINARY_DIR}/build)
endmacro()
