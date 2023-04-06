set(NON_CMAKE_PACKAGE_DOWNLOADER_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/DownloadNonCMakeCMakeLists.txt.in")


#   Load a package from the url provided and run configure (Non-CMake projects)
#
#   \param package_name     Name of the package
#   \param package_url      Url of the package
#   \param package_tag      Tag or version of the package to be downloaded.
#   \param working_dir      The directory where the configure/build should happen.
#   \param config_command   The command for the configuration step.
#
function(ginkgo_load_and_configure_package package_name package_url package_hash working_dir config_command)
    set(GINKGO_THIRD_PARTY_BUILD_TYPE "Debug")
    if (CMAKE_BUILD_TYPE MATCHES "[Rr][Ee][Ll][Ee][Aa][Ss][Ee]")
        set(GINKGO_THIRD_PARTY_BUILD_TYPE "Release")
    endif()
    configure_file(${NON_CMAKE_PACKAGE_DOWNLOADER_SCRIPT}
        download/CMakeLists.txt)
    set(TOOLSET "")
    if (NOT "${CMAKE_GENERATOR_TOOLSET}" STREQUAL "")
        set(TOOLSET "-T${CMAKE_GENERATOR_TOOLSET}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" "${TOOLSET}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download)
    if(result)
        message(FATAL_ERROR
            "CMake step for ${package_name}/download failed: ${result}")
        return()
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/download)
    if(result)
        message(FATAL_ERROR
            "Build step for ${package_name}/download failed: ${result}")
        return()
    endif()
endfunction()


#   Download a file and verify the download
#
#   \param url          The url of file to be downloaded
#   \param filename     The name of the file
#   \param hash_type    The type of hash, See CMake file() documentation for more details.
#   \param hash         The hash itself, See CMake file() documentation for more details.
#
function(ginkgo_download_file url filename hash_type hash)
    file(DOWNLOAD ${url} ${filename}
        TIMEOUT 60  # seconds
        EXPECTED_HASH "${hash_type}=${hash}"
        TLS_VERIFY ON)
    if(EXISTS ${filename})
        message(STATUS "${filename} downloaded from ${url}")
    else()
        message(FATAL_ERROR "Download of ${filename} failed.")
    endif()
endfunction(ginkgo_download_file)
