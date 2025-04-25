# generates the API documentation
function(ginkgo_doc_gen)
    set(GINKGO_DOXYGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    set(GINKGO_DOXYGEN_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/header.html ${CMAKE_CURRENT_BINARY_DIR}/header.html COPYONLY)
    file(GLOB doxygen_depend
        ${GINKGO_DOXYGEN_DIR}/headers/*.hpp
        ${Ginkgo_SOURCE_DIR}/include/ginkgo/**/*.hpp
        )
    list(APPEND doxygen_depend
        ${Ginkgo_BINARY_DIR}/include/ginkgo/config.hpp
        ${Ginkgo_BINARY_DIR}/include/ginkgo/ginkgo.hpp
        )
    list(APPEND doxygen_depend
        ${CMAKE_CURRENT_BINARY_DIR}/examples/examples.hpp
        )
    FILE(GLOB _ginkgo_examples
        ${Ginkgo_SOURCE_DIR}/examples/*.hpp
        )
    LIST(APPEND doxygen_depend ${_ginkgo_examples})
    add_custom_target(doxygen ALL
        COMMAND "${DOXYGEN_EXECUTABLE}" ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${GINKGO_DOXYGEN_BUILD_DIR}
        DEPENDS
        examples
        ${CMAKE_CURRENT_SOURCE_DIR}/mainpage.md
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        ${doxygen_depend}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
        )
    if(GINKGO_DOC_GENERATE_PDF)
        add_custom_command(TARGET "doxygen" POST_BUILD
            COMMAND make
            COMMAND "${CMAKE_COMMAND}" -E copy refman.pdf
            "${CMAKE_CURRENT_BINARY_DIR}/${name}.pdf"
            WORKING_DIRECTORY "${GINKGO_DOXYGEN_BUILD_DIR}/latex"
            DEPENDS doxygen
            COMMENT "Generating ${name} PDF from LaTeX"
            VERBATIM
        )
    endif()
endfunction()
