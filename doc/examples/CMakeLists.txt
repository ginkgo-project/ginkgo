# Collect all of the directory names for the examples programs
FILE(GLOB _ginkgo_examples
    ${CMAKE_SOURCE_DIR}/examples/*
    )
LIST(REMOVE_ITEM _ginkgo_examples "${CMAKE_SOURCE_DIR}/examples/CMakeLists.txt")

ADD_CUSTOM_TARGET(examples)

file(GLOB _ginkgo_examples_tooltip
    ${CMAKE_SOURCE_DIR}/examples/*/doc/tooltip
    )

file(GLOB _ginkgo_examples_kind
    ${CMAKE_SOURCE_DIR}/examples/*/doc/kind
    )
file(GLOB _ginkgo_examples_buildson
    ${CMAKE_SOURCE_DIR}/examples/*/doc/builds-on
    )

ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/examples.hpp
    COMMAND ${PERL_EXECUTABLE}
    ARGS
    ${CMAKE_SOURCE_DIR}/doc/scripts/examples.pl
    ${CMAKE_CURRENT_SOURCE_DIR}/examples.hpp.in
    ${_ginkgo_examples}
    > ${CMAKE_CURRENT_BINARY_DIR}/examples.hpp
    DEPENDS
    ${CMAKE_SOURCE_DIR}/doc/scripts/examples.pl
    ${CMAKE_CURRENT_SOURCE_DIR}/examples.hpp.in
    ${_ginkgo_examples_tooltip}
    ${_ginkgo_examples_kind}
    ${_ginkgo_examples_buildson}
    )
ADD_CUSTOM_TARGET(build_examples_hpp
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/examples.hpp
    COMMENT
    "Building examples.hpp")
ADD_DEPENDENCIES(examples build_examples_hpp)

#
# Prepare the examples for documentation generation
#

FOREACH(example ${_ginkgo_examples})
    GET_FILENAME_COMPONENT(example "${example}" NAME)

    ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${example}.cpp
        COMMAND ${PERL_EXECUTABLE}
        ARGS
        ${CMAKE_SOURCE_DIR}/doc/scripts/program2plain
        < ${CMAKE_SOURCE_DIR}/examples/${example}/${example}.cpp
        > ${CMAKE_CURRENT_BINARY_DIR}/${example}.cpp
        DEPENDS
        ${CMAKE_SOURCE_DIR}/doc/scripts/program2plain
        ${CMAKE_SOURCE_DIR}/examples/${example}/${example}.cpp
        VERBATIM
        )

    ADD_CUSTOM_COMMAND(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${example}.hpp
        COMMAND ${PERL_EXECUTABLE}
        ARGS
        ${CMAKE_SOURCE_DIR}/doc/scripts/make_example.pl
        ${example} ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
        > ${CMAKE_CURRENT_BINARY_DIR}/${example}.hpp
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        ${CMAKE_SOURCE_DIR}/doc/scripts/make_example.pl
        ${CMAKE_SOURCE_DIR}/doc/scripts/intro2toc
        ${CMAKE_SOURCE_DIR}/doc/scripts/create_anchors
        ${CMAKE_SOURCE_DIR}/doc/scripts/program2doxygen
        ${CMAKE_SOURCE_DIR}/examples/${example}/${example}.cpp
        ${CMAKE_SOURCE_DIR}/examples/${example}/doc/intro.dox
        ${CMAKE_SOURCE_DIR}/examples/${example}/doc/results.dox
        )

    ADD_CUSTOM_TARGET(examples_${example}
        DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/${example}.hpp
        ${CMAKE_CURRENT_BINARY_DIR}/${example}.cpp
        COMMENT
        "Building doxygen input file for examples program <${example}>"
        )
    ADD_DEPENDENCIES(examples examples_${example})
ENDFOREACH()
