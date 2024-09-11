# Collect all of the directory names for the examples programs
file(GLOB _ginkgo_examples ${PROJECT_SOURCE_DIR}/examples/*)
list(
    REMOVE_ITEM
    _ginkgo_examples
    "${PROJECT_SOURCE_DIR}/examples/CMakeLists.txt"
    "${PROJECT_SOURCE_DIR}/examples/build-setup.sh"
    "${PROJECT_SOURCE_DIR}/examples/compare-output.py"
)

add_custom_target(examples)

file(GLOB _ginkgo_examples_tooltip ${PROJECT_SOURCE_DIR}/examples/*/doc/tooltip)

file(GLOB _ginkgo_examples_kind ${PROJECT_SOURCE_DIR}/examples/*/doc/kind)
file(
    GLOB _ginkgo_examples_buildson
    ${PROJECT_SOURCE_DIR}/examples/*/doc/builds-on
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/examples.hpp
    COMMAND ${PERL_EXECUTABLE}
    ARGS
        ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/examples.pl
        ${CMAKE_CURRENT_SOURCE_DIR}/examples.hpp.in ${_ginkgo_examples} >
        ${CMAKE_CURRENT_BINARY_DIR}/examples.hpp
    DEPENDS
        ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/examples.pl
        ${CMAKE_CURRENT_SOURCE_DIR}/examples.hpp.in
        ${_ginkgo_examples_tooltip}
        ${_ginkgo_examples_kind}
        ${_ginkgo_examples_buildson}
)
add_custom_target(
    build_examples_hpp
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/examples.hpp
    COMMENT "Building examples.hpp"
)
add_dependencies(examples build_examples_hpp)

#
# Prepare the examples for documentation generation
#

foreach(example ${_ginkgo_examples})
    get_filename_component(example "${example}" NAME)

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${example}.cpp
        COMMAND ${PERL_EXECUTABLE}
        ARGS
            ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/program2plain <
            ${PROJECT_SOURCE_DIR}/examples/${example}/${example}.cpp >
            ${CMAKE_CURRENT_BINARY_DIR}/${example}.cpp
        DEPENDS
            ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/program2plain
            ${PROJECT_SOURCE_DIR}/examples/${example}/${example}.cpp
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${example}.hpp
        COMMAND ${PERL_EXECUTABLE}
        ARGS
            ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/make_example.pl
            ${example} ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR} >
            ${CMAKE_CURRENT_BINARY_DIR}/${example}.hpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
            ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/make_example.pl
            ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/intro2toc
            ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/create_anchors
            ${PROJECT_SOURCE_DIR}/doc/_doxygen/scripts/program2doxygen
            ${PROJECT_SOURCE_DIR}/examples/${example}/${example}.cpp
            ${PROJECT_SOURCE_DIR}/examples/${example}/doc/intro.dox
            ${PROJECT_SOURCE_DIR}/examples/${example}/doc/results.dox
    )

    add_custom_target(
        examples_${example}
        DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/${example}.hpp
            ${CMAKE_CURRENT_BINARY_DIR}/${example}.cpp
        COMMENT "Building doxygen input file for examples program <${example}>"
    )
    add_dependencies(examples examples_${example})
endforeach()
