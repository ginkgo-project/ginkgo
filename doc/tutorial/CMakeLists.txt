# Collect all of the directory names for the tutorial programs
FILE(GLOB _ginkgo_examples
  ${CMAKE_SOURCE_DIR}/examples/*
  )

ADD_CUSTOM_TARGET(tutorial)

ADD_CUSTOM_TARGET(build_tutorial_h
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tutorial.h
    COMMENT
      "Building tutorial.h")
ADD_DEPENDENCIES(tutorial build_tutorial_h)


#
# Prepare the steps for documentation generation
#

FOREACH(_ex ${_ginkgo_examples})
  GET_FILENAME_COMPONENT(_ex "${_ex}" NAME)

  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_ex}.cpp
    COMMAND ${PERL_EXECUTABLE}
    ARGS
      ${CMAKE_SOURCE_DIR}/doc/doxygen/scripts/program2plain
      < ${CMAKE_SOURCE_DIR}/examples/${_ex}/${_ex}.cc
      > ${CMAKE_CURRENT_BINARY_DIR}/${_ex}.cc
    DEPENDS
      ${CMAKE_SOURCE_DIR}/doc/doxygen/scripts/program2plain
      ${CMAKE_SOURCE_DIR}/examples/${_ex}/${_ex}.cc
    VERBATIM
    )

  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_ex}.h
    COMMAND ${PERL_EXECUTABLE}
    ARGS
      ${CMAKE_SOURCE_DIR}/doc/doxygen/scripts/make_ex.pl
      ${_ex} ${CMAKE_SOURCE_DIR}
      > ${CMAKE_CURRENT_BINARY_DIR}/${_ex}.h
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
      ${CMAKE_SOURCE_DIR}/doc/doxygen/scripts/make_ex.pl
      ${CMAKE_SOURCE_DIR}/doc/doxygen/scripts/intro2toc
      ${CMAKE_SOURCE_DIR}/doc/doxygen/scripts/create_anchors
      ${CMAKE_SOURCE_DIR}/doc/doxygen/scripts/program2doxygen
      ${CMAKE_SOURCE_DIR}/examples/${_ex}/${_ex}.cc
      ${CMAKE_SOURCE_DIR}/examples/${_ex}/doc/intro.dox
      ${CMAKE_SOURCE_DIR}/examples/${_ex}/doc/results.dox
    )

  ADD_CUSTOM_TARGET(tutorial_${_ex}
    DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/${_ex}.h
      ${CMAKE_CURRENT_BINARY_DIR}/${_ex}.cc
      COMMENT
        "Building doxygen input file for tutorial program <${_ex}>"
    )
  ADD_DEPENDENCIES(tutorial tutorial_${_ex})
ENDFOREACH()
