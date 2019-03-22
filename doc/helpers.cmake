# configures the file <in> into the variable <variable>
function(ginkgo_configure_to_string in variable)
    set(fin "${in}")
    file(READ "${fin}" str)
    string(CONFIGURE "${str}" str_conf)
    set(${variable} "${str_conf}" PARENT_SCOPE)
endfunction()

# writes the concatenated configured files <in1,2>
# in <base_in> into <out>
function(ginkgo_doc_conf_concat base_in in1 in2 out)
    ginkgo_configure_to_string("${base_in}/${in1}" s1)
    ginkgo_configure_to_string("${base_in}/${in2}" s2)
    string(CONCAT so "${s1}" "\n" "${s2}")
    file(WRITE "${out}" "${so}")
endfunction()

# adds a pdflatex build step
function(ginkgo_doc_pdf name path)
    add_custom_command(TARGET "${name}" POST_BUILD
        COMMAND make
        COMMAND "${CMAKE_COMMAND}" -E copy refman.pdf
        "${CMAKE_CURRENT_BINARY_DIR}/${name}.pdf"
        WORKING_DIRECTORY "${path}"
        COMMENT "Generating ${name} PDF from LaTeX"
        VERBATIM
        )
endfunction()

macro(to_string variable)
  set(${variable} "")
  foreach(var  ${ARGN})
    set(${variable} "${${variable}} ${var}")
  endforeach()
  string(STRIP "${${variable}}" ${variable})
endmacro()


# generates the documentation named <name> with the additional
# config file <in> in <pdf/html> format
function(ginkgo_doc_gen name in pdf mainpage)
    set(DIR_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
    set(DIR_BASE "${CMAKE_SOURCE_DIR}")
    set(DOC_BASE "${CMAKE_CURRENT_SOURCE_DIR}")
    set(DIR_OUT "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    set(MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/pages/${mainpage}")
    set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-${name}")
    set(layout "${CMAKE_CURRENT_SOURCE_DIR}/DoxygenLayout.xml")
    set(doxygen_base_input
      "${CMAKE_CURRENT_SOURCE_DIR}/headers/ "
      )
    list(APPEND doxygen_base_input
      ${DIR_BASE}/include
      ${MAINPAGE}
      )
    if(GINKGO_DOC_GENERATE_EXAMPLES)
      list(APPEND doxygen_base_input
        ${CMAKE_CURRENT_BINARY_DIR}/examples/examples.hpp
        )
    endif()
    set(doxygen_dev_input
      "${DIR_BASE}/core"
      )
    list(APPEND doxygen_dev_input
      ${DIR_BASE}/omp
      ${DIR_BASE}/cuda
      ${DIR_BASE}/reference
      )
    set(doxygen_image_path "${CMAKE_CURRENT_SOURCE_DIR}/images/")
    file(GLOB doxygen_depend
      ${CMAKE_CURRENT_SOURCE_DIR}/headers/*.hpp
      ${CMAKE_SOURCE_DIR}/include/ginkgo/**/*.hpp
      )
    list(APPEND doxygen_depend
      ${CMAKE_BINARY_DIR}/include/ginkgo/config.hpp
      )
    if(GINKGO_DOC_GENERATE_EXAMPLES)
      list(APPEND doxygen_depend
        ${CMAKE_CURRENT_BINARY_DIR}/examples/examples.hpp
        )
      FILE(GLOB _ginkgo_examples
        ${CMAKE_SOURCE_DIR}/examples/example-*
        )
      FOREACH(_ex ${_ginkgo_examples})
        GET_FILENAME_COMPONENT(_ex "${_ex}" NAME)
        LIST(APPEND doxygen_depend
          ${CMAKE_CURRENT_BINARY_DIR}/examples/${_ex}.hpp
          # ${CMAKE_SOURCE_DIR}/examples/example-*/*.cpp
          )
        LIST(APPEND doxygen_base_input
          ${CMAKE_CURRENT_BINARY_DIR}/examples/${_ex}.hpp
          # ${CMAKE_SOURCE_DIR}/examples/example-*/*.cpp
          )
      ENDFOREACH()
      list(APPEND doxygen_dev_input
        ${doxygen_base_input}
        )
    endif()
    to_string(doxygen_base_input_str ${doxygen_base_input} )
    to_string(doxygen_dev_input_str ${doxygen_dev_input} )
    to_string(doxygen_image_path_str ${doxygen_image_path} )
    add_custom_target("${name}" ALL
      #DEPEND "${doxyfile}.stamp" Doxyfile.in ${in} ${in2}
        COMMAND "${DOXYGEN_EXECUTABLE}" ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        examples
        ${doxyfile}
        ${layout}
        ${doxygen_depend}
        #COMMAND "${CMAKE_COMMAND}" cmake -E touch "${doxyfile}.stamp"
        COMMENT "Generating ${name} documentation with Doxygen"
        VERBATIM
        )
    if(pdf)
        ginkgo_doc_pdf("${name}" "${DIR_OUT}")
    endif()
    ginkgo_doc_conf_concat("${CMAKE_CURRENT_SOURCE_DIR}/conf"
        Doxyfile.in "${in}" "${doxyfile}"
        )
endfunction()
