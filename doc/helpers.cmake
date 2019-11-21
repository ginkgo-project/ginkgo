# configures the file <in> into the variable <variable>
function(ginkgo_configure_to_string in variable)
    set(fin "${in}")
    file(READ "${fin}" str)
    string(CONFIGURE "${str}" str_conf)
    set(${variable} "${str_conf}" PARENT_SCOPE)
endfunction()

macro(ginkgo_to_string variable)
  set(${variable} "")
  foreach(var  ${ARGN})
    set(${variable} "${${variable}} ${var}")
  endforeach()
  string(STRIP "${${variable}}" ${variable})
endmacro()

# writes the concatenated configured files <in1,2>
# in <base_in> into <out>
function(ginkgo_file_concat base_path in1 in2 out)
    ginkgo_configure_to_string("${base_path}/${in1}" s1)
    ginkgo_configure_to_string("${base_path}/${in2}" s2)
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


# generates the documentation named <name> with the additional
# config file <in> in <pdf/html> format
function(ginkgo_doc_gen name in pdf mainpage-in)
    set(DIR_BASE "${CMAKE_SOURCE_DIR}")
    set(DOC_BASE "${CMAKE_CURRENT_SOURCE_DIR}")
    set(DIR_SCRIPT "${DOC_BASE}/scripts")
    set(DIR_OUT "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    set(MAINPAGE "${DIR_OUT}/MAINPAGE-${name}.md")
    set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-${name}")
    set(layout "${DOC_BASE}/DoxygenLayout.xml")
    ginkgo_file_concat("${DOC_BASE}/pages"
      "${mainpage-in}" BASE_DOC.md "${MAINPAGE}"
      )
    set(doxygen_base_input
      "${DOC_BASE}/headers/"
      )
    list(APPEND doxygen_base_input
      ${CMAKE_BINARY_DIR}/include/ginkgo/config.hpp
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
      ${DIR_BASE}/hip
      ${DIR_BASE}/reference
      )
    set(doxygen_image_path "${CMAKE_SOURCE_DIR}/doc/images/")
    file(GLOB doxygen_depend
      ${DOC_BASE}/headers/*.hpp
      ${DIR_BASE}/include/ginkgo/**/*.hpp
      )
    list(APPEND doxygen_depend
      ${CMAKE_BINARY_DIR}/include/ginkgo/config.hpp
      )
    if(GINKGO_DOC_GENERATE_EXAMPLES)
      list(APPEND doxygen_depend
        ${CMAKE_CURRENT_BINARY_DIR}/examples/examples.hpp
        )
      FILE(GLOB _ginkgo_examples
        ${DIR_BASE}/examples/*
        )
      LIST(REMOVE_ITEM _ginkgo_examples "${DIR_BASE}/examples/CMakeLists.txt")
      FOREACH(_ex ${_ginkgo_examples})
        GET_FILENAME_COMPONENT(_ex "${_ex}" NAME)
        LIST(APPEND doxygen_depend
          ${CMAKE_CURRENT_BINARY_DIR}/examples/${_ex}.hpp
          )
        LIST(APPEND doxygen_base_input
          ${CMAKE_CURRENT_BINARY_DIR}/examples/${_ex}.hpp
          )
      ENDFOREACH()
    endif()
    list(APPEND doxygen_dev_input
      ${doxygen_base_input}
      )
    # pick some markdown files we want as pages
    set(doxygen_markdown_files "../../INSTALL.md ../../TESTING.md ../../BENCHMARKING.md")
    ginkgo_to_string(doxygen_base_input_str ${doxygen_base_input} )
    ginkgo_to_string(doxygen_dev_input_str ${doxygen_dev_input} )
    ginkgo_to_string(doxygen_image_path_str ${doxygen_image_path} )
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
    ginkgo_file_concat("${DOC_BASE}/conf"
        Doxyfile.in "${in}" "${doxyfile}"
        )
endfunction()
