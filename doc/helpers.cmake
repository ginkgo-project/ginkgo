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

function(ginkgo_doc_generate_header_footer name path)
  set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-${name}")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/header.html
           ${CMAKE_CURRENT_BINARY_DIR}/footer.html
    COMMAND ${CMAKE_COMMAND} -E touch header.html
    COMMAND ${CMAKE_COMMAND} -E touch footer.html
    COMMAND ${DOXYGEN_EXECUTABLE} -w html header.html footer.html ${doxyfile}
    # COMMAND ${PERL_EXECUTABLE} -pi ${CMAKE_CURRENT_BINARY_DIR}/scripts/mod_header.pl header.html
    # COMMAND ${PERL_EXECUTABLE} -pi ${CMAKE_CURRENT_BINARY_DIR}/scripts/mod_footer.pl footer.html
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${doxyfile}
    # ${CMAKE_CURRENT_BINARY_DIR}/scripts/mod_header.pl
    # ${CMAKE_CURRENT_BINARY_DIR}/scripts/mod_footer.pl
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
    set(DIR_BASE "${CMAKE_CURRENT_SOURCE_DIR}/..")
    # set(REPO_BASE "${CMAKE_SOURCE_DIR}")
    set(REPO_BASE "${CMAKE_SOURCE_DIR}")
    set(DIR_OUT "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    set(MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/pages/${mainpage}")
    set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-${name}")
    set(header "${CMAKE_CURRENT_BINARY_DIR}/${name}/header.html")
    set(footer "${CMAKE_CURRENT_BINARY_DIR}/${name}/footer.html")
    set(layout "${CMAKE_CURRENT_SOURCE_DIR}/DoxygenLayout.xml")
    set(doxygen_input
      "${CMAKE_CURRENT_SOURCE_DIR}/headers/ "
      )
    list(APPEND doxygen_input
      ${REPO_BASE}/core
      ${REPO_BASE}/include
      ${REPO_BASE}/omp
      ${REPO_BASE}/cuda
      ${REPO_BASE}/reference
      ${MAINPAGE}
      # ${CMAKE_CURRENT_BINARY_DIR}/tutorial/tutorial.hpp
      )
    set(doxygen_image_path "${CMAKE_CURRENT_SOURCE_DIR}/images/")
    file(GLOB doxygen_depend
      ${CMAKE_CURRENT_SOURCE_DIR}/headers/*.hpp
      ${CMAKE_SOURCE_DIR}/include/ginkgo/**/*.hpp
      )
    list(APPEND doxygen_depend
      ${CMAKE_BINARY_DIR}/include/ginkgo/config.hpp
      # ${CMAKE_CURRRENT_BINARY_DIR}/tutorial/tutorial.hpp
      )
    to_string(doxygen_input_str ${doxygen_input} )
    to_string(doxygen_image_path_str ${doxygen_image_path} )
    add_custom_target("${name}" ALL
        #DEPEND "${doxyfile}.stamp" Doxyfile.in ${in} ${in2}
        COMMAND "${DOXYGEN_EXECUTABLE}" ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS
        ${doxyfile}
        # ${header}
        # ${footer}
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
    # add_custom_command(
    #   OUTPUT ${header}
    #          ${footer}
    #   COMMAND ${CMAKE_COMMAND} -E touch ${DIR_OUT}/header.html
    #   COMMAND ${CMAKE_COMMAND} -E touch ${DIR_OUT}/footer.html
    #   COMMAND ${DOXYGEN_EXECUTABLE} -w html header.html footer.html style.css Doxyfile-${name}
    #   COMMENT "Creating header and footers"
    #   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${name}
    #   DEPENDS ${doxyfile}
    #   )
endfunction()
