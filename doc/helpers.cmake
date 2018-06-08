# configures the file <in> into the variable <variable>
function(configure_to_string in variable)
  set(fin "${in}")
  file(READ "${fin}" str)
  string(CONFIGURE "${str}" str_conf)
  set(${variable} "${str_conf}" PARENT_SCOPE)
endfunction()

# writes the concatenated configured files <in1,2,3>
# in <base_in> into <out>
function(doc_conf_concat base_in in1 in2 in3 out)
  configure_to_string("${base_in}/${in1}" s1)
  configure_to_string("${base_in}/${in2}" s2)
  configure_to_string("${base_in}/${in3}" s3)
  string(CONCAT so "${s1}" "${s2}" "${s3}")
  file(WRITE "${out}" "${so}")
endfunction()

# adds a pdflatex build step
function(doc_pdf name path)
  add_custom_command(TARGET ${name} POST_BUILD
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
function(doc_gen name in pdf)
  set(DIR_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
  set(DIR_BASE "${CMAKE_CURRENT_SOURCE_DIR}/..")
  set(DIR_OUT "${CMAKE_CURRENT_BINARY_DIR}/${name}")
  set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-${name}")
  add_custom_target(${name} ALL
    #DEPEND "${doxyfile}.stamp" Doxyfile.in ${in} ${in2}
    COMMAND "${DOXYGEN_EXECUTABLE}" "${doxyfile}"
    #COMMAND "${CMAKE_COMMAND}" cmake -E touch "${doxyfile}.stamp"
    COMMENT "Generating ${name} documentation with Doxygen"
    VERBATIM
    )
  if(pdf)
    set(in2 "Doxyfile-pdf.in")
    doc_pdf(${name} "${DIR_OUT}")
  else()
    set(in2 "Doxyfile-html.in")
  endif()
  doc_conf_concat("${CMAKE_CURRENT_SOURCE_DIR}"
    Doxyfile.in ${in2} ${in} "${doxyfile}"
    )
endfunction()
