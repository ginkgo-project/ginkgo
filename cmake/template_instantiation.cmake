function(add_instantiation_files source_dir source_file output_files_var)
    # if instantiation is disabled, compile the file directly
    if(NOT GINKGO_SPLIT_TEMPLATE_INSTANTIATIONS)
        set(${output_files_var} "${source_dir}/${source_file}" PARENT_SCOPE)
        return()
    endif()
    # read full file into variable
    set(source_path "${source_dir}/${source_file}")
    file(READ "${source_path}" file_contents)
    # escape semicolons and use them for line separation
    string(REPLACE ";" "<semicolon>" file_contents "${file_contents}")
    string(REGEX REPLACE "[\r\n]" ";" file_contents "${file_contents}")
    # find location of // begin|split|end comments
    set(begin_location)
    set(end_location)
    set(split_locations)
    list(LENGTH file_contents total_length)
    set(counter 0)
    foreach(line IN LISTS file_contents)
        if(line MATCHES "// begin")
            if(begin_location)
                message(FATAL_ERROR "Duplicate begin in line ${counter}, first found in ${begin_location}")
            endif()
            set(begin_location ${counter})
        elseif(line MATCHES "// split")
            if((NOT begin_location) OR end_location)
                message(FATAL_ERROR "Found split outside begin/end in line ${counter}")
            endif()
            list(APPEND split_locations ${counter})
        elseif(line MATCHES "// end")
            if(end_location)
                message(FATAL_ERROR "Duplicate end in line ${counter}, first found in ${end_location}")
            endif()
            set(end_location ${counter})
        endif()
        math(EXPR counter "${counter} + 1")
    endforeach()
    if (NOT (begin_location AND end_location AND split_locations))
        message(FATAL_ERROR "Nothing to split")
    endif()
    if (begin_location GREATER_EQUAL end_location)
        message(FATAL_ERROR "Incorrect begin/end order")
    endif()
    # determine which lines belong to the header and footer
    set(range_begins ${begin_location} ${split_locations})
    set(range_ends ${split_locations} ${end_location})
    list(LENGTH split_locations range_count_minus_one)
    math(EXPR length_header "${begin_location}")
    math(EXPR end_location_past "${end_location} + 1")
    math(EXPR length_footer "${total_length} - ${end_location_past}")
    list(SUBLIST file_contents 0 ${length_header} header)
    list(SUBLIST file_contents ${end_location_past} ${length_footer} footer)
    set(output_files)
    # for each range between // begin|split|end pairs
    foreach(range RANGE 0 ${range_count_minus_one})
        # create an output filename
        string(REGEX REPLACE "(\.hip\.cpp|\.dp\.cpp|\.cpp|\.cu)$" ".${range}\\1" target_file "${source_file}")
        set(target_path "${CMAKE_CURRENT_BINARY_DIR}/${target_file}")
        list(APPEND output_files "${target_path}")
        # extract the range between the comments
        list(GET range_begins ${range} begin)
        list(GET range_ends ${range} end)
        math(EXPR begin "${begin} + 1")
        math(EXPR length "${end} - ${begin}")
        list(SUBLIST file_contents ${begin} ${length} content)
        # concatenate header, content and footer and turn semicolons into newlines
        string(REPLACE ";" "\n" content "${header};${content};${footer}")
        # and escaped semicolons into regular semicolons again
        string(REPLACE "<semicolon>" ";" content "${content}")
        # create a .tmp file, but only copy it over if source file changed
        # this way, we don't rebuild unnecessarily
        file(WRITE "${target_path}.tmp" "${content}")
        add_custom_command(
            OUTPUT "${target_path}"
            COMMAND ${CMAKE_COMMAND} -E copy "${target_path}.tmp" "${target_path}"
            MAIN_DEPENDENCY "${source_path}")
    endforeach()
    # make sure cmake gets called when the source file was updated
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_path}")
    set(${output_files_var} ${output_files} PARENT_SCOPE)
endfunction()
