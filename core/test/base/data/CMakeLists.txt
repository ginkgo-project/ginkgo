add_custom_target(test_data)

function(copy_mtx name)
    add_custom_command(TARGET test_data POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/${name}.mtx"
        "${CMAKE_CURRENT_BINARY_DIR}/")
endfunction()

copy_mtx(dense_real)
copy_mtx(dense_integer)
copy_mtx(dense_complex)

copy_mtx(sparse_real)
copy_mtx(sparse_real_symmetric)
copy_mtx(sparse_real_skew_symmetric)
copy_mtx(sparse_complex)
copy_mtx(sparse_complex_hermitian)
copy_mtx(sparse_pattern)
