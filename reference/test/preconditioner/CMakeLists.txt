ginkgo_create_test(batch_jacobi_kernels)
ginkgo_create_test(gauss_seidel)
ginkgo_create_test(ilu)
ginkgo_create_test(ic)
ginkgo_create_test(isai_kernels)
# need to split the isai_kernels due to NVHPC compilation limitation
if(
    CMAKE_CXX_COMPILER_ID MATCHES "PGI|NVHPC"
    AND (GINKGO_ENABLE_HALF OR GINKGO_ENABLE_BFLOAT16)
)
    ginkgo_create_test(isai_kernels EXECUTABLE_NAME isai_kernels_float16)
    ginkgo_build_test_name(
        isai_kernels
        isai_half_target
        EXECUTABLE_NAME isai_kernels_float16
    )
    target_compile_definitions(${isai_half_target} PRIVATE "NVHPC_FLOAT16")
endif()
ginkgo_create_test(jacobi)
ginkgo_create_test(jacobi_kernels)
ginkgo_create_test(sor_kernels)
