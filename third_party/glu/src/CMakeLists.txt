add_subdirectory(preprocess)
add_subdirectory(nicslu)

add_library(glu 
            Timer.cpp 
            symbolic.cc 
            numeric.cu
            preprocess/preprocess.c
            nicslu/source/nicslu_aegraph.c
            nicslu/source/nicslu_amd.c
            nicslu/source/nicslu_analyze.c
            nicslu/source/nicslu_check.c
            nicslu/source/nicslu_cond.c
            nicslu/source/nicslu_destroy.c
            nicslu/source/nicslu_dumpa.c
            nicslu/source/nicslu_dumplu.c
            nicslu/source/nicslu_etree.c
            nicslu/source/nicslu_flops.c
            nicslu/source/nicslu_global.c
            nicslu/source/nicslu_initialize.c
            nicslu/source/nicslu_internal.c
            nicslu/source/nicslu_kernel.c
            nicslu/source/nicslu_kernel_fact.c
            nicslu/source/nicslu_kernel_fact_parallel.c
            nicslu/source/nicslu_kernel_parallel.c
            nicslu/source/nicslu_matrix.c
            nicslu/source/nicslu_memuse.c
            nicslu/source/nicslu_mergedup.c
            nicslu/source/nicslu_mc64.c
            nicslu/source/nicslu_mc64scale.c
            nicslu/source/nicslu_refact.c
            nicslu/source/nicslu_refact_parallel.c
            nicslu/source/nicslu_refine.c
            nicslu/source/nicslu_residual.c
            nicslu/source/nicslu_scale.c
            nicslu/source/nicslu_scheduler.c
            nicslu/source/nicslu_solve.c
            nicslu/source/nicslu_sort.c
            nicslu/source/nicslu_static_symbolic.c
            nicslu/source/nicslu_thread.c
            nicslu/source/nicslu_thread_proc.c
            nicslu/source/nicslu_throughput.c
            nicslu/source/nicslu_transpose.c
            nicslu/source/system.c
            nicslu/source/thread.c
            nicslu/source/timer_c.c
            nicslu/util/nicslu_readfile.c)

set_target_properties(glu PROPERTIES CUDA_ARCHITECTURES "70")

target_include_directories(glu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include/" "${CMAKE_CURRENT_SOURCE_DIR}/nicslu/include/")