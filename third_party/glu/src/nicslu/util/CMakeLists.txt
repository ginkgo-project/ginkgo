add_library(nicslu_util nicslu_readfile.c)

target_include_directories(nicslu_util PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include/")