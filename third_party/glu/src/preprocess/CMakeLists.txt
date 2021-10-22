add_library(preprocess preprocess.c)

target_include_directories(preprocess PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../nicslu/include/" "${CMAKE_CURRENT_SOURCE_DIR}/../nicslu/util/")