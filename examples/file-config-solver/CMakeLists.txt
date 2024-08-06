cmake_minimum_required(VERSION 3.16)
project(file-config-solver)

# We only need to find Ginkgo/nlohmann_json if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
    find_package(nlohmann_json 3.9.1 REQUIRED)
endif()

add_executable(file-config-solver file-config-solver.cpp)
target_link_libraries(file-config-solver Ginkgo::ginkgo nlohmann_json::nlohmann_json)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
# Copy the config files to the execution directory
configure_file(config/cg.json config/cg.json COPYONLY)
configure_file(config/blockjacobi-cg.json config/blockjacobi-cg.json COPYONLY)
configure_file(config/ir.json config/ir.json COPYONLY)
configure_file(config/mixed-pgm-multigrid-cg.json config/mixed-pgm-multigrid-cg.json COPYONLY)
configure_file(config/parilu-gmres.json config/parilu-gmres.json COPYONLY)
configure_file(config/pgm-multigrid-cg.json config/pgm-multigrid-cg.json COPYONLY)
