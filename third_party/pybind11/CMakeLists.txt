message(STATUS "Fetching external library pybind11")
include(CPM.cmake)

cpmaddpackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 VERSION 2.12.0)
