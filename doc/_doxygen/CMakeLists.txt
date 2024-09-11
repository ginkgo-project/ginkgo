cmake_minimum_required(VERSION 3.16)
include(helpers.cmake)
find_package(Doxygen REQUIRED)
find_package(Perl REQUIRED)
option(GINKGO_DOC_GENERATE_PDF "Generate PDF documentation" OFF)
set(GINKGO_DOC_GENERATE_PDF_YESNO NO)
if(GINKGO_DOC_GENERATE_PDF)
    set(GINKGO_DOC_GENERATE_PDF_YESNO YES)
endif()

add_subdirectory(examples)

if(GINKGO_DOC_GENERATE_PDF)
    find_package(LATEX COMPONENTS PDFLATEX REQUIRED)
endif()

include(FetchContent)
FetchContent_Declare(
    doxygen_awesome
    GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
    GIT_TAG v2.3.3
)
FetchContent_MakeAvailable(doxygen_awesome)

ginkgo_doc_gen()
