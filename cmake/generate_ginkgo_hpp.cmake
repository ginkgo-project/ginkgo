function(ginkgo_generate_ginkgo_hpp)
    file(
        GLOB_RECURSE headers
        CONFIGURE_DEPENDS
        "${Ginkgo_SOURCE_DIR}/include/ginkgo/*.hpp"
    )
    set(GKO_PUBLIC_HEADER_CONTENTS)
    foreach(file IN LISTS headers)
        file(RELATIVE_PATH file "${Ginkgo_SOURCE_DIR}/include" "${file}")
        # just making sure it uses / path separators
        file(TO_CMAKE_PATH file "${file}")
        if(
            (file MATCHES "^ginkgo/extensions/.*$")
            OR (file MATCHES "^ginkgo/core/stop/residual_norm_reduction.hpp$")
            OR (file MATCHES "^ginkgo/core/solver/.*_trs.hpp$")
        )
            continue()
        endif()
        set(GKO_PUBLIC_HEADER_CONTENTS
            "${GKO_PUBLIC_HEADER_CONTENTS}#include <${file}>\n"
        )
    endforeach()
    configure_file(
        "${Ginkgo_SOURCE_DIR}/include/ginkgo/ginkgo.hpp.in"
        "${Ginkgo_BINARY_DIR}/include/ginkgo/ginkgo.hpp"
        @ONLY
    )
    if(EXISTS "${Ginkgo_SOURCE_DIR}/include/ginkgo/ginkgo.hpp")
        message(
            FATAL_ERROR
            "ginkgo.hpp is auto-generated and should not be in include/ginkgo"
        )
    endif()
endfunction()
