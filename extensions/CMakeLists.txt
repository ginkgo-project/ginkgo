if (GINKGO_EXTENSION_KOKKOS)
    find_package(Kokkos 4.1 REQUIRED)

    set(GINKGO_Kokkos_VERSION ${Kokkos_VERSION} PARENT_SCOPE)
    set(GINKGO_Kokkos_DEVICES ${Kokkos_DEVICES} PARENT_SCOPE)

    add_library(ginkgo_ext_kokkos INTERFACE)
    add_library(Ginkgo::ext::kokkos ALIAS ginkgo_ext_kokkos)
    set_property(TARGET ginkgo_ext_kokkos PROPERTY EXPORT_NAME ext::kokkos)

    target_link_libraries(ginkgo_ext_kokkos INTERFACE Kokkos::kokkos)
    target_compile_features(ginkgo_ext_kokkos INTERFACE cxx_std_17)

    configure_file("${Ginkgo_SOURCE_DIR}/include/ginkgo/extensions/kokkos/config.hpp.in"
            "${Ginkgo_BINARY_DIR}/include/ginkgo/extensions/kokkos/config.hpp"
            @ONLY
    )

    ginkgo_install_library(ginkgo_ext_kokkos)
    install(FILES "${Ginkgo_BINARY_DIR}/include/ginkgo/extensions/kokkos/config.hpp"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ginkgo/extensions/kokkos"
    )
    install(DIRECTORY "${Ginkgo_SOURCE_DIR}/include/ginkgo/extensions/kokkos"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ginkgo/extensions"
            FILES_MATCHING PATTERN "*.hpp"
    )
    install(FILES "${Ginkgo_SOURCE_DIR}/include/ginkgo/extensions/kokkos.hpp"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ginkgo/extensions"
    )
endif ()

if (GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif ()
