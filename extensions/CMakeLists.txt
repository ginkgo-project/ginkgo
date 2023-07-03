add_library(ginkgo_ext INTERFACE)
add_library(Ginkgo::ext ALIAS ginkgo_ext)
set_property(TARGET ginkgo_ext PROPERTY EXPORT_NAME ext)

if(GINKGO_EXTENSION_KOKKOS)
    find_package(Kokkos 4.1 REQUIRED)

    set(GINKGO_Kokkos_VERSION ${Kokkos_VERSION} PARENT_SCOPE)
    set(GINKGO_Kokkos_DEVICES ${Kokkos_DEVICES} PARENT_SCOPE)

    add_library(ginkgo_ext_kokkos INTERFACE)
    add_library(Ginkgo::ext::kokkos ALIAS ginkgo_ext_kokkos)
    set_property(TARGET ginkgo_ext_kokkos PROPERTY EXPORT_NAME ext::kokkos)

    target_link_libraries(ginkgo_ext_kokkos INTERFACE Kokkos::kokkos)

    target_link_libraries(ginkgo_ext INTERFACE ginkgo_ext_kokkos)

    ginkgo_install_library(ginkgo_ext_kokkos)
endif()

ginkgo_install_library(ginkgo_ext)
