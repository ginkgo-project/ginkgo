include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
    "#include <type_traits>
    #include <cstdint>
    static_assert(std::is_same<std::uint64_t, std::size_t>::value, \"INSTANTIATE_UINT64\");
    int main() {}"
    GKO_SIZE_T_IS_UINT64_T
    FAIL_REGEX ".*INSTANTIATE_UINT64.#"
)
