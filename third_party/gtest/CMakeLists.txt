# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

load_git_package(gtest
    "https://github.com/google/googletest.git"
    "d5266326752f0a1dadbd310932d8f4fd8c3c5e7d")

