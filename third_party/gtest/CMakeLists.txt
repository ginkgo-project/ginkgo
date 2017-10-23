# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

load_git_package(gtest
    "https://github.com/google/googletest.git"
    "master")

