from conans import ConanFile, CMake, tools


class GinkgoConan(ConanFile):
    name = "Ginkgo"
    version = "1.3.0"
    license = "BSD-3-Clause"
    author = "Terry Cojean (tcojean@kit.edu)"
    url = "https://github.com/ginkgo-project/ginkgo"
    description = "High-performance linear algebra library for manycore systems, with a focus on sparse solution of linear systems."
    topics = ("hpc", "linear-algebra")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "reference": [True, False], "openmp": [
        True, False], "cuda": [True, False], "hip": [True, False], "dpcpp": [True, False]}
    default_options = {"shared": True, "reference": True,
                       "openmp": False, "cuda": False, "hip": False, "dpcpp": False}
    generators = "cmake"
    exports_sources = "cmake/*", "common/*", "core/*", "cuda/*", "dpcpp/*", "hip/*", "include/*", "matrices/*", "omp/*", "reference/*", "third_party/*", "CMakeLists.txt", "*.cmake", "*.md", "LICENSE", "contributors.txt"

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["GINKGO_BUILD_CONAN"] = "ON"
        cmake.definitions["GINKGO_BUILD_BENCHMARKS"] = "OFF"
        cmake.definitions["GINKGO_BUILD_EXAMPLES"] = "OFF"
        cmake.definitions["GINKGO_BUILD_TESTS"] = "ON"
        cmake.definitions["GINKGO_BUILD_REFERENCE"] = "ON" if self.options["reference"] else "OFF"
        cmake.definitions["GINKGO_BUILD_OMP"] = "ON" if self.options["openmp"] else "OFF"
        cmake.definitions["GINKGO_BUILD_CUDA"] = "ON" if self.options["cuda"] else "OFF"
        cmake.definitions["GINKGO_BUILD_HIP"] = "ON" if self.options["hip"] else "OFF"
        cmake.definitions["GINKGO_BUILD_DPCPP"] = "ON" if self.options["dpcpp"] else "OFF"
        cmake.definitions["BUILD_SHARED_LIBS"] = "ON" if self.options["shared"] else "OFF"
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        cmake.test()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()
