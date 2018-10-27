"""
Ginkgo Base image
Contents:
	GNU compilers version set by the user
	LLVM/Clang version set by the user
	OpenMP latest apt version for Clang+OpenMP
	Python 2 and 3 (upstream)
	cmake (upstream)
	build-essential, git, openssh, doxygen, curl latest apt version
"""
# pylint: disable=invalid-name, undefined-variable, used-before-assignment


Stage0.baseimage('ubuntu:16.04')


# Setup extra tools
Stage0 += python()
Stage0 += cmake(eula=True)
Stage0 += apt_get(ospackages=['build-essential', 'git', 'openssh-client', 'doxygen', 'curl'])

# GNU compilers
gnu_version = USERARG.get('gnu', '8')
Stage0 += gnu(version=gnu_version, extra_repository=True)

# Clang compilers
llvm_version = USERARG.get('llvm', '7.0')
Stage0 += llvm(version=llvm_version, extra_repository=True)
Stage0 += apt_get(ospackages=['libomp-dev']) #required for openmp+clang

