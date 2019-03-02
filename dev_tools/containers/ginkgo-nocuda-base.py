"""
Ginkgo Base image
Contents:
	GNU compilers version set by the user
	LLVM/Clang version set by the user
	OpenMP latest apt version for Clang+OpenMP
	Python 2 and 3 (upstream)
	cmake (upstream)
	build-essential, git, openssh, doxygen, curl latest apt version
	papi: adds package libpfm4, and copy precompiled papi headers and files from a directory called 'papi'
"""
# pylint: disable=invalid-name, undefined-variable, used-before-assignment

import os

Stage0.baseimage('ubuntu:16.04')


# Setup extra tools
Stage0 += python()
Stage0 += cmake(eula=True)
Stage0 += apt_get(ospackages=['build-essential', 'git', 'openssh-client', 'doxygen', 'curl', 'valgrind'])

# GNU compilers
gnu_version = USERARG.get('gnu', '8')
Stage0 += gnu(version=gnu_version, extra_repository=True)

# Clang compilers
llvm_version = USERARG.get('llvm', '7.0')
Stage0 += llvm(version=llvm_version, extra_repository=True)
Stage0 += apt_get(ospackages=['libomp-dev']) #required for openmp+clang

# Copy PAPI libs
add_papi = USERARG.get('papi', 'False')
if os.path.isdir('papi/') and add_papi == 'True':
	Stage0 += apt_get(ospackages=['libpfm4'])
	Stage0 += copy(src='papi/include/*', dest='/usr/include/') 
	Stage0 += copy(src='papi/lib/*', dest='/usr/lib/') 
	Stage0 += copy(src='papi/bin/*', dest='/usr/bin/') 
