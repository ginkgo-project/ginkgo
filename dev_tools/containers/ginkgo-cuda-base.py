"""
Ginkgo Base image
Contents:
	CUDA version set by the user
	GNU compilers version set by the user
	LLVM/Clang clang-tidy version set by the user
	OpenMP latest apt version for Clang+OpenMP
	Python 2 and 3 (upstream)
	cmake (upstream)
	git, openssh, doxygen, curl, valgrind, graphviz latest apt version
	iwyu precompiled version 6.0
"""
# pylint: disable=invalid-name, undefined-variable, used-before-assignment

import os

cuda_version = USERARG.get('cuda', '10.0')

image = 'nvidia/cuda:{}-devel-ubuntu16.04'.format(cuda_version)
Stage0.baseimage(image)


# Setup extra tools
Stage0 += python()
Stage0 += cmake(eula=True)
Stage0 += apt_get(ospackages=['git', 'openssh-client', 'doxygen', 'curl', 'valgrind', 'graphviz'])
Stage0 += apt_get(ospackages=['iwyu'])

# GNU compilers
gnu_version = USERARG.get('gnu', '7')
Stage0 += gnu(version=gnu_version, extra_repository=True)

# Clang compilers
llvm_version = USERARG.get('llvm', '6.0')
Stage0 += llvm(version=llvm_version, extra_repository=True)
Stage0 += apt_get(ospackages=['libomp-dev']) #required for openmp+clang

# clang-tidy
clangtidy = ['clang-tidy-{}'.format(llvm_version)]
Stage0 += packages(apt_ppas=['ppa:xorg-edgers/ppa'], apt=clangtidy)
clangtidyln = ['ln -s /usr/bin/clang-tidy-{} /usr/bin/clang-tidy'.format(llvm_version)]
Stage0 += shell(commands=clangtidyln)

# IWYU
if os.path.isdir('bin/'):
        Stage0 += copy(src='bin/*', dest='/usr/bin/')

if os.path.isdir('sonar-scanner/'):
        Stage0 += copy(src='sonar-scanner/', dest='/')

# Correctly set the LIBRARY_PATH
Stage0 += environment(variables={'LIBRARY_PATH': '/usr/local/cuda/lib64/stubs'})
Stage0 += environment(variables={'LD_LIBRARY_PATH': '/usr/local/cuda/lib64/stubs'})

