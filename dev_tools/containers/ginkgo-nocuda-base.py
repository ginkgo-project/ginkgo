"""
Ginkgo Base image
Contents:
    GNU compilers version set by the user
    LLVM/Clang version set by the user
    Intel ICC and ICPC version set to the latest available version
    OpenMP latest apt version for Clang+OpenMP
    Python 2 and 3 (upstream)
    cmake (upstream)
    build-essential, git, openssh, doxygen, curl, valgrind latest apt version
    jq, graphviz, ghostscript, texlive, texlive-latex-extra, latest apt version
    texlive-science, texlive-fonts-extra, texlive-publishers latest apt version
    clang-tidy, iwyu: latest apt version
    hwloc, libhwloc-dev, pkg-config latest apt version
    papi: adds package libpfm4, and copy precompiled papi headers and files
          from a directory called 'papi'
"""
# pylint: disable=invalid-name, undefined-variable, used-before-assignment

import os

Stage0.baseimage('ubuntu:18.04')


# Setup extra tools
Stage0 += python()
Stage0 += cmake(eula=True)
Stage0 += apt_get(ospackages=['build-essential', 'git', 'openssh-client', 'doxygen', 'curl', 'valgrind'])
Stage0 += apt_get(ospackages=['jq', 'graphviz', 'ghostscript', 'texlive', 'texlive-latex-extra'])
Stage0 += apt_get(ospackages=['texlive-science', 'texlive-fonts-extra', 'texlive-publishers'])
Stage0 += apt_get(ospackages=['clang-tidy', 'iwyu'])
Stage0 += apt_get(ospackages=['hwloc', 'libhwloc-dev', 'pkg-config'])

# GNU compilers
gnu_version = USERARG.get('gnu', '8')
Stage0 += gnu(version=gnu_version, extra_repository=True)

# Clang compilers
llvm_version = USERARG.get('llvm', '7')
Stage0 += llvm(version=llvm_version, extra_repository=True)
Stage0 += apt_get(ospackages=['libomp-dev']) #required for openmp+clang

# Copy PAPI libs
add_papi = USERARG.get('papi', 'False')
if os.path.isdir('papi/') and add_papi == 'True':
    Stage0 += apt_get(ospackages=['libpfm4'])
    Stage0 += copy(src='papi/include/*', dest='/usr/include/')
    Stage0 += copy(src='papi/lib/*', dest='/usr/lib/')
    Stage0 += copy(src='papi/bin/*', dest='/usr/bin/')

intel_path = 'intel/parallel_studio_xe_2019/compilers_and_libraries/linux/'
if os.path.isdir(intel_path):
    Stage0 += copy(src=intel_path+'bin/', dest='/opt/intel/bin/')
    Stage0 += copy(src=intel_path+'compiler/lib/intel64/', dest='/opt/intel/lib/')
    Stage0 += copy(src=intel_path+'compiler/include/', dest='/opt/intel/include/')
    Stage0 += environment(variables={'INTEL_LICENSE_FILE': '28518@scclic1.scc.kit.edu'})
    Stage0 += environment(variables={'PATH': '$PATH:/opt/intel/bin/intel64'})
    Stage0 += environment(variables={'LIBRARY_PATH': '$LIBRARY_PATH:/opt/intel/lib'})
    Stage0 += environment(variables={'LD_LIBRARY_PATH': '$LD_LIBRARY_PATH:/opt/intel/lib'})
    Stage0 += environment(variables={'LD_RUN_PATH': '$LD_RUN_PATH:/opt/intel/lib'})
