"""
Ginkgo Base image
Contents:
    CUDA version set by the user
    GNU compilers version set by the user
    LLVM/Clang clang-tidy version set by the user
    Intel ICC and ICPC version set according to the CUDA version
    OpenMP latest apt version for Clang+OpenMP
    Python 2 and 3 (upstream)
    cmake (upstream)
    git, openssh, doxygen, curl, valgrind, graphviz, jq latest apt version
    build-essential, automake, pkg-config, libtool, latest apt version
    iwyu precompiled version 6.0
"""
# pylint: disable=invalid-name, undefined-variable, used-before-assignment

import os

cuda_version = USERARG.get('cuda', '10.0')

image = 'nvidia/cuda:{}-devel-ubuntu16.04'.format(cuda_version)
Stage0.baseimage(image)


# Correctly set the LIBRARY_PATH
Stage0 += environment(variables={'CUDA_INSTALL_PATH': '/usr/local/cuda/'})
Stage0 += environment(variables={'CUDA_PATH': '/usr/local/cuda/'})
Stage0 += environment(variables={'CUDA_ROOT': '/usr/local/cuda/'})
Stage0 += environment(variables={'CUDA_SDK': '/usr/local/cuda/'})
Stage0 += environment(variables={'CUDA_INC_PATH': '/usr/local/cuda/include'})
Stage0 += environment(variables={'PATH': '$PATH:/usr/local/cuda/bin'})
Stage0 += environment(variables={'LIBRARY_PATH': '$LIBRARY_PATH:/usr/local/cuda/lib64/stubs'})
Stage0 += environment(variables={'LD_LIBRARY_PATH': '$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs'})
Stage0 += environment(variables={'LD_RUN_PATH': 'usr/local/cuda/lib64/stubs'})
Stage0 += environment(variables={'INCLUDEPATH': '/usr/local/cuda/include'})
Stage0 += environment(variables={'CPATH': '/usr/local/cuda/include'})
Stage0 += environment(variables={'MANPATH': '/usr/local/cuda/doc/man'})


# Setup extra tools
Stage0 += python()
Stage0 += cmake(eula=True)
Stage0 += apt_get(ospackages=['git', 'openssh-client', 'doxygen', 'curl', 'valgrind', 'graphviz'])
Stage0 += apt_get(ospackages=['jq', 'iwyu'])
Stage0 += apt_get(ospackages=['build-essential', 'automake', 'pkg-config', 'libtool'])


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

# hwloc
if float(cuda_version) >= float(9.2):
    Stage0 += shell(commands=['cd /var/tmp',
                              'git clone https://github.com/open-mpi/hwloc.git hwloc'])
    Stage0 += shell(commands=['cd /var/tmp/hwloc', './autogen.sh',
                              './configure --prefix=/usr --disable-nvml', 'make -j10', 'make install'])

    # upload valid FineCI topology and set it for hwloc
    if os.path.isfile('topology/fineci.xml'):
        Stage0 += copy(src='topology/fineci.xml', dest='/')
        Stage0 += environment(variables={'HWLOC_XMLFILE': '/fineci.xml'})
        Stage0 += environment(variables={'HWLOC_THISSYSTEM': '1'})


# Convert from CUDA version to Intel Compiler years
intel_versions = {'9.0' : 'none', '9.1' : '2017', '9.2' : '2017', '10.0' : '2018'}
intel_path = 'intel/parallel_studio_xe_{}/compilers_and_libraries/linux/'.format(intel_versions.get(cuda_version))
if os.path.isdir(intel_path):
        Stage0 += copy(src=intel_path+'bin/intel64/', dest='/opt/intel/bin/')
        Stage0 += copy(src=intel_path+'lib/intel64/', dest='/opt/intel/lib/')
        Stage0 += copy(src=intel_path+'include/', dest='/opt/intel/include/')
        Stage0 += environment(variables={'INTEL_LICENSE_FILE': '28518@scclic1.scc.kit.edu'})
        Stage0 += environment(variables={'PATH': '$PATH:/opt/intel/bin'})
        Stage0 += environment(variables={'LIBRARY_PATH': '$LIBRARY_PATH:/opt/intel/lib'})
        Stage0 += environment(variables={'LD_LIBRARY_PATH': '$LD_LIBRARY_PATH:/opt/intel/lib'})
        Stage0 += environment(variables={'LD_RUN_PATH': '$LD_RUN_PATH:/opt/intel/lib'})
