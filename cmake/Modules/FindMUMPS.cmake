#.rst:
# FindMUMPS
# -------
#
# Find the MUMPS sparse direct solver library.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``MUMPS::MUMPS``
#   The MUMPS library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``MUMPS_INCLUDE_DIRS``
#   where to find dmumps_c.h, smumps_c.h, etc.
#
# ``MUMPS_LIBRARIES``
#   the libraries to link against in order to use MUMPS.
#
# ``MUMPS_FOUND``
#   If false, do not try to use the MUMPS library.

find_path(
    MUMPS_INCLUDE_DIR
    NAMES dmumps_c.h
    HINTS ${MUMPS_DIR}
    ENV MUMPS_DIR
    PATH_SUFFIXES include
)

find_library(
    MUMPS_DMUMPS_LIBRARY
    NAMES dmumps
    HINTS ${MUMPS_DIR}
    ENV MUMPS_DIR
    PATH_SUFFIXES lib lib64
)

find_library(
    MUMPS_SMUMPS_LIBRARY
    NAMES smumps
    HINTS ${MUMPS_DIR}
    ENV MUMPS_DIR
    PATH_SUFFIXES lib lib64
)

find_library(
    MUMPS_CMUMPS_LIBRARY
    NAMES cmumps
    HINTS ${MUMPS_DIR}
    ENV MUMPS_DIR
    PATH_SUFFIXES lib lib64
)

find_library(
    MUMPS_ZMUMPS_LIBRARY
    NAMES zmumps
    HINTS ${MUMPS_DIR}
    ENV MUMPS_DIR
    PATH_SUFFIXES lib lib64
)

find_library(
    MUMPS_COMMON_LIBRARY
    NAMES mumps_common
    HINTS ${MUMPS_DIR}
    ENV MUMPS_DIR
    PATH_SUFFIXES lib lib64
)

find_library(
    MUMPS_PORD_LIBRARY
    NAMES pord
    HINTS ${MUMPS_DIR}
    ENV MUMPS_DIR
    PATH_SUFFIXES lib lib64
)

# MUMPS is written in Fortran and calls Fortran MPI bindings
# (mpi_comm_rank_, mpi_comm_dup_, etc.), so we need the Fortran
# MPI runtime and the Fortran runtime library.
find_library(
    MUMPS_MPIFORT_LIBRARY
    NAMES mpifort mpi_mpifh
    HINTS ${MPI_Fortran_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
)

find_library(MUMPS_GFORTRAN_LIBRARY NAMES gfortran PATH_SUFFIXES lib lib64)

# MUMPS depends on ScaLAPACK for parallel operations.
find_library(
    MUMPS_SCALAPACK_LIBRARY
    NAMES scalapack
    HINTS ${SCALAPACK_DIR} ${PETSC_DIR}/${PETSC_ARCH}
    ENV SCALAPACK_DIR
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    MUMPS
    REQUIRED_VARS MUMPS_DMUMPS_LIBRARY MUMPS_COMMON_LIBRARY MUMPS_INCLUDE_DIR
)

if(MUMPS_FOUND)
    set(MUMPS_INCLUDE_DIRS ${MUMPS_INCLUDE_DIR})
    set(MUMPS_LIBRARIES
        ${MUMPS_DMUMPS_LIBRARY}
        ${MUMPS_SMUMPS_LIBRARY}
        ${MUMPS_CMUMPS_LIBRARY}
        ${MUMPS_ZMUMPS_LIBRARY}
        ${MUMPS_COMMON_LIBRARY}
    )
    if(MUMPS_PORD_LIBRARY)
        list(APPEND MUMPS_LIBRARIES ${MUMPS_PORD_LIBRARY})
    endif()
    if(MUMPS_MPIFORT_LIBRARY)
        list(APPEND MUMPS_LIBRARIES ${MUMPS_MPIFORT_LIBRARY})
    endif()
    if(MUMPS_GFORTRAN_LIBRARY)
        list(APPEND MUMPS_LIBRARIES ${MUMPS_GFORTRAN_LIBRARY})
    endif()
    if(MUMPS_SCALAPACK_LIBRARY)
        list(APPEND MUMPS_LIBRARIES ${MUMPS_SCALAPACK_LIBRARY})
    endif()
    unset(MUMPS_INCLUDE_DIR)
    unset(MUMPS_DMUMPS_LIBRARY)
    unset(MUMPS_SMUMPS_LIBRARY)
    unset(MUMPS_CMUMPS_LIBRARY)
    unset(MUMPS_ZMUMPS_LIBRARY)
    unset(MUMPS_COMMON_LIBRARY)
    unset(MUMPS_PORD_LIBRARY)
    unset(MUMPS_MPIFORT_LIBRARY)
    unset(MUMPS_GFORTRAN_LIBRARY)
    unset(MUMPS_SCALAPACK_LIBRARY)

    if(NOT TARGET MUMPS::MUMPS)
        add_library(MUMPS::MUMPS INTERFACE IMPORTED)
        set_target_properties(
            MUMPS::MUMPS
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${MUMPS_LIBRARIES}"
        )
    endif()
endif()
