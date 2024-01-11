// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_GINKGO_HPP_
#define GKO_GINKGO_HPP_


#include <ginkgo/config.hpp>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_dim.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/block_operator.hpp>
#include <ginkgo/core/base/combination.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/device.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/fwd_decls.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/intrinsics.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/machine_topology.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_assembly_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/memory.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/perturbation.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/scoped_device_id_guard.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/stream.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/timer.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/base/version.hpp>

#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/partition_helpers.hpp>
#include <ginkgo/core/distributed/polymorphic_object.hpp>

#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>

#include <ginkgo/core/distributed/vector.hpp>

#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/factorization/factorization.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/factorization/par_ict.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/factorization/par_ilut.hpp>

#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/log/papi.hpp>
#include <ginkgo/core/log/performance_hint.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>
#include <ginkgo/core/log/record.hpp>
#include <ginkgo/core/log/stream.hpp>

#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/fft.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include <ginkgo/core/multigrid/fixed_coarsening.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>

#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>

#include <ginkgo/core/reorder/amd.hpp>
#include <ginkgo/core/reorder/mc64.hpp>
#include <ginkgo/core/reorder/nested_dissection.hpp>
#include <ginkgo/core/reorder/rcm.hpp>
#include <ginkgo/core/reorder/reordering_base.hpp>
#include <ginkgo/core/reorder/scaled_reordered.hpp>

#include <ginkgo/core/solver/batch_bicgstab.hpp>
#include <ginkgo/core/solver/batch_solver_base.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/solver/solver_traits.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/solver/workspace.hpp>

#include <ginkgo/core/stop/batch_stop_enum.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>
#include <ginkgo/core/stop/time.hpp>

#include <ginkgo/core/synthesizer/containers.hpp>


#endif  // GKO_GINKGO_HPP_
