#!/usr/bin/env bash
# Compile-checks omp/base/kernel_launch_simd.hpp for a target ISA. Highway's
# Vec/Mask are sizeless on SVE and RVV, so misuse fails to build -- which makes
# this check meaningful even on hardware that cannot run the result.
#
# Always reports the target the flags actually resolved to: Highway silently
# drops targets it deems broken (HWY_BROKEN_SVE) and falls back to NEON, which
# would compile clean and prove nothing. --require-scalable makes that loud.
#
# Usage:
#   check_simd_portability.sh <compiler> [--require-scalable] <flags...>
#
# Options:
#   --require-scalable  fail unless the resolved target has sizeless vectors
#                       (HWY_HAVE_SCALABLE), i.e. SVE or RVV. NEON never does.
#
# Environment:
#   GKO_HWY_INCLUDE    Highway include root; defaults to
#                      <repo>/build/_deps/hwy-src
#   GKO_BUILD_INCLUDE  directory holding the generated ginkgo/config.hpp;
#                      defaults to <repo>/build/include
#
# Examples:
#   check_simd_portability.sh g++ -march=native
#   check_simd_portability.sh g++ -DHWY_COMPILE_ONLY_EMU128
#   check_simd_portability.sh g++ --require-scalable -march=armv8.2-a+sve
#   check_simd_portability.sh g++ --require-scalable -march=rv64gcv
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <compiler> [--require-scalable] [flags...]" >&2
    exit 2
fi

compiler="$1"
shift

require_scalable=0
flags=()
for arg in "$@"; do
    case "$arg" in
        --require-scalable) require_scalable=1 ;;
        *) flags+=("$arg") ;;
    esac
done

# Both the repo root and its public include/ directory are required: the repo
# root resolves "omp/base/...", and include/ resolves <ginkgo/core/base/types.hpp>.
# build/include supplies the generated ginkgo/config.hpp.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hwy_include="${GKO_HWY_INCLUDE:-${repo_root}/build/_deps/hwy-src}"
build_include="${GKO_BUILD_INCLUDE:-${repo_root}/build/include}"

if [ ! -f "${hwy_include}/hwy/highway.h" ]; then
    echo "RESULT   : FAIL - no hwy/highway.h under '${hwy_include}'; set GKO_HWY_INCLUDE to a Highway 1.4.0 checkout" >&2
    exit 3
fi
if [ ! -f "${build_include}/ginkgo/config.hpp" ]; then
    echo "RESULT   : FAIL - no ginkgo/config.hpp under '${build_include}'; configure Ginkgo once or set GKO_BUILD_INCLUDE" >&2
    exit 3
fi

src="$(mktemp /tmp/gko_simd_check_XXXXXX.cpp)"
obj="${src%.cpp}.o"
trap 'rm -f "$src" "$obj"' EXIT

cat > "$src" <<'EOF'
#include "omp/base/kernel_launch_simd.hpp"

#include <cstdio>

namespace simd = gko::kernels::omp::simd;
namespace hn = hwy::HWY_NAMESPACE;

#define GKO_SIMD_STR2(x) #x
#define GKO_SIMD_STR(x) GKO_SIMD_STR2(x)

#ifdef GKO_SIMD_REQUIRE_SCALABLE
#if !HWY_HAVE_SCALABLE
#error "resolved Highway target has fixed-size vectors; --require-scalable needs SVE or RVV"
#endif
#endif

// External linkage so -O2 cannot drop it: the script reads this string back
// out of the object file, which is the only way to report the resolved target
// when cross-compiling, where the driver itself cannot be executed.
extern const char gko_simd_target[];
const char gko_simd_target[] =
    "GKO_HWY_TARGET=" GKO_SIMD_STR(HWY_NAMESPACE);

// Instantiate every node shape, in both full and partial form and for both
// spec value types, so that a sizeless-type violation anywhere in the layer is
// a compile error here. float and double are separate instantiations on
// purpose: on SVE they differ in Lanes and in TFromD.
template <typename T, typename Access>
void instantiate(const T* xp, T* yp, T* zp, std::size_t count, T alpha)
{
    using D = hn::ScalableTag<T>;
    simd::ref_leaf<D, Access> x{xp, count};
    simd::ref<D, Access> y{yp, count};
    y += alpha * x;
    y -= x * alpha;
    y = x + alpha * y;
    y = x - y;
    y = x / alpha;
    y = -x;
    y = abs(x);
    y = conj(x);
    y = squared_norm(x);
    y = zero(x);
    y = one(x);
    y *= alpha;
    y /= alpha;

    // ref-from-ref assignment: the hand-written copy-assignment operator is
    // only compiled when odr-used from a same-type ref, and it exists because
    // the implicit one would copy the pointer and silently store nothing.
    simd::ref<D, Access> z{zp, count};
    z = y;

    // The reduction shape from Phase 3: accumulators are named locals, never
    // an array, and are initialized with Set rather than default-constructed.
    const D d;
    auto acc0 = hn::Set(d, T{0});
    auto acc1 = hn::Set(d, T{0});
    acc0 = hn::Add(acc0, simd::eval_to_vec(d, squared_norm(x)));
    acc1 = hn::Add(acc1, simd::eval_to_vec(d, x * y));
    // The non-expression branch of eval_to_vec takes a raw hn::Vec<D> by
    // value, which is the exact sizeless-type shape most likely to break.
    acc0 = hn::Add(acc0, simd::eval_to_vec(d, hn::Set(d, T{1})));
    acc0 = hn::Add(acc0, acc1);
    (void)hn::ReduceSum(d, acc0);
}

#define GKO_SIMD_INSTANTIATE(_type)                                       \
    template void instantiate<_type, simd::full_access>(                  \
        const _type*, _type*, _type*, std::size_t, _type);                \
    template void instantiate<_type, simd::partial_access>(               \
        const _type*, _type*, _type*, std::size_t, _type)

GKO_SIMD_INSTANTIATE(float);
GKO_SIMD_INSTANTIATE(double);

int main()
{
    std::printf("%s\n", gko_simd_target);
    return 0;
}
EOF

echo "compiler : ${compiler}"
echo "flags    : ${flags[*]:-}"
echo "scalable : $([ "$require_scalable" -eq 1 ] && echo required || echo "not required")"

# HWY_COMPILE_ONLY_STATIC is the default single-target mode, matching
# cmake/highway.cmake. A caller passing a different HWY_COMPILE_ONLY_* selects
# another target on purpose, and Highway hard-errors if two are defined.
extra_defs=(-DHWY_DISABLE_ATTR -DHWY_DISABLE_PCLMUL_AES)
caller_selects_target=0
for arg in ${flags[@]+"${flags[@]}"}; do
    case "$arg" in
        -DHWY_COMPILE_ONLY_*) caller_selects_target=1 ;;
    esac
done
if [ "$caller_selects_target" -eq 0 ]; then
    extra_defs+=(-DHWY_COMPILE_ONLY_STATIC)
fi
if [ "$require_scalable" -eq 1 ]; then
    extra_defs+=(-DGKO_SIMD_REQUIRE_SCALABLE)
fi

if ! "$compiler" -std=c++17 -O2 ${flags[@]+"${flags[@]}"} \
    "${extra_defs[@]}" \
    -I"$repo_root" -I"$repo_root/include" -I"$build_include" -I"$hwy_include" \
    -c "$src" -o "$obj"; then
    echo "RESULT   : FAIL - expression layer does not compile for these flags" >&2
    exit 1
fi

# Read the target back out of the object rather than running the driver, so
# this reports correctly under cross-compilation too.
target="$(LC_ALL=C grep -ao 'GKO_HWY_TARGET=[A-Za-z0-9_]*' "$obj" | head -n 1 |
    cut -d= -f2)"
echo "target   : ${target:-unknown} (from HWY_NAMESPACE)"
echo "RESULT   : OK - expression layer compiles for these flags"
