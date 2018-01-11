#include <core/base/exception_helpers.hpp>


#include <gtest/gtest.h>


namespace {


void not_implemented_func() NOT_IMPLEMENTED;


TEST(NotImplemented, ThrowsWhenUsed)
{
    ASSERT_THROW(not_implemented_func(), gko::NotImplemented);
}


void not_compiled_func() NOT_COMPILED(cpu);


TEST(NotCompiled, ThrowsWhenUsed)
{
    ASSERT_THROW(not_compiled_func(), gko::NotCompiled);
}


void does_not_support_int() { throw NOT_SUPPORTED(int); }


TEST(NotSupported, ReturnsNotSupportedException)
{
    ASSERT_THROW(does_not_support_int(), gko::NotSupported);
}


TEST(AssertConformant, DoesNotThrowWhenConformant)
{
    ASSERT_NO_THROW(ASSERT_CONFORMANT(gko::size(3, 5), gko::size(5, 6)));
}


TEST(AssertConformant, ThrowsWhenNotConformant)
{
    ASSERT_THROW(ASSERT_CONFORMANT(gko::size(3, 5), gko::size(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualRows, DoesNotThrowWhenEqualRowSize)
{
    ASSERT_NO_THROW(ASSERT_EQUAL_ROWS(gko::size(5, 3), gko::size(5, 6)));
}


TEST(AssertEqualRows, ThrowsWhenDifferentRowSize)
{
    ASSERT_THROW(ASSERT_EQUAL_ROWS(gko::size(3, 5), gko::size(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualCols, DoesNotThrowWhenEqualColSize)
{
    ASSERT_NO_THROW(ASSERT_EQUAL_COLS(gko::size(3, 6), gko::size(5, 6)));
}


TEST(AssertEqualCols, ThrowsWhenDifferentColSize)
{
    ASSERT_THROW(ASSERT_EQUAL_COLS(gko::size(3, 5), gko::size(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualDimensions, DoesNotThrowWhenEqualDimensions)
{
    ASSERT_NO_THROW(ASSERT_EQUAL_DIMENSIONS(gko::size(5, 6), gko::size(5, 6)));
}


TEST(AssertEqualDimensions, ThrowsWhenDifferentDimensions)
{
    ASSERT_THROW(ASSERT_EQUAL_DIMENSIONS(gko::size(3, 5), gko::size(7, 5)),
                 gko::DimensionMismatch);
}


TEST(EnsureAllocated, DoesNotThrowWhenAllocated)
{
    int x = 5;
    ASSERT_NO_THROW(ENSURE_ALLOCATED(&x, "CPU", 4));
}


TEST(EnsureAllocated, ThrowsWhenNotAllocated)
{
    ASSERT_THROW(ENSURE_ALLOCATED(nullptr, "CPU", 20), gko::AllocationError);
}


}  // namespace
