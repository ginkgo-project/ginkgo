#include <exception_helpers.hpp>


#include <gtest/gtest.h>


namespace {


void my_func() NOT_IMPLEMENTED;


TEST(NotImplemented, ThrowsWhenUsed)
{
    ASSERT_THROW(my_func(), msparse::NotImplemented);
}


void does_not_support_int() { throw NOT_SUPPORTED(int); }


TEST(NotSupported, ReturnsNotSupportedException)
{
    ASSERT_THROW(does_not_support_int(), msparse::NotSupported);
}


struct dummy_matrix {
    int rows;
    int cols;
    int get_num_rows() const { return rows; }
    int get_num_cols() const { return cols; }
};


TEST(AssertConformant, DoesNotThrowWhenConformant)
{
    dummy_matrix oper{3, 5};
    dummy_matrix vecs{5, 6};
    ASSERT_NO_THROW(ASSERT_CONFORMANT(&oper, &vecs));
}


TEST(AssertConformant, ThrowsWhenNotConformant)
{
    dummy_matrix oper{3, 5};
    dummy_matrix vecs{7, 3};
    ASSERT_THROW(ASSERT_CONFORMANT(&oper, &vecs), msparse::DimensionMismatch);
}


TEST(EnsureAllocated, DoesNotThrowWhenAllocated)
{
    int x = 5;
    ASSERT_NO_THROW(ENSURE_ALLOCATED(&x, "CPU", 4));
}


TEST(EnsureAllocated, ThrowsWhenNotAllocated)
{
    ASSERT_THROW(ENSURE_ALLOCATED(nullptr, "CPU", 20),
                 msparse::AllocationError);
}


}  // namespace
