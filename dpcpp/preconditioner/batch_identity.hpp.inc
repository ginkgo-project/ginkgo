// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @see reference/preconditioner/batch_identity.hpp
 */
template <typename ValueType>
class Identity final {
public:
    using value_type = ValueType;

    static constexpr int work_size = 0;

    static int dynamic_work_size(int, int) { return 0; }

    template <typename batch_item_type>
    void generate(size_type, const batch_item_type&, ValueType*,
                  sycl::nd_item<3> item_ct1)
    {}

    __dpct_inline__ void apply(const int num_rows, const ValueType* const r,
                               ValueType* const z,
                               sycl::nd_item<3> item_ct1) const
    {
        for (int li = item_ct1.get_local_linear_id(); li < num_rows;
             li += item_ct1.get_local_range().size()) {
            z[li] = r[li];
        }
    }
};
