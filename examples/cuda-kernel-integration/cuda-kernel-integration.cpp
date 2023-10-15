/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <cstdlib>
#include <fstream>
#include <ginkgo/ginkgo.hpp>
#include <iostream>

void parsinv(
    int n, // matrix size
    int Lnnz, // number of nonzeros in LT stored in CSR, upper triangular  (equivalent to L in CSC)
    int *Lrowptr, // row pointer L
    int *Lcolidx, //col index L
    double *Lval, // val array L
    int Snnz, // number of nonzeros in S (stored in CSR, full sparse)
    int *Srowptr, // row pointer S
    int *Srowidx, // row index S
    int *Scolidx, //col index S
    double *Sval // val array S
    );


void parsinv_residual(
    int n, // matrix size
    int Annz, // number of nonzeros in A
    int *Arowptr, // row pointer A
    int *Arowidx, //row index A
    int *Acolidx, //col index A
    double *Aval, // val array A
    int *Srowptr, // row pointer S
    int *Scolidx, //col index S
    double *Sval, // val array S
    double *tval
    );


int main(int argc, char** argv)
{
    using value_type = double;
    using index_type = gko::int32;
    if (argc != 5) {
        std::cout << "Please execute with the following parameters:\n"
                  << argv[0]
                  << " <Astd::ifstream S_file(argv[1]); matrix path> <S matrix path> <L matrix path> <I matrix path>\n";
        std::exit(1);
    }
    std::ifstream A_file(argv[1]);
    std::ifstream S_file(argv[2]);
    std::ifstream L_file(argv[3]);
    std::ifstream I_file(argv[4]);
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    // Instantiate a CUDA executor
    auto gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    // Read data
    auto A_csr = gko::share(gko::read<Csr>(A_file, gpu));
    auto S_csr = gko::read<Csr>(S_file, gpu);
    auto L = gko::read<Csr>(L_file, gpu);
    auto I = gko::read<Csr>(I_file, gpu);

    const auto num_row_ptrs = S_csr->get_size()[0] + 1;
    gko::array<index_type> row_ptrs_array(gpu, num_row_ptrs);
    gpu->copy_from(gpu, num_row_ptrs, S_csr->get_const_row_ptrs(),
                   row_ptrs_array.get_data());
    auto S_coo = Coo::create(gpu);
    S_csr->move_to(S_coo);

    auto LL = gko::experimental::factorization::Cholesky<value_type, index_type>::build().on(gpu)->generate(A_csr);
    auto LLU = LL->unpack();

    gko::array<index_type> Arow_ptrs_array(gpu, num_row_ptrs);
    gpu->copy_from(gpu, num_row_ptrs, A_csr->get_const_row_ptrs(),
                   Arow_ptrs_array.get_data());
    auto A_coo = Coo::create(gpu);
    A_csr->move_to(A_coo);


    auto A_row_ptrs = Arow_ptrs_array.get_data();
    auto A_row_idxs = A_coo->get_row_idxs();
    auto A_col_idxs = A_coo->get_col_idxs();
    auto A_values = A_coo->get_values();

    auto S_row_ptrs = row_ptrs_array.get_data();
    auto S_row_idxs = S_coo->get_row_idxs();
    auto S_col_idxs = S_coo->get_col_idxs();
    auto S_values = S_coo->get_values();

    auto L_transpose_linop = LLU->get_lower_factor()->transpose();
    auto L_transpose =
        static_cast<typename Csr::transposed_type*>(L_transpose_linop.get());
    
    auto size = I->get_num_stored_elements();
    auto A_vec = Dense::create_const(
            gpu, gko::dim<2>{size, 1},
            gko::array<value_type>::const_view(gpu, size, I->get_const_values()),
            1);
    auto B_vec = Dense::create_const(
            gpu, gko::dim<2>{size, 1},
            gko::array<value_type>::const_view(gpu, size, S_coo->get_const_values()),
            1);
    auto neg_one = gko::initialize<Dense>({-gko::one<value_type>()}, gpu);
    


    auto result =
           gko::matrix::Dense<value_type>::create(gpu, gko::dim<2>{1, 1});
        A_vec->compute_norm2(result);
        std::cout << "norm selected inverse : "
                  << gpu->copy_val_to_host(result->get_values()) << std::endl;
	
        B_vec->compute_norm2(result);
        std::cout << "norm initial guess : "
                  << gpu->copy_val_to_host(result->get_values()) << std::endl;

        auto work_vec = A_vec->clone();
        work_vec->add_scaled(neg_one, B_vec);
           work_vec->compute_norm2(result);
        printf("Frobenious norm iteration %2d: %.4e\n",
                  0, gpu->copy_val_to_host(result->get_values()));

    // Solve system
    for(int i=0; i<20; i++){
    	parsinv( L_transpose->get_size()[0], 
		    L_transpose->get_num_stored_elements(), 
		    L_transpose->get_row_ptrs(), 
		    L_transpose->get_col_idxs(),
		    L_transpose->get_values(),
                    S_coo->get_num_stored_elements(),
		    S_row_ptrs,
		    S_row_idxs,
		    S_col_idxs,
		    S_values
    	);
/*
         // the next block is completely useless as (AS-I)_spy(A)=0 does not hold
        auto work_vec_t = Dense::create(gpu, gko::dim<2>(A_coo->get_num_stored_elements(), 1));
	parsinv_residual(
		    A_coo->get_size()[0],
                    A_coo->get_num_stored_elements(),
                    A_row_ptrs,
                    A_row_idxs,
                    A_col_idxs,
                    A_values,
		    //S_row_ptrs,
                    //S_col_idxs,
                    //S_values,
		    I->get_row_ptrs(),
                    I->get_col_idxs(),
                    I->get_values(),
		    work_vec_t->get_values());
*/		
	auto work_vec = A_vec->clone();
	work_vec->add_scaled(neg_one, B_vec);
	auto result =
    		gko::matrix::Dense<value_type>::create(gpu, gko::dim<2>{1, 1});
	work_vec->compute_norm2(result);
	printf("Frobenious norm iteration %2d: %.4e\n",
        	  i+1, gpu->copy_val_to_host(result->get_values()));
	
    }

    // Write result
    //write(std::cout, I);
    //write(std::cout, S_coo);
}
