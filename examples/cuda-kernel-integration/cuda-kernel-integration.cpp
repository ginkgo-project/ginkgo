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
#include <chrono>

void parsinv(
    int n, // matrix size
    int Lnnz, // number of nonzeros in LT stored in CSR, upper triangular  (equivalent to L in CSC)
    const int *Lrowptr, // row pointer L
    const int *Lcolidx, //col index L
    const double *Lval, // val array L
    int Snnz, // number of nonzeros in S (stored in CSR, full sparse)
    const int *Srowptr, // row pointer S
    const int *Srowidx, // row index S
    const int *Scolidx, //col index S
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
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;

    // Instantiate a CUDA executor
    auto gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    
    int debug = 0;
    // debug = 0 : only compute selected inverse based on matrix
    // debug = 1 : feed in reference solution and compare against solution
    if (argc == 2 ) {
    }
    else if ( argc == 3 ){
	    debug = 1;
	    std::cout << " Computing error ro reference solution in iterations "  << std::endl;
	}
    else {
        std::cout << "Please execute with the following parameters:\n"
                  << argv[0]
                  << "<A matrix path> <I matrix path>\n";
        std::exit(1);
    }
    std::ifstream A_file(argv[1]);
    auto A_csr = gko::share(gko::read<Csr>(A_file, gpu));
    std::unique_ptr<Csr> I;

    if( debug > 0 ){
    	std::ifstream I_file(argv[2]);
	I = gko::read<Csr>(I_file, gpu);
    }
    // use Ginkgo Cholesky factorization and use the combined as sparsity pattern S
    auto start = std::chrono::steady_clock::now();
    auto S_csr = gko::experimental::factorization::Cholesky<value_type, index_type>::build().on(gpu)->generate(A_csr);
    auto LLU = S_csr->unpack();
    auto L = LLU->get_upper_factor();
    auto end = std::chrono::steady_clock::now();
    double factorization_time = std::chrono::duration<double>(end-start).count();

    const auto num_row_ptrs = S_csr->get_size()[0] + 1;
    gko::array<index_type> row_ptrs_array(gpu, num_row_ptrs);
    gpu->copy_from(gpu, num_row_ptrs, S_csr->get_combined()->get_const_row_ptrs(),
                   row_ptrs_array.get_data());
    auto S_coo = Coo::create(gpu);
    S_csr->get_combined()->convert_to(S_coo);


    auto S_row_ptrs = row_ptrs_array.get_const_data();
    auto S_row_idxs = S_coo->get_const_row_idxs();
    auto S_col_idxs = S_coo->get_const_col_idxs();
    auto S_values = S_coo->get_values();
   
    // compute error to correct solution
    auto size = S_coo->get_num_stored_elements();
    auto neg_one = gko::initialize<Dense>({-gko::one<value_type>()}, gpu);
    std::unique_ptr<const Dense>A_vec;
    std::unique_ptr<const Dense>B_vec;
    if( debug > 0 ){
        A_vec = Dense::create_const(
            gpu, gko::dim<2>{size, 1},
            gko::array<value_type>::const_view(gpu, size, I->get_const_values()),
            1);
        B_vec = Dense::create_const(
            gpu, gko::dim<2>{size, 1},
            gko::array<value_type>::const_view(gpu, size, S_coo->get_const_values()),
            1);
    
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
    }
    // end error computation
    
    start = std::chrono::steady_clock::now();
    // Solve system
    for(int i=0; i<50; i++){
    	parsinv( L->get_size()[0], 
		    L->get_num_stored_elements(), 
		    L->get_const_row_ptrs(), 
		    L->get_const_col_idxs(),
		    L->get_const_values(),
                    S_coo->get_num_stored_elements(),
		    S_row_ptrs,
		    S_row_idxs,
		    S_col_idxs,
		    S_values
    	);
	if( debug > 0 ){
		// compute after every iteration the error to correct solution
		auto work_vec = A_vec->clone();
		work_vec->add_scaled(neg_one, B_vec);
		auto result =
    			gko::matrix::Dense<value_type>::create(gpu, gko::dim<2>{1, 1});
		work_vec->compute_norm2(result);
		printf("Frobenious norm iteration %2d: %.4e\n",
        		 i+1, gpu->copy_val_to_host(result->get_values()));
	}
    }
    end = std::chrono::steady_clock::now();
    double inverse_time = std::chrono::duration<double>(end-start).count();

    printf("Factorization time: %.4e\nSelected inverse time: %.4e\n", factorization_time, inverse_time);

    // Write result
    //write(std::cout, I);
    //write(std::cout, S_coo);
}
