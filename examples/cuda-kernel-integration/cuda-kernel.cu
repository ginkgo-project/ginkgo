#include <iostream>
constexpr unsigned int default_block_size = 512;


__global__ __launch_bounds__(default_block_size) void parsinv_kernel( 
    int n, // matrix size
    int Lnnz, // number of nonzeros in LT stored in CSR, upper triangular  (equivalent to L in CSC)
    const int*  Lrowptr, // row pointer L
    const int*  Lcolidx, //col index L 
    const double* Lval, // val array L
    int Snnz, // number of nonzeros in S (stored in CSR, full sparse)
    const int* Srowptr, // row pointer S
    const int* Srowidx, // row index S 
    const int* Scolidx, //col index S 
    double *Sval, // val array S
    double *tval 
    ){
    
    int threadidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadidx < Snnz) {
        int i, j, il, is, jl, js;
        double Lii, s=0.0, sp;

        // handle element S(threadidx) = S(i,j)
        i = Srowidx[ threadidx ];
        j = Scolidx[ threadidx ];


	// we are working on a symmetric matrix S
        // if we notice j>i, we compute S(j,i) instead of S(i,j)
        // maybe later
        if( i>j ){
		return;
	    // swap indices - there might be a more efficient way, though
            int t = i;
            i = j;
            j = t;
        }

        // retrieve L(i,i), easy as these are the first element in each row
        Lii = Lval[ Lrowptr[ i ] ];
        // compute L(i,:).* S(j,:)
        // il and is are iterating over the nonzero entries in the respective rows
        il = Lrowptr[ i ]+1;
        is = Srowptr[ j ]+1;
        while( il < Lrowptr[i+1] && is < Srowptr[ j+1 ] ){
            sp = 0.0;
            // jl and js are the col-indices of the respective nonzero entries
            jl = Lcolidx[ il ];
            js = Scolidx[ is ];
            sp = (jl == js) ? Lval[ il ] * Sval[ is ] : sp;
            s = (jl == js) ? s+sp : s;
            il = (jl <= js) ? il+1 : il;
            is = (jl >= js) ? is+1 : is;
        }
        s = 1. / Lii * s; // scaling
        
        if (i == j) // diagonal element
            Sval[ threadidx ] = 1. / ( Lii * Lii) - s;
        else{  
            Sval[ threadidx ] = - s;
	    for(int t = Srowptr[ j ]; t<Srowptr[ j+1 ]; t++){
		if( Scolidx[t] == i ){
			Sval[t] = -s;
		       break;	
		}
	    }
	    
	}


    }
}



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
    ){
    unsigned int grid_dim = (Snnz + default_block_size - 1) / default_block_size;
    // use tval for Jacbi-style updates
    double *tval;
    //cudaMalloc(&tval, sizeof(double)*Snnz);

    parsinv_kernel<<<dim3(grid_dim), dim3(default_block_size)>>>(n, Lnnz, Lrowptr, Lcolidx, Lval, Snnz, Srowptr, Srowidx, Scolidx, Sval, tval);
    //cudaMemcpy(Sval, tval, sizeof(double)*Snnz, cudaMemcpyDeviceToDevice);
    // cudaFree(tval);
}












__global__ void parsinv_residual_kernel(
    int n, // matrix size
    int Annz, // number of nonzeros in A stored in CSR
    int *Arowptr, // row pointer A
    int *Arowidx, //row index A
    int *Acolidx, //col index A
    double *Aval, // val array A
    int *Srowptr, // row pointer S
    int *Scolidx, //col index S
    double *Sval, // val array S
    double *tval // residual vector of length Annz
    ){

    int threadidx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadidx < Annz) {
        int i, j, ia, is, ja, js;
        double  s=0.0, sp;
        // handle element A(threadidx) = A(i,j)
        i = Arowidx[ threadidx ];
        j = Acolidx[ threadidx ];





	// tmp = A*S - I
	// if diagonal element, set tmp to -1.0
	// tval[ threadidx ] = (i==j) ? -1.0 : 0.0; 


        // compute tmp(i,j) = A(i,:) * S(:,j) = A(i,:) * S(j,:)
        // il and is are iterating over the nonzero entries in the respective rows
        ia = Arowptr[ i ];
        is = Srowptr[ j ];
 	while( ia < Arowptr[i+1] && is < Srowptr[ j+1 ] ){
            sp = 0.0;
            // ja and js are the col-indices of the respective nonzero entries
            ja = Acolidx[ ia ];
            js = Scolidx[ is ];
            sp = (ja == js) ? Aval[ ia ] * Sval[ is ] : sp;
            s = (ja == js) ? s+sp : s;
            ia = (ja <= js) ? ia+1 : ia;
            is = (ja >= js) ? is+1 : is;
        }
	tval[ threadidx ] = (i==j) ? s-1.0 : s;
    }
}




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
    ){
    unsigned int grid_dim = (Annz + default_block_size - 1) / default_block_size;

    parsinv_residual_kernel<<<dim3(grid_dim), dim3(default_block_size)>>>(n, Annz, Arowptr, Arowidx, Acolidx, Aval, Srowptr, Scolidx, Sval, tval);
}








// computes spones(A).*B and writes result in A
__global__ void ASpOnesB_kernel(
    int n, // matrix size
    int *Arowptr, // row pointer A
    int *Acolidx, //col index A
    double *Aval, // val array A
    const int *Browptr, // row pointer B
    const int *Bcolidx, //col index B
    const double *Bval // val array B
    ){

    int threadidx = blockIdx.x * blockDim.x + threadIdx.x;
    int ia, ib, ja, jb;

    if (threadidx < n) {
	int row = threadidx;

        ia = Arowptr[ row ];
        ib = Browptr[ row ];
        while( ia < Arowptr[ row+1 ] && ib < Browptr[ row+1 ] ){
            // ja and jb are the col-indices of the respective nonzero entries
            ja = Acolidx[ ia ];
            jb = Bcolidx[ ib ];
	    if( ja == jb ){
		    Aval[ia] = Bval[ib];
	    }
            ia = (ja <= jb) ? ia+1 : ia;
            ib = (ja >= jb) ? ib+1 : ib;
        }
    }
}






void ASpOnesB(
    int n, // matrix size
    int *Arowptr, // row pointer A
    int *Acolidx, //col index A
    double *Aval, // val array A
    const int *Browptr, // row pointer B
    const int *Bcolidx, //col index B
    const double *Bval // val array B
    ){
    unsigned int grid_dim = (n + default_block_size - 1) / default_block_size;

    ASpOnesB_kernel<<<dim3(grid_dim), dim3(default_block_size)>>>(n, Arowptr, Acolidx, Aval, Browptr, Bcolidx, Bval );
}
