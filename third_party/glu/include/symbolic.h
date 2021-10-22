#ifndef _SYMBOLIC_H_
#define _SYMBOLIC_H_
#include <vector>
#include <unordered_set>
#include <iostream>
#include "type.h"
#include "../src/nicslu/include/nicslu.h"

class Symbolic_Matrix
{
public:
    unsigned int n;
    unsigned int nnz;
    int num_lev;
    std::vector<unsigned> sym_c_ptr;
    std::vector<unsigned> sym_r_idx;
    std::vector<unsigned> csr_r_ptr;
    std::vector<unsigned> csr_c_idx;
    std::vector<unsigned> csr_diag_ptr;
    std::vector<REAL> val;
    std::vector<unsigned> l_col_ptr; //Indices of diagonal elements
    std::vector<int> level_idx;
    std::vector<int> level_ptr;

    void predictLU(unsigned*, unsigned*, double*);
    void csr();
    void leveling();
    void fill_in(unsigned*, unsigned*);
    std::vector<REAL> solve(SNicsLU*, const std::vector<REAL> &);

    void PrintLevel();
    //ABFT
    std::vector<REAL> CCA;
    void ABFTCalculateCCA();
    void ABFTCheckResult();

    Symbolic_Matrix(unsigned n, std::ostream &out, std::ostream &err):
        n(n),
        num_lev(0),
        m_out(out),
        m_err(err) {};

    Symbolic_Matrix(std::ostream &out, std::ostream &err):
        m_out(out), m_err(err) {};

private:
    std::ostream &m_out;
    std::ostream &m_err;
};


#endif
