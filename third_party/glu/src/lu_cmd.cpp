#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include "symbolic.h"
#include "numeric.h"
#include "Timer.h"
#include "preprocess.h"
#include "nicslu.h"

using namespace std;

void help_message()
{
    cout << endl;
    cout << "GLU program V3.0" << endl;
    cout << "Usage: ./lu_cmd -i inputfile" << endl;
    cout << "Additional usage: ./lu_cmd -i inputfile -p" << endl;
    cout << "-p to enable perturbation" << endl;
}

int main(int argc, char** argv)
{
    Timer t;
    double utime;
    SNicsLU *nicslu;

    char *matrixName = NULL;
    bool PERTURB = false;

    double *ax = NULL;
    unsigned int *ai = NULL, *ap = NULL;
    unsigned int n;

    if (argc < 3) {
        help_message();
        return -1;
    }

    for (int i = 1; i < argc;) {
        if (strcmp(argv[i], "-i") == 0) {
            if(i+1 > argc) {
                help_message();
                return -1;
            }
            matrixName = argv[i+1];
            i += 2;
        }
        else if (strcmp(argv[i], "-p") == 0) {
            PERTURB = true;
            i += 1;
        }        
        else {
            help_message();
            return -1;
        }
    }
    

    nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));

    int err = preprocess(matrixName, nicslu, &ax, &ai, &ap);
    if (err)
    {
        // cout << "Reading matrix error" << endl;
        exit(1);
    }

    n = nicslu->n;

    cout << "Matrix Row: " << n << endl;
    cout << "Original nonzero: " << nicslu->nnz << endl;

    t.start();

    Symbolic_Matrix A_sym(n, cout, cerr);
    A_sym.fill_in(ai, ap);
    t.elapsedUserTime(utime);
    cout << "Symbolic time: " << utime << " ms" << endl;

    t.start();
    A_sym.csr();
    t.elapsedUserTime(utime);
    cout << "CSR time: " << utime << " ms" << endl;

    t.start();
    A_sym.predictLU(ai, ap, ax);
    t.elapsedUserTime(utime);
    cout << "PredictLU time: " << utime << " ms" << endl;

    t.start();
    A_sym.leveling();
    t.elapsedUserTime(utime);
    cout << "Leveling time: " << utime << " ms" << endl;

    A_sym.ABFTCalculateCCA();
//    A_sym.PrintLevel();

    LUonDevice(A_sym, cout, cerr, PERTURB);

    A_sym.ABFTCheckResult();

    //solve Ax=b
    vector<REAL> b(n, 1.);
    vector<REAL> x = A_sym.solve(nicslu, b);
    {
        ofstream x_f("x.dat");
        for (double xx: x)
            x_f << xx << '\n';
    }

}
