// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>

#include "ginkgo/core/base/matrix_data.hpp"
/**
 * Reads an input file
 *
 *
 * This read an electrostatic input matrix file and returns a matrix, and a load
 * vector
 *
 *
 * @param fstring filename contating the matrix
 * @param mode either binary or ascii 0 for binary 1 for ascii
 * @return sum of `values`, or 0.0 if `values` is empty.
 */

template <typename ValueType, typename IndexType>
std::vector<gko::matrix_data<ValueType, IndexType>> read_inputAscii(
    std::string fstring)
{
    std::ifstream fstream;
    std::string fname = "data/" + fstring + ".amtx";
    fstream.open(fname);

    int num_rows = 0;
    fstream >> num_rows;
    std::vector<gko::matrix_data<ValueType, IndexType>> mat_data{
        gko::matrix_data<ValueType, IndexType>(gko::dim<2>(num_rows)),
        gko::matrix_data<ValueType, IndexType>(gko::dim<2>(num_rows, 1))};
    for (auto row = 0; row < num_rows; row++) {
        std::string start_row;
        fstream >> start_row;
        for (auto col = 0; col < num_rows; col++) {
            ValueType mat_val = 0.0;
            fstream >> mat_val;
            mat_data[0].nonzeros.emplace_back(row, col, mat_val);
        }
        ValueType rhs_val = 0.0;
        fstream >> rhs_val;
        mat_data[1].nonzeros.emplace_back(row, 0, rhs_val);
    }
    return mat_data;
}


template <typename ValueType, typename IndexType>
std::vector<gko::matrix_data<ValueType, IndexType>> read_inputBinary(
    std::string fstring)
{
    int num_rows, index_placeholder;
    std::string fname = "data/" + fstring + ".bmtx";

    std::ifstream fstream(fname, std::ios::binary);
    if (!fstream) {
        std::cerr << "Error opening file: " << fname << std::endl;
        exit(1);
    }
    // skipping 4 initial padding bytes given by the fortran formatting
    fstream.seekg(4, std::ios::cur);

    fstream.read(reinterpret_cast<char*>(&num_rows), sizeof(IndexType));


    std::vector<gko::matrix_data<ValueType, IndexType>> mat_data{
        gko::matrix_data<ValueType, IndexType>(gko::dim<2>(num_rows)),
        gko::matrix_data<ValueType, IndexType>(gko::dim<2>(num_rows, 1))};

    fstream.seekg(4, std::ios::cur);

    for (auto row = 0; row < num_rows; row++) {
        // skipping 8 initial padding bytes (one for line break, and one for
        // fortran formatting)
        fstream.seekg(8, std::ios::cur);

        fstream.read(reinterpret_cast<char*>(&index_placeholder),
                     sizeof(IndexType));

        // skipping 4 bytes
        fstream.seekg(4, std::ios::cur);

        for (auto col = 0; col < num_rows; col++) {
            float mat_val = 0.0;
            fstream.read(reinterpret_cast<char*>(&mat_val), sizeof(float));
            mat_data[0].nonzeros.emplace_back(row, col, (double)mat_val);
        }
        float rhs_val = 0.0;
        fstream.read(reinterpret_cast<char*>(&rhs_val), sizeof(float));
        mat_data[1].nonzeros.emplace_back(row, 0, (double)rhs_val);
    }
    fstream.close();
    return mat_data;
}

/**
 * Reads config file for GInkgo instance
 *
 * parses config file for the file name
 *
 * @param fstring config file name, default set to electro.config
 * @return vector of strings containing parameters value: executor, solver,
 * problem_name, input_mode
 */
std::vector<std::string> read_config(std::string fstring = "electro.config")
{
    std::string tmp_string;
    std::vector<std::string> config_strings;

    int i;
    // parse config file to get different parameters for the solver instance

    std::ifstream fstream_config;
    fstream_config.open(fstring);
    if (!fstream_config) {
        std::cerr << "Config file doesn't exist \n"
                  << "please add a electro.conf file in the current directory "
                     "to run Ginkgo for electrostatic casopt problems";
        exit(1);
    }
    for (i = 0; i < 4; i++) {
        if (!fstream_config.eof()) {
            fstream_config >> tmp_string;
        } else {
            switch (i) {
            case 0:
                tmp_string = "reference";

            case 1:
                tmp_string = "gmres";

            case 2:
                tmp_string = "sphere";
            default:
                tmp_string = "ascii";
            }
        }
        config_strings.push_back(tmp_string);
    }

    // display configuration

    std::cout << "Current Ginkgo configuration for electrostatic" << std::endl
              << "    exectutor: " << config_strings[0] << std::endl
              << "    solver: " << config_strings[1] << std::endl
              << "    problem name: " << config_strings[2] << std::endl
              << "    input mode: " << config_strings[3] << std::endl;


    return config_strings;
}