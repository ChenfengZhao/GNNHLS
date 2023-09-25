#ifndef UTIL_H
#define UTIL_H


// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <math.h>
// #include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cassert>

#include "../defines.h"
#include "../data/defines_graphsage_mean.h"

int read_2d_mat(char* fn, uint64_t row_num, uint64_t col_num, uint64_t line_size, TYPE* rst);
int dequal(double a, double b);
int fequal(float a, float b);
int check_rst(uint64_t mat_tot_size, TYPE* rst, TYPE* cor_rst);
int read_2d_mat_int(char* fn, uint64_t row_num, uint64_t col_num, uint64_t line_size, int* rst);
int read_2d_mat_ui64(char* fn, uint64_t row_num, uint64_t col_num, uint64_t line_size, uint64_t* rst);
void init_2d_mat(uint64_t row_num, uint64_t col_num, TYPE init_vaule, TYPE* rst);
#endif