/*
Utility Functions
*/

#include "util.h"

/**
 * @brief Read a 2D float/double matrix from an txt file into an array
 * 
 * @param fn Path+file_name of an external file
 * @param row_num The total number of rows in the matrix
 * @param col_num The total number of colums in the matrix
 * @param line_size The max size of line array
 * @param rst Pointer of the result array
 * 
 * @return No return, fill in rst with vaules
 */
int read_2d_mat(char* fn, uint64_t row_num, uint64_t col_num, uint64_t line_size, TYPE* rst)
{
    using namespace std;

    char line[line_size];
    char* token = nullptr;
    // TYPE rst[row_num * col_num];
    uint64_t row = 0;
    uint64_t col = 0;

    FILE* fp = fopen(fn, "r");

    if (fp == nullptr)
    {
        printf("Can't open %s\n", fn);
        exit(EXIT_FAILURE);
    }
    printf("%s has been opened\n", fn);

    while(fgets(line, sizeof(line), fp) != nullptr)
    {
        #ifdef DEBUG
            printf("line: %s\n", line);
        #endif
        token = strtok(line, ",");

        col = 0;
        while(token != nullptr)
        {
            // printf("elem %s | %f\n", token, atof(token));
            // printf("elem %f\n", atof(token));
        #ifdef FLOAT32
            rst[row * col_num + col] = strtof(token, nullptr);
        #else
            rst[row * col_num + col] = atof(token);
        #endif

        #ifdef DEBUG
            #ifdef FLOAT32
                printf("%llu: float %.6f\n", row * col_num + col, rst[row * col_num + col]);
            #else
                printf("%llu: double %.15lf\n", row * col_num + col, rst[row * col_num + col]);
            #endif
        #endif
            token = strtok(nullptr, ",");
            col++;
        }
        
        row++;
    }

    return 0;
}

/**
 * @brief Read a 2D int matrix from an txt file into an array
 * 
 * @param fn Path_file_name of an external file
 * @param row_num The total number of rows in the matrix
 * @param col_num The total number of colums in the matrix
 * @param line_size The max size of line array
 * 
 * @return No return, fill in rst with vaules
 */
int read_2d_mat_int(char* fn, uint64_t row_num, uint64_t col_num, uint64_t line_size, int* rst)
{
    using namespace std;

    char line[line_size];
    char* token = nullptr;
    // TYPE rst[row_num * col_num];
    uint64_t row = 0;
    uint64_t col = 0;

    FILE* fp = fopen(fn, "r");

    if (fp == nullptr)
    {
        printf("Can't open %s\n", fn);
        exit(EXIT_FAILURE);
    }
    printf("%s has been opened\n", fn);

    while(fgets(line, sizeof(line), fp) != nullptr)
    {
        #ifdef DEBUG
            printf("line: %s\n", line);
        #endif
        token = strtok(line, ",");

        col = 0;
        while(token != nullptr)
        {
            // printf("elem %s | %f\n", token, atof(token));
            // printf("elem %f\n", atof(token));
            rst[row * col_num + col] = atoi(token);

            #ifdef DEBUG
            printf("%llu: int %d\n", row * col_num + col, rst[row * col_num + col]);
            #endif
            token = strtok(nullptr, ",");
            col++;
        }
        
        row++;
    }

    return 0;
}

/**
 * @brief Read a 2D uint64_t (unsigned long long int) matrix from an txt file into an array
 * 
 * @param fn Path+file_name of an external file
 * @param row_num The total number of rows in the matrix
 * @param col_num The total number of colums in the matrix
 * @param line_size The max size of line array
 * @param rst Pointer of the result array
 * 
 * @return No return, fill in rst with vaules
 * 
 * @note Usually to read graph info, e.g. vertex index and edge index
 */
int read_2d_mat_ui64(char* fn, uint64_t row_num, uint64_t col_num, uint64_t line_size, uint64_t* rst)
{
    using namespace std;

    char line[line_size];
    char* token = nullptr;
    // TYPE rst[row_num * col_num];
    uint64_t row = 0;
    uint64_t col = 0;

    FILE* fp = fopen(fn, "r");

    if (fp == nullptr)
    {
        printf("Can't open %s\n", fn);
        exit(EXIT_FAILURE);
    }
    printf("%s has been opened\n", fn);

    while(fgets(line, sizeof(line), fp) != nullptr)
    {
        #ifdef DEBUG
            printf("line: %s\n", line);
        #endif
        token = strtok(line, ",");

        col = 0;
        while(token != nullptr)
        {
            // printf("elem %s | %f\n", token, atof(token));
            // printf("elem %f\n", atof(token));

            // rst[row * col_num + col] = (uint64_t) atoll(token);
            rst[row * col_num + col] = strtoull(token, nullptr, 10);

            #ifdef DEBUG
            printf("%llu: uint64_t %llu\n", row * col_num + col, rst[row * col_num + col]);
            #endif
            token = strtok(nullptr, ",");
            col++;
        }
        
        row++;
    }

    return 0;
}

/**
 * @brief Initialize a 2D TYPE matrix using an initial vaule
 * 
 * @param row_num The total number of rows in the matrix
 * @param col_num The total number of colums in the matrix
 * @param init_vaule The initial vaule used to initialize a 2D matrix
 * @param rst Pointer of the result array, the size of which is row_num * col_num
 * 
 * @return No return, fill in rst with vaules
 */
void init_2d_mat(uint64_t row_num, uint64_t col_num, TYPE init_vaule, TYPE* rst){
    for(int i = 0; i < row_num; i++){
        for(int j = 0; j < col_num; j++){
            rst[i*col_num + j] = init_vaule;
        }
    }
}

// int read_graph_info()

/**
 * @brief Check equality of two double vaules
 * 
 * @param a Double vaule
 * @param b Double vaule
 * 
 * @return
 *  @retval 1 Equal
 *  @retval 0 Not equal
 */
int dequal(double a, double b)
{
    double c = fabs(a - b);
    if(c <= 1e-15){
        return 1;
    }

    double abs_a = fabs(a);
    double abs_b = fabs(b);
    double largest = (abs_b > abs_a) ? abs_b : abs_a;

    if(c <= largest * 1e-15){
        return 1;
    }
    return 0;
}

/**
 * @brief Check equality of two float vaules
 * 
 * @param a Float vaule
 * @param b Float vaule
 * 
 * @return
 *  @retval 1 Equal
 *  @retval 0 Not equal
 */
int fequal(float a, float b)
{
    float c = fabsf(a - b);
    if(c <= 1e-5){
        return 1;
    }

    float abs_a = fabsf(a);
    float abs_b = fabsf(b);
    float largest = (abs_b > abs_a) ? abs_b : abs_a;

    if(c <= largest * 1e-4){
        return 1;
    }
    return 0;

}

/**
 * @brief Check whether two array/matrix are the same
 * 
 * @param mat_tot_size Total size of the array/matrix
 * @param rst Actual matrix to be compared
 * @param cor_rst Correct matrix as baseline
 * 
 * @return No meaningful return
 */
int check_rst(uint64_t mat_tot_size, TYPE* rst, TYPE* cor_rst)
{
    using namespace std;
    
    printf("Checking results...\n");
    uint64_t err_flag = 0;
    for(uint64_t i=0; i<(mat_tot_size); i++)
    {   
    #ifdef FLOAT32
        if(!fequal(rst[i], cor_rst[i]))
    #else
        if(!dequal(rst[i], cor_rst[i]))
    #endif
        {
            err_flag += 1;
        #ifdef FLOAT32
            // printf("rst[%llu] incorrect: %f, %f\n", i, rst[i], cor_rst[i]);
            if(err_flag <= FEATS_OUT*2){
                printf("rst[%llu] incorrect: %.8f, %f\n", i, rst[i], cor_rst[i]);
            }
        #else
            printf("rst[%llu] incorrect: %.15lf, %.15lf\n", i, rst[i], cor_rst[i]);
        #endif
        }
    }

    if(err_flag == 0){
        printf("Test Passed!\n");
    }
    else{
        printf("Test Failed! Wrong elements number: %llu\n", err_flag);
    }
    return 0;

}

