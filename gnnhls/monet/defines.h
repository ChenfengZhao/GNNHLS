#ifndef __DEFINES_H__
#define __DEFINES_H__

#include <stdio.h>
#include <stdint.h>

#define FLOAT32

#ifdef FLOAT32
    #define TYPE float
#else
    #define TYPE double
#endif

#define NEG_SLOPE 0.2

#define CHECK
#define SCATTER_TEMP //only works for CPU verification system 

#define ACTIVATION
// #define DEBUG
// #define DEBUG_READFILE
// #define CHECK_INTER_RST // check intermidiate results with correct inter rst stored in files



// Larger than necessary for small graphs, but appropriate for large ones
typedef uint64_t edge_index_t;
typedef uint64_t node_index_t;
// typedef int8_t level_t;

typedef uint32_t level_t;
typedef uint64_t addr_t;
// typedef uint64_t weight_t;

typedef uint32_t length_t;

// typedef struct edge_t_struct {
//   // These fields are common in practice, but we elect not to use them.
//   //weight_t weight;
//   //node_index_t src;
//   node_index_t dst;
// } edge_t;

// typedef struct node_t_struct {
//   edge_index_t edge_begin;
//   edge_index_t edge_end;
// } node_t;

#endif