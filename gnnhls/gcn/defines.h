#ifndef __DEFINES_H__
#define __DEFINES_H__

#include <stdio.h>
#include <stdint.h>

// #include <ap_fixed.h>

#define FLOAT32

#ifdef FLOAT32
    #define TYPE float
    // #define TYPE ap_fixed<16,9>
#else
    #define TYPE double
#endif

#define CHECK
#define SCATTER_TEMP //only works for CPU verification system 

#define ACTIVATION
// #define DEBUG

#define PREFETCH
// #define PREFETCH_DEBUG
// depth of prefetching list in the prefetcher
#define PF_DEPTH 30

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

// #define PRAGMA_SUB(x) _Pragma (#x)
// #define DO_PRAGMA(x) PRAGMA_SUB(x)

#endif