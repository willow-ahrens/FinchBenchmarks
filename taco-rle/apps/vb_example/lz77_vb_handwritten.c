#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_LCM(_a,_b) (taco_lcm((_a),(_b)))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse, taco_mode_sparse3 } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  uint8_t*     fill_value;    // tensor fill value
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
int omp_get_thread_num() { return 0; }
int omp_get_max_threads() { return 1; }
int cmp(const void *a, const void *b) {
  return *((const int*)a) - *((const int*)b);
}
int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                                  int32_t* dimensions, int32_t* mode_ordering,
                                  taco_mode_t* mode_types) {
  taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
  t->order         = order;
  t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
  t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
  t->csize         = csize;
  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = mode_ordering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse:
        t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse3:
        t->indices[i] = (uint8_t **) malloc(3 * sizeof(uint8_t **));
        break;
    }
  }
  return t;
}
void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free(t->indices[i]);
  }
  free(t->indices);
  free(t->dimensions);
  free(t->mode_ordering);
  free(t->mode_types);
  free(t);
}
unsigned int taco_gcd(unsigned int u, unsigned int v) {
  // TODO: https://lemire.me/blog/2013/12/26/fastest-way-to-compute-the-greatest-common-divisor/
  int shift;
  if (u == 0)
    return v;
  if (v == 0)
    return u;
  for (shift = 0; ((u | v) & 1) == 0; ++shift) {
    u >>= 1;
    v >>= 1;
  }

  while ((u & 1) == 0)
    u >>= 1;

  do {
    while ((v & 1) == 0)
      v >>= 1;
    if (u > v) {
      unsigned int t = v;
      v = u;
      u = t;
    }
    v = v - u;
  } while (v != 0);
  return u << shift;
}
int taco_lcm(int a, int b) {
  int temp = (int) taco_gcd((unsigned)a, (unsigned)b);
  return temp ? (a / temp * b) : 0;
}
#endif

#define TYPE int

// merge({B_i_blk, D_i_blk}, forall(i, C(i) = B(B_i_blk, i) + D(D_i_blk, i)))
int compute(taco_tensor_t *C, taco_tensor_t *B, taco_tensor_t *D) {
  TYPE* restrict C_vals = (TYPE*)(C->vals);
  TYPE C_fill_value = *((TYPE*)(C->fill_value));
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);
  int* restrict B2_run = (int*)(B->indices[1][2]);
  int* restrict B2_dist = (int*)(B->indices[1][3]);
  TYPE* restrict B_vals = (TYPE*)(B->vals);
  TYPE B_fill_value = *((TYPE*)(B->fill_value));
  int D1_dimension = (int)(D->dimensions[0]);
  int* restrict D2_pos = (int*)(D->indices[1][0]);
  int* restrict D2_crd = (int*)(D->indices[1][1]);
  int* restrict D2_run = (int*)(D->indices[1][2]);
  int* restrict D2_dist = (int*)(D->indices[1][3]);
  TYPE* restrict D_vals = (TYPE*)(D->vals);
  TYPE D_fill_value = *((TYPE*)(D->fill_value));

  for (int32_t pC = 0; pC < 6; pC++) {
    C_vals[pC] = C_fill_value;
  }

  int32_t B_fill_len = 1;
  int32_t B_fill_index = 0;
  int32_t D_fill_len = 1;
  int32_t D_fill_index = 0;


  int32_t IB_crd = 0;
  int32_t ID_crd = 0;

  int32_t iB_pos = B2_pos[IB_crd];
  int32_t pB2_end = B2_pos[(IB_crd + 1)];
  int32_t iD_pos = D2_pos[ID_crd];
  int32_t pD2_end = D2_pos[(ID_crd + 1)];
  while (IB_crd < B1_dimension && ID_crd < D1_dimension){
    while (iB_pos < pB2_end && iD_pos < pD2_end) {
      int32_t iB_crd = B2_crd[IB_crd] + (iB_pos - B2_pos[IB_crd]);
      int32_t iD_crd = D2_crd[ID_crd] + (iD_pos - D2_pos[ID_crd]);
      int32_t i_crd = TACO_MIN(iB_crd,iD_crd);
      if (iB_crd == i_crd && iD_crd == i_crd) {
        C_vals[i_crd] = C_vals[i_crd] + (B_vals[iB_pos] + D_vals[iD_pos]);
      }
      else if (iB_crd == i_crd) {
        C_vals[i_crd] = C_vals[i_crd] + (B_vals[iB_pos] + D_fill_value);
      }
      else {
        C_vals[i_crd] = C_vals[i_crd] + (B_fill_value + D_vals[iD_pos]);
      }
      iB_pos += (int32_t)(iB_crd == i_crd);
      iD_pos += (int32_t)(iD_crd == i_crd);

      B_fill_value = B_fill_region[B_fill_index];
      B_fill_index = (B_fill_index + 1) % B_fill_len;
      D_fill_value = D_fill_region[D_fill_index];
      D_fill_index = (D_fill_index + 1) % D_fill_len;
    }
    if (iB_pos == pB2_end){
      IB_crd++;
      if (IB_crd < B1_dimension) {
        iB_pos = B2_pos[IB_crd];
        pB2_end = B2_pos[(IB_crd + 1)];
      }
    }
    if (iD_pos == pD2_end){
      ID_crd++;
      if (ID_crd < D1_dimension) {
        iD_pos = D2_pos[ID_crd];
        pD2_end = D2_pos[(ID_crd + 1)];
      }
    }
  }
  while (IB_crd < B1_dimension){
    int32_t iB_pos = B2_pos[IB_crd];
    int32_t pB2_end = B2_pos[(IB_crd + 1)];
    while (iB_pos < pB2_end) {
      int32_t i_crd = B2_crd[IB_crd] + (iB_pos - B2_pos[IB_crd]);
      C_vals[i_crd] = C_vals[i_crd] + (B_vals[iB_pos] + D_fill_value);
      iB_pos++;
      // TODO: fills?
    }
    IB_crd++;
  }
  while (ID_crd < D1_dimension){
    int32_t iD_pos = D2_pos[ID_crd];
    int32_t pD2_end = D2_pos[(ID_crd + 1)];
    while (iD_pos < pD2_end) {
      int32_t i_crd = D2_crd[ID_crd] + (iD_pos - D2_pos[ID_crd]);
      C_vals[i_crd] = C_vals[i_crd] + (B_fill_value + D_vals[iD_pos]);
      iD_pos++;
      // TODO: fills?
    }
    ID_crd++;
  }
  // TODO: rest of tensor?
  return 0;
}

#include "vb_handwritten.h"
int _shim_compute(void** parameterPack) {
  return compute((taco_tensor_t*)(parameterPack[0]), (taco_tensor_t*)(parameterPack[1]),  (taco_tensor_t*)(parameterPack[2]));
}
