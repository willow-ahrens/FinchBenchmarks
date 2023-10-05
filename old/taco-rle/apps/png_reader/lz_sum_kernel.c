#include "lz_sum_kernel.h"

int compute_kernel(taco_tensor_t *expected_, taco_tensor_t *T1, taco_tensor_t *T1001) {
  int* restrict expected_1_pos = (int*)(expected_->indices[0][0]);
  uint8_t* restrict expected__vals = (uint8_t*)(expected_->vals);
  int* restrict T11_pos = (int*)(T1->indices[0][0]);
  uint8_t* restrict T1_vals = (uint8_t*)(T1->vals);
  uint8_t T1_fill_value = *((uint8_t*)(T1->fill_value));
  uint8_t* T1_fill_region = ((uint8_t*)(T1->fill_value));
  int T10011_dimension = (int)(T1001->dimensions[0]);
  int* restrict T10011_pos = (int*)(T1001->indices[0][0]);
  uint8_t* restrict T1001_vals = (uint8_t*)(T1001->vals);
  uint8_t T1001_fill_value = *((uint8_t*)(T1001->fill_value));
  uint8_t* T1001_fill_region = ((uint8_t*)(T1001->fill_value));

  int32_t T1_fill_len = 1;

  int32_t T1_fill_index = 0;

  int32_t T1001_fill_len = 1;

  int32_t T1001_fill_index = 0;

  expected_1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
  expected_1_pos[0] = 0;
  int32_t expected_1_cnt_pos = 0;
  int32_t expected_1_cnt_val = 0;
  bool expected_1_is_filling = 0;
  int32_t iexpected__pos = 0;
  int32_t expected__capacity = 1048576;
  expected__vals = (uint8_t*)malloc(sizeof(uint8_t) * expected__capacity);




  int32_t i_crd = 0;
  int32_t T11_coord = 0;
  int32_t T11_dist = 0;
  int32_t T11_run = 0;
  int32_t T11_found_cnt = 0;
  int32_t T11_pos_coord = T11_coord;
  int32_t iT1_pos = T11_pos[0];
  int32_t pT11_end = T11_pos[1];
  int32_t T10011_coord = 0;
  int32_t T10011_dist = 0;
  int32_t T10011_run = 0;
  int32_t T10011_found_cnt = 0;
  int32_t T10011_pos_coord = T10011_coord;
  int32_t iT1001_pos = T10011_pos[0];
  int32_t pT10011_end = T10011_pos[1];

  int32_t iT1_count = T11_found_cnt;
  int32_t iT1001_count = T10011_found_cnt;
  int32_t iT1_crd = 0;
  int32_t iT1001_crd = 0;
  while (iT1_pos < pT11_end && iT1001_pos < pT10011_end) {
    if (!(bool)iT1_count) {
      T11_pos_coord = T11_coord;
      if ((((uint16_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] >> 15 & 1) == 0) {
        T11_found_cnt = ((uint16_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] & 32767;
        iT1_pos += 2;
        T11_coord += T11_found_cnt;
      }
      else {
        T11_found_cnt = 0;
        T11_dist = ((uint16_t*)&(((uint8_t*)T1_vals)[(iT1_pos + 2)]))[0];
        T11_run = ((uint16_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] & 32767;
        iT1_pos += 4;
        T11_coord += T11_run;
        T11_pos_coord = T11_coord;
      }
      if (!(bool)T11_found_cnt) {
        T1_fill_len = TACO_MIN(T11_dist,T11_run);
        if (T1_fill_len == 1) T1_fill_index = 0;

        else {
          T1_fill_index = 1;
        }
        T1_fill_region = &(((uint8_t*)T1_vals)[((iT1_pos - 4) - T11_dist)]);
        T1_fill_value = T1_fill_region[0];
      }
      iT1_crd = T11_pos_coord;
      iT1_count = (int32_t)T11_found_cnt;
    }
    if (!(bool)iT1001_count) {
      T10011_pos_coord = T10011_coord;
      if ((((uint16_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0] >> 15 & 1) == 0) {
        T10011_found_cnt = ((uint16_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0] & 32767;
        iT1001_pos += 2;
        T10011_coord += T10011_found_cnt;
      }
      else {
        T10011_found_cnt = 0;
        T10011_dist = ((uint16_t*)&(((uint8_t*)T1001_vals)[(iT1001_pos + 2)]))[0];
        T10011_run = ((uint16_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0] & 32767;
        iT1001_pos += 4;
        T10011_coord += T10011_run;
        T10011_pos_coord = T10011_coord;
      }
      if (!(bool)T10011_found_cnt) {
        T1001_fill_len = TACO_MIN(T10011_dist,T10011_run);
        if (T1001_fill_len == 1) T1001_fill_index = 0;

        else {
          T1001_fill_index = 1;
        }
        T1001_fill_region = &(((uint8_t*)T1001_vals)[((iT1001_pos - 4) - T10011_dist)]);
        T1001_fill_value = T1001_fill_region[0];
      }
      iT1001_crd = T10011_pos_coord;
      iT1001_count = (int32_t)T10011_found_cnt;
    }
    if (((iT1_crd == i_crd && iT1001_crd == i_crd) && T11_found_cnt) && T10011_found_cnt) {
      int32_t for_end = TACO_MIN(iT1_count,iT1001_count);
      if (expected_1_is_filling && expected_1_cnt_val+for_end < 32767) {
        if (expected__capacity <= iexpected__pos+for_end) {
          expected__capacity = (iexpected__pos+for_end)*2;
          expected__vals = (uint8_t *) realloc(expected__vals, sizeof(uint8_t) * (expected__capacity));
        }

        for (int32_t l = 0; l < for_end; l++) {
          ((uint8_t *) &(((uint8_t *) expected__vals)[iexpected__pos]))[0] =
                  0.500000 * ((uint8_t *) &(((uint8_t *) T1_vals)[iT1_pos]))[0] +
                  0.500000 * ((uint8_t *) &(((uint8_t *) T1001_vals)[iT1001_pos]))[0];
          iexpected__pos++;
          iT1_pos++;
          iT1001_pos++;
          i_crd++;
        }
        expected_1_cnt_val+= for_end;
      } else {
        for (int32_t l = 0; l < for_end; l++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t *) realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t *) &(((uint8_t *) expected__vals)[iexpected__pos]))[0] =
                  0.500000 * ((uint8_t *) &(((uint8_t *) T1_vals)[iT1_pos]))[0] +
                  0.500000 * ((uint8_t *) &(((uint8_t *) T1001_vals)[iT1001_pos]))[0];
          uint8_t curr_value = ((uint8_t *) &(((uint8_t *) expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          } else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t *) &(((uint8_t *) expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t *) realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t *) &(((uint8_t *) expected__vals)[(iexpected__pos + 2)]))[0] = curr_value;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          iT1_pos++;
          iT1001_pos++;
          i_crd++;
        }
      }
      iT1_count -= for_end;
      iT1_crd += for_end;
      iT1001_count -= for_end;
      iT1001_crd += for_end;
      continue;
    }
    else if (iT1_crd == i_crd && T11_found_cnt) {
      int32_t for_end0 = TACO_MIN(iT1_count,iT1001_crd - i_crd);
      if (T1001_fill_len == 1)
        for (int32_t l0 = 0; l0 < for_end0; l0++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * ((uint8_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] + 0.500000 * T1001_fill_value;
          uint8_t curr_value0 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value0;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          iT1_pos++;
          i_crd++;
        }

      else {
        for (int32_t l0 = 0; l0 < for_end0; l0++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * ((uint8_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] + 0.500000 * T1001_fill_value;
          uint8_t curr_value0 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value0;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1001_fill_value = T1001_fill_region[T1001_fill_index];
          T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
          iT1_pos++;
          i_crd++;
        }
      }
      iT1_count -= for_end0;
      iT1_crd += for_end0;
      continue;
    }
    else if (iT1001_crd == i_crd && T10011_found_cnt) {
      int32_t for_end1 = TACO_MIN(iT1001_count,iT1_crd - i_crd);
      if (T1_fill_len == 1)
        for (int32_t l1 = 0; l1 < for_end1; l1++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * ((uint8_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0];
          uint8_t curr_value1 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value1;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          iT1001_pos++;
          i_crd++;
        }

      else {
        for (int32_t l1 = 0; l1 < for_end1; l1++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * ((uint8_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0];
          uint8_t curr_value1 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value1;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1_fill_value = T1_fill_region[T1_fill_index];
          T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
          iT1001_pos++;
          i_crd++;
        }
      }
      iT1001_count -= for_end1;
      iT1001_crd += for_end1;
      continue;
    }
    else {
      int32_t lengthsLcm = TACO_LCM(T1_fill_len,T1001_fill_len);
      int32_t coordMin = TACO_MIN(iT1_crd,iT1001_crd);
      if (coordMin - i_crd <= lengthsLcm) {
        while (i_crd < coordMin) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * T1001_fill_value;
          uint8_t curr_value2 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value2;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1_fill_value = T1_fill_region[T1_fill_index];
          T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
          T1001_fill_value = T1001_fill_region[T1001_fill_index];
          T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
          i_crd++;
        }
        continue;
      }
      else {
        int32_t loopBound = i_crd + lengthsLcm;
        int32_t startVar = iexpected__pos;
        while (i_crd < loopBound) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * T1001_fill_value;
          uint8_t curr_value2 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value2;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1_fill_value = T1_fill_region[T1_fill_index];
          T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
          T1001_fill_value = T1001_fill_region[T1001_fill_index];
          T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
          i_crd++;
        }
        int32_t runValue = coordMin - i_crd;
        if (expected_1_is_filling && expected_1_cnt_val) {
          ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
        }
        expected_1_is_filling = 0;
        ((uint16_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = iexpected__pos - startVar;
        ((uint16_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = runValue | 32768;
        iexpected__pos += 4;
        T1_fill_value = T1_fill_region[T1_fill_index];
        T1_fill_index = (T1_fill_index + runValue) % T1_fill_len;
        T1001_fill_value = T1001_fill_region[T1001_fill_index];
        T1001_fill_index = (T1001_fill_index + runValue) % T1001_fill_len;
        iT1_pos += (int32_t)(iT1_crd == i_crd);
        iT1_crd += (int32_t)(iT1_crd == i_crd);
        iT1001_pos += (int32_t)(iT1001_crd == i_crd);
        iT1001_crd += (int32_t)(iT1001_crd == i_crd);
        i_crd = coordMin;
        continue;
      }
    }
    T1_fill_value = T1_fill_region[T1_fill_index];
    T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
    T1001_fill_value = T1001_fill_region[T1001_fill_index];
    T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
    iT1_pos += (int32_t)(iT1_crd == i_crd);
    iT1_crd += (int32_t)(iT1_crd == i_crd);
    iT1001_pos += (int32_t)(iT1001_crd == i_crd);
    iT1001_crd += (int32_t)(iT1001_crd == i_crd);
    i_crd++;
  }
  while (iT1_pos < pT11_end) {
    if (!(bool)iT1_count) {
      T11_pos_coord = T11_coord;
      if ((((uint16_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] >> 15 & 1) == 0) {
        T11_found_cnt = ((uint16_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] & 32767;
        iT1_pos += 2;
        T11_coord += T11_found_cnt;
      }
      else {
        T11_found_cnt = 0;
        T11_dist = ((uint16_t*)&(((uint8_t*)T1_vals)[(iT1_pos + 2)]))[0];
        T11_run = ((uint16_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] & 32767;
        iT1_pos += 4;
        T11_coord += T11_run;
        T11_pos_coord = T11_coord;
      }
      if (!(bool)T11_found_cnt) {
        T1_fill_len = TACO_MIN(T11_dist,T11_run);
        if (T1_fill_len == 1) T1_fill_index = 0;

        else {
          T1_fill_index = 1;
        }
        T1_fill_region = &(((uint8_t*)T1_vals)[((iT1_pos - 4) - T11_dist)]);
        T1_fill_value = T1_fill_region[0];
      }
      iT1_crd = T11_pos_coord;
      iT1_count = (int32_t)T11_found_cnt;
    }
    if (iT1_crd == i_crd && T11_found_cnt) {
      int32_t for_end2 = iT1_count;
      if (T1001_fill_len == 1)
        for (int32_t l2 = 0; l2 < iT1_count; l2++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * ((uint8_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] + 0.500000 * T1001_fill_value;
          uint8_t curr_value3 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value3;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          iT1_pos++;
          i_crd++;
        }

      else {
        for (int32_t l2 = 0; l2 < iT1_count; l2++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * ((uint8_t*)&(((uint8_t*)T1_vals)[iT1_pos]))[0] + 0.500000 * T1001_fill_value;
          uint8_t curr_value3 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value3;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1001_fill_value = T1001_fill_region[T1001_fill_index];
          T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
          iT1_pos++;
          i_crd++;
        }
      }
      iT1_count -= iT1_count;
      iT1_crd += for_end2;
      continue;
    }
    else {
      int32_t lengthsLcm0 = TACO_LCM(T1_fill_len,T1001_fill_len);
      int32_t coordMin0 = iT1_crd;
      if (coordMin0 - i_crd <= lengthsLcm0) {
        while (i_crd < coordMin0) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * T1001_fill_value;
          uint8_t curr_value4 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value4;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1_fill_value = T1_fill_region[T1_fill_index];
          T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
          T1001_fill_value = T1001_fill_region[T1001_fill_index];
          T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
          i_crd++;
        }
        continue;
      }
      else {
        int32_t loopBound0 = i_crd + lengthsLcm0;
        int32_t startVar0 = iexpected__pos;
        while (i_crd < loopBound0) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * T1001_fill_value;
          uint8_t curr_value4 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value4;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1_fill_value = T1_fill_region[T1_fill_index];
          T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
          T1001_fill_value = T1001_fill_region[T1001_fill_index];
          T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
          i_crd++;
        }
        int32_t runValue0 = coordMin0 - i_crd;
        if (expected_1_is_filling && expected_1_cnt_val) {
          ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
        }
        expected_1_is_filling = 0;
        ((uint16_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = iexpected__pos - startVar0;
        ((uint16_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = runValue0 | 32768;
        iexpected__pos += 4;
        T1_fill_value = T1_fill_region[T1_fill_index];
        T1_fill_index = (T1_fill_index + runValue0) % T1_fill_len;
        T1001_fill_value = T1001_fill_region[T1001_fill_index];
        T1001_fill_index = (T1001_fill_index + runValue0) % T1001_fill_len;
        iT1_pos++;
        i_crd = coordMin0;
        continue;
      }
    }
    T1_fill_value = T1_fill_region[T1_fill_index];
    T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
    T1001_fill_value = T1001_fill_region[T1001_fill_index];
    T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
    iT1_pos += (int32_t)(iT1_crd == i_crd);
    iT1_crd += (int32_t)(iT1_crd == i_crd);
    i_crd++;
  }
  while (iT1001_pos < pT10011_end) {
    if (!(bool)iT1001_count) {
      T10011_pos_coord = T10011_coord;
      if ((((uint16_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0] >> 15 & 1) == 0) {
        T10011_found_cnt = ((uint16_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0] & 32767;
        iT1001_pos += 2;
        T10011_coord += T10011_found_cnt;
      }
      else {
        T10011_found_cnt = 0;
        T10011_dist = ((uint16_t*)&(((uint8_t*)T1001_vals)[(iT1001_pos + 2)]))[0];
        T10011_run = ((uint16_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0] & 32767;
        iT1001_pos += 4;
        T10011_coord += T10011_run;
        T10011_pos_coord = T10011_coord;
      }
      if (!(bool)T10011_found_cnt) {
        T1001_fill_len = TACO_MIN(T10011_dist,T10011_run);
        if (T1001_fill_len == 1) T1001_fill_index = 0;

        else {
          T1001_fill_index = 1;
        }
        T1001_fill_region = &(((uint8_t*)T1001_vals)[((iT1001_pos - 4) - T10011_dist)]);
        T1001_fill_value = T1001_fill_region[0];
      }
      iT1001_crd = T10011_pos_coord;
      iT1001_count = (int32_t)T10011_found_cnt;
    }
    if (iT1001_crd == i_crd && T10011_found_cnt) {
      int32_t for_end3 = iT1001_count;
      if (T1_fill_len == 1)
        for (int32_t l3 = 0; l3 < iT1001_count; l3++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * ((uint8_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0];
          uint8_t curr_value5 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value5;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          iT1001_pos++;
          i_crd++;
        }

      else {
        for (int32_t l3 = 0; l3 < iT1001_count; l3++) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * ((uint8_t*)&(((uint8_t*)T1001_vals)[iT1001_pos]))[0];
          uint8_t curr_value5 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value5;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1_fill_value = T1_fill_region[T1_fill_index];
          T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
          iT1001_pos++;
          i_crd++;
        }
      }
      iT1001_count -= iT1001_count;
      iT1001_crd += for_end3;
      continue;
    }
    else {
      int32_t lengthsLcm1 = TACO_LCM(T1_fill_len,T1001_fill_len);
      int32_t coordMin1 = iT1001_crd;
      if (coordMin1 - i_crd <= lengthsLcm1) {
        while (i_crd < coordMin1) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * T1001_fill_value;
          uint8_t curr_value6 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value6;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1_fill_value = T1_fill_region[T1_fill_index];
          T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
          T1001_fill_value = T1001_fill_region[T1001_fill_index];
          T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
          i_crd++;
        }
        continue;
      }
      else {
        int32_t loopBound1 = i_crd + lengthsLcm1;
        int32_t startVar1 = iexpected__pos;
        while (i_crd < loopBound1) {
          if (expected__capacity <= iexpected__pos) {
            expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
            expected__capacity *= 2;
          }
          ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * T1001_fill_value;
          uint8_t curr_value6 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
          if (expected_1_is_filling && expected_1_cnt_val < 32767) {
            expected_1_cnt_val++;
          }
          else {
            if (expected_1_is_filling && expected_1_cnt_val) {
              ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
            }
            expected_1_is_filling = 1;
            expected_1_cnt_pos = iexpected__pos;
            if (expected__capacity <= iexpected__pos + 1) {
              expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
              expected__capacity *= 2;
            }
            expected_1_cnt_val = 1;
            ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value6;
            iexpected__pos += 2;
          }
          iexpected__pos++;
          T1_fill_value = T1_fill_region[T1_fill_index];
          T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
          T1001_fill_value = T1001_fill_region[T1001_fill_index];
          T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
          i_crd++;
        }
        int32_t runValue1 = coordMin1 - i_crd;
        if (expected_1_is_filling && expected_1_cnt_val) {
          ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
        }
        expected_1_is_filling = 0;
        ((uint16_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = iexpected__pos - startVar1;
        ((uint16_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = runValue1 | 32768;
        iexpected__pos += 4;
        T1_fill_value = T1_fill_region[T1_fill_index];
        T1_fill_index = (T1_fill_index + runValue1) % T1_fill_len;
        T1001_fill_value = T1001_fill_region[T1001_fill_index];
        T1001_fill_index = (T1001_fill_index + runValue1) % T1001_fill_len;
        iT1001_pos++;
        i_crd = coordMin1;
        continue;
      }
    }
    T1_fill_value = T1_fill_region[T1_fill_index];
    T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
    T1001_fill_value = T1001_fill_region[T1001_fill_index];
    T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
    iT1001_pos += (int32_t)(iT1001_crd == i_crd);
    iT1001_crd += (int32_t)(iT1001_crd == i_crd);
    i_crd++;
  }
  while (i_crd < T10011_dimension && i_crd >= 0) {
    if (expected__capacity <= iexpected__pos) {
      expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
      expected__capacity *= 2;
    }
    ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0] = 0.500000 * T1_fill_value + 0.500000 * T1001_fill_value;
    uint8_t curr_value7 = ((uint8_t*)&(((uint8_t*)expected__vals)[iexpected__pos]))[0];
    if (expected_1_is_filling && expected_1_cnt_val < 32767) {
      expected_1_cnt_val++;
    }
    else {
      if (expected_1_is_filling && expected_1_cnt_val) {
        ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
      }
      expected_1_is_filling = 1;
      expected_1_cnt_pos = iexpected__pos;
      if (expected__capacity <= iexpected__pos + 1) {
        expected__vals = (uint8_t*)realloc(expected__vals, sizeof(uint8_t) * (expected__capacity * 2));
        expected__capacity *= 2;
      }
      expected_1_cnt_val = 1;
      ((uint8_t*)&(((uint8_t*)expected__vals)[(iexpected__pos + 2)]))[0] = curr_value7;
      iexpected__pos += 2;
    }
    iexpected__pos++;
    T1_fill_value = T1_fill_region[T1_fill_index];
    T1_fill_index = (T1_fill_index + 1) % T1_fill_len;
    T1001_fill_value = T1001_fill_region[T1001_fill_index];
    T1001_fill_index = (T1001_fill_index + 1) % T1001_fill_len;
    i_crd++;
  }

  expected_1_pos[1] = iexpected__pos;

  if (expected_1_is_filling && expected_1_cnt_val) {
    ((uint16_t*)&(((uint8_t*)expected__vals)[expected_1_cnt_pos]))[0] = expected_1_cnt_val;
  }

  expected_->indices[0][0] = (uint8_t*)(expected_1_pos);
  expected_->vals = (uint8_t*)expected__vals;
  return 0;
}

