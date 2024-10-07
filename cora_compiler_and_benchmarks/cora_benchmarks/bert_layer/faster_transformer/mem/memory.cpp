#include <fstream>
#include <iostream>
#include <unordered_map>

typedef int Tensor;

std::unordered_map<Tensor, size_t> alloc_map;
int max_memory = 0;
int current_memory = 0;

int count = 0;

void Alloc(Tensor* t, size_t size) {
  *t = count++;
  alloc_map[*t] = size;
  current_memory += size;
  if (current_memory > max_memory) {
    max_memory = current_memory;
  }
}

void Free(Tensor t) { current_memory -= alloc_map[t]; }

void BERTforward(int batch_size, int seq_len, int hidden_dim, int head_num, int size_per_head) {
  Tensor d_from_tensor;
  Tensor d_transformer_out;
  Tensor d_attr_kernel_Q;
  Tensor d_attr_bias_Q;
  Tensor d_attr_bias_K;
  Tensor d_attr_bias_V;
  Tensor d_attr_output_kernel;
  Tensor d_attr_output_bias;
  Tensor d_attr_output_layernorm_beta;
  Tensor d_attr_output_layernorm_gamma;
  Tensor d_inter_kernel;
  Tensor d_inter_bias;
  Tensor d_output_kernel;
  Tensor d_output_bias;
  Tensor d_output_layernorm_beta;
  Tensor d_output_layernorm_gamma;

  Alloc(&d_from_tensor, batch_size * seq_len * hidden_dim);
  Alloc(&d_attr_kernel_Q, hidden_dim * hidden_dim * 3);
  Alloc(&d_attr_bias_Q, hidden_dim);
  Alloc(&d_attr_bias_K, hidden_dim);
  Alloc(&d_attr_bias_V, hidden_dim);
  Alloc(&d_attr_output_kernel, hidden_dim * hidden_dim);
  Alloc(&d_attr_output_bias, hidden_dim);
  Alloc(&d_attr_output_layernorm_beta, hidden_dim);
  Alloc(&d_attr_output_layernorm_gamma, hidden_dim);
  Alloc(&d_inter_kernel, hidden_dim * hidden_dim * 4);
  Alloc(&d_inter_bias, hidden_dim * 4);
  Alloc(&d_output_kernel, hidden_dim * hidden_dim * 4);
  Alloc(&d_output_bias, hidden_dim);
  Alloc(&d_output_layernorm_beta, hidden_dim);
  Alloc(&d_output_layernorm_gamma, hidden_dim);

  Tensor query_buf_;
  Tensor key_buf_;
  Tensor value_buf_;
  Tensor q_buf_;
  Tensor k_buf_;
  Tensor v_buf_;
  Tensor qk_buf_;
  Tensor transpose_dst_;
  Tensor attr_out_buf_;
  Tensor attr_matmul_buf_;
  Tensor inter_matmul_buf_;

  {
    Alloc(&query_buf_, batch_size * head_num * seq_len * size_per_head);

    // // Used tensors:
    // //  In: param_.self_attention.query_weight.kernel
    // //  In: from_tensor
    // //  Out: query_buf_ (batch_size * head_num * seq_len * size_per_head)
    // cublasMM_cublasLtMM_wrapper(
    //     param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
    //     param_.self_attention.query_weight.kernel, AType_, n, from_tensor, BType_, k, &beta,
    //     (DataType_*)query_buf_, CType_, n, param_.stream, cublasAlgoMap_, sm_,
    //     cublas_workspace_);

    Alloc(&key_buf_, batch_size * head_num * seq_len * size_per_head);

    // // Used tensors:
    // //  In: param_.self_attention.key_weight.kernel
    // //  In: to_tensor
    // //  Out: key_buf_ (batch_size * head_num * seq_len * size_per_head)
    // cublasMM_cublasLtMM_wrapper(
    //     param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
    //     param_.self_attention.key_weight.kernel, AType_, n, to_tensor, BType_, k, &beta,
    //     (DataType_*)key_buf_, CType_, n, param_.stream, cublasAlgoMap_, sm_, cublas_workspace_);

    Alloc(&value_buf_, batch_size * head_num * seq_len * size_per_head);

    // // Used tensors:
    // //  In: param_.self_attention.value_weight.kernel
    // //  In: to_tensor
    // //  Out: value_buf_ (batch_size * head_num * seq_len * size_per_head)
    // cublasMM_cublasLtMM_wrapper(
    //     param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
    //     param_.self_attention.value_weight.kernel, AType_, n, to_tensor, BType_, k, &beta,
    //     (DataType_*)value_buf_, CType_, n, param_.stream, cublasAlgoMap_, sm_,
    //     cublas_workspace_);

    // multiHeadAttr_nofuse_kernelLauncher(
    //     param_.stream, param_.cublas_handle, param_.cublaslt_handle, query_buf_,
    //     param_.self_attention.query_weight.bias, key_buf_, param_.self_attention.key_weight.bias,
    //     value_buf_, param_.self_attention.value_weight.bias, param_.attr_mask, param_.attr_out,
    //     batch_size, seq_len, head_num, size_per_head, int8_mode_, scalar) {
    //   const int k = head_num * size_per_head;

    Alloc(&q_buf_, batch_size * head_num * seq_len * size_per_head);
    Alloc(&k_buf_, batch_size * head_num * seq_len * size_per_head);
    Alloc(&v_buf_, batch_size * head_num * seq_len * size_per_head);

    // // Used tensors:
    // //  query_buf_
    // //  key_buf_
    // //  value_buf_
    // //  Out: q_buf_ (batch_size * head_num * seq_len * size_per_head)
    // //  Out: k_buf_ (batch_size * head_num * seq_len * size_per_head)
    // //  Out: v_buf_ (batch_size * head_num * seq_len * size_per_head)
    // //  In: bias_Q
    // //  In: bias_K
    // //  In: bias_V
    // if (param_.valid_word_num == batch_size * seq_len) {
    //   add_QKV_bias_transpose_kernelLauncher(q_buf_, k_buf_, v_buf_, Q, bias_Q, K, bias_K, V,
    //                                         bias_V, batch_size, seq_len, head_num,
    //                                         size_per_head, stream);
    // } else {
    //   add_QKV_bias_rebuild_padding_kernelLauncher(
    //       Q, bias_Q, K, bias_K, V, bias_V, q_buf_, k_buf_, v_buf_, batch_size, seq_len, head_num,
    //       size_per_head, param_.valid_word_num, param_.sequence_id_offset, stream);
    // }

    Free(query_buf_);
    Free(key_buf_);
    Free(value_buf_);

    Alloc(&qk_buf_, batch_size * head_num * seq_len * seq_len);

    // // Used tensors:
    // //  In: k_buf_
    // //  In: q_buf_
    // //  Out: qk_buf_
    // check_cuda_error(cublasGemmStridedBatchedEx(
    //     cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, size_per_head, &alpha, k_buf_,
    //     AType_, size_per_head, seq_len * size_per_head, q_buf_, BType_, size_per_head,
    //     seq_len * size_per_head, &beta, qk_buf_, CType_, seq_len, seq_len * seq_len,
    //     batch_size * head_num, computeType_, static_cast<cublasGemmAlgo_t>(cublasBmmAlgo_[0])));

    // attn_softmax_kernelLauncher(qk_buf_, attr_mask, batch_size, seq_len, head_num, scalar,
    //                             stream);

    Free(q_buf_);
    Free(k_buf_);

    Alloc(&transpose_dst_, batch_size * head_num * seq_len * size_per_head);

    // // Used tensors:
    // //  In: v_buf_
    // //  In: qk_buf_ (batch_size * head_num * seq_len * seq_len)
    // //  Out: transpose_dst_
    // check_cuda_error(cublasGemmStridedBatchedEx(
    //     cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_per_head, seq_len, seq_len, &alpha, v_buf_,
    //     AType_, size_per_head, seq_len * size_per_head, qk_buf_, BType_, seq_len,
    //     seq_len * seq_len, &beta, transpose_dst_, CType_, size_per_head, seq_len * size_per_head,
    //     batch_size * head_num, computeType_, static_cast<cublasGemmAlgo_t>(cublasBmmAlgo_[1])));

    Free(qk_buf_);
    Free(v_buf_);

    Alloc(&attr_out_buf_, batch_size * seq_len * head_num * size_per_head);

    // // Used tensors:
    // //  In: transpose_dst_ ()
    // //  Out: attr_out_buf_
    // if (param_.valid_word_num == batch_size * seq_len) {
    //   transpose_kernelLauncher(transpose_dst_, dst, batch_size, seq_len, head_num, size_per_head,
    //                            stream);
    // } else {
    //   transpose_rebuild_padding_kernelLauncher(transpose_dst_, dst, param_.valid_word_num,
    //                                            batch_size, seq_len, head_num, size_per_head,
    //                                            param_.sequence_id_offset, stream);
    // }

    Free(transpose_dst_);
  }
  // }

  // DataType_ alpha = (DataType_)1.0f;
  // DataType_ beta = (DataType_)0.0f;
  // const int m =
  //     param_.sequence_id_offset == nullptr ? batch_size * seq_len : param_.valid_word_num;
  // int k = head_num * size_per_head;
  // int n = k;

  Alloc(&attr_matmul_buf_, batch_size * seq_len * head_num * size_per_head);

  // // Used tensors:
  // //  In: param_.self_attention.attention_output_weight.kernel
  // //  In: attr_out_buf_ (batch_size * seq_len * head_num * size_per_head)
  // //  Out: attr_matmul_buf_ (batch_size * seq_len * head_num * size_per_head)
  // //  Buf: cublas_workspace_
  // cublasMM_cublasLtMM_wrapper(param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N,
  //                             CUBLAS_OP_N, n, m, k, &alpha,
  //                             param_.self_attention.attention_output_weight.kernel, AType_, n,
  //                             attr_out_buf_, BType_, k, &beta, (DataType_*)attr_matmul_buf_,
  //                             CType_, n, param_.stream, cublasAlgoMap_, sm_, cublas_workspace_);

  Free(attr_out_buf_);

  // // Used tensors:
  // //  attr_matmul_buf_ (batch_size * seq_len * head_num * size_per_head)
  // //  param_.from_tensor
  // //  param_.self_attention.attention_output_weight.bias
  // //  param_.self_layernorm.gamma
  // //  param_.self_layernorm.beta
  // add_bias_input_layernorm_kernelLauncher<DataType_>(
  //     attr_matmul_buf_, param_.from_tensor, param_.self_attention.attention_output_weight.bias,
  //     param_.self_layernorm.gamma, param_.self_layernorm.beta, m, n, param_.stream);

  // n *= 4;

  Alloc(&inter_matmul_buf_, 4 * batch_size * seq_len * head_num * size_per_head);

  // // Used tensors:
  // //  In: param_.ffn.intermediate_weight.kernel
  // //  In: attr_matmul_buf_ (batch_size * seq_len * head_num * size_per_head)
  // //  Out: inter_matmul_buf_ (4 * batch_size * seq_len * head_num * size_per_head)
  // //  Buf: cublas_workspace_
  // cublasMM_cublasLtMM_wrapper(param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N,
  //                             CUBLAS_OP_N, n, m, k, &alpha,
  //                             param_.ffn.intermediate_weight.kernel, AType_, n, attr_matmul_buf_,
  //                             BType_, k, &beta, (DataType_*)inter_matmul_buf_, CType_, n,
  //                             param_.stream, cublasAlgoMap_, sm_, cublas_workspace_);

  // // Used tensors:
  // //  inter_matmul_buf_ (4 * batch_size * seq_len * head_num * size_per_head)
  // //  In: param_.ffn.intermediate_weight.bias
  // add_bias_act_kernelLauncher<DataType_>(inter_matmul_buf_, param_.ffn.intermediate_weight.bias,
  // m,
  //                                        n, ActivationType::GELU, param_.stream);

  // n = k;
  // k *= 4;

  Alloc(&d_transformer_out, batch_size * seq_len * hidden_dim);

  // // Used tensors:
  // //  In: param_.ffn.output_weight.kernel
  // //  In: inter_matmul_buf_ (4 * batch_size * seq_len * head_num * size_per_head)
  // //  Out: param_.transformer_out
  // //  Buf: cublas_workspace_
  // cublasMM_cublasLtMM_wrapper(param_.cublaslt_handle, param_.cublas_handle, CUBLAS_OP_N,
  //                             CUBLAS_OP_N, n, m, k, &alpha, param_.ffn.output_weight.kernel,
  //                             AType_, n, inter_matmul_buf_, BType_, k, &beta,
  //                             (DataType_*)(param_.transformer_out), CType_, n, param_.stream,
  //                             cublasAlgoMap_, sm_, cublas_workspace_);

  Free(inter_matmul_buf_);

  // // Used tensors:
  // //  param_.transformer_out
  // //  attr_matmul_buf_ (batch_size * seq_len * head_num * size_per_head)
  // //  In: param_.ffn.output_weight.bias
  // //  In: param_.ffn_layernorm.gamma
  // //  In: param_.ffn_layernorm.beta
  // add_bias_input_layernorm_kernelLauncher<DataType_>(
  //     param_.transformer_out, attr_matmul_buf_, param_.ffn.output_weight.bias,
  //     param_.ffn_layernorm.gamma, param_.ffn_layernorm.beta, m, n, param_.stream);

  Free(attr_matmul_buf_);

  {
    Free(d_from_tensor);

    Free(d_transformer_out);

    Free(d_attr_kernel_Q);

    Free(d_attr_bias_Q);
    Free(d_attr_bias_K);
    Free(d_attr_bias_V);

    Free(d_attr_output_kernel);
    Free(d_attr_output_bias);
    Free(d_attr_output_layernorm_beta);
    Free(d_attr_output_layernorm_gamma);

    Free(d_inter_kernel);
    Free(d_inter_bias);

    Free(d_output_kernel);
    Free(d_output_bias);

    Free(d_output_layernorm_beta);
    Free(d_output_layernorm_gamma);
  }
}

int main(int argc, char** argv) {
  int NUM_HEADS = 8;
  int HEAD_SIZE = 64;
  int MODEL_DIM = 512;

  int batch_size = std::stoi(argv[1]);
  std::string data_file = argv[2];

  std::fstream fs;
  fs.open(data_file);
  if (!fs.is_open()) {
    std::cout << "Error opening input" << std::endl;
    exit(-1);
  }

  int max_len = -1;
  for (int i = 0; i < batch_size; ++i) {
    int length;
    fs >> length;
    if (length > max_len) {
      max_len = length;
    }
  }

  BERTforward(batch_size, max_len, MODEL_DIM, NUM_HEADS, HEAD_SIZE);
  float memory = (max_memory * 4.0) / (1024.0 * 1024.0);
  std::cout << "MEM," << memory << std::endl;
}
