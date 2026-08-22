#include "nn/rmsnorm.h"
#include <cmath> 

TensorPtr rmsnorm(const TensorPtr& input, const TensorPtr& weight, scalar_t eps) {
    /*
        input = [seq_len, embed_dim]
        weight = [embed_dim]
    */

    TensorPtr output = Tensor::create(input->shape());   

    size_t last_dim = input->shape().back(); //embed dim 
    size_t num_elements = input->numel();

    for (size_t i = 0; i < num_elements; i += last_dim){
        scalar_t sum_squares = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            scalar_t val = input->data()[i + j]; //offset token and embed dim 
            sum_squares += val * val;
        }

        scalar_t rms = std::sqrt(sum_squares / last_dim + eps);

        for (size_t j = 0; j < last_dim; ++j) {
            scalar_t normalized = input->data()[i + j] / rms;
            output->mutable_data()[i + j] = normalized * weight->data()[j]; 
        }
    }

    return output;
}