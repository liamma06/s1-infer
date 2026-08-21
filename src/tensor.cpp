#include "core/tensor.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <immintrin.h> // AVX2

namespace {
    // odometer-style index increment over a shape
    bool increment_index(std::vector<size_t>& idx, const std::vector<size_t>& shape) {
        for (int i = static_cast<int>(idx.size()) - 1; i >= 0; i--) {
            if (++idx[i] < shape[i]) return true;
            idx[i] = 0;
        }
        return false;
    }
}

Tensor::Tensor(std::vector<size_t> shape, scalar_t fill) {
    shape_ = shape;
    offset_ = 0;
    size_t total = compute_strides();
    data_ = std::make_shared<std::vector<scalar_t>>(total, fill);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<scalar_t> data) {
    shape_ = shape;
    offset_ = 0;
    size_t total = compute_strides();
    assert(data.size() == total && "Data size does not match shape size");
    data_ = std::make_shared<std::vector<scalar_t>>(std::move(data));
}

Tensor::Tensor(std::shared_ptr<std::vector<scalar_t>> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) {
    data_ = data;
    shape_ = shape;
    strides_ = strides;
    offset_ = offset;
}

TensorPtr Tensor::create(std::vector<size_t> shape, scalar_t fill) {
    return std::make_shared<Tensor>(shape, fill);
}

TensorPtr Tensor::zeros(std::vector<size_t> shape) {
    return Tensor::create(shape);
}

TensorPtr Tensor::from_vector(std::vector<scalar_t> data) {
    return std::make_shared<Tensor>(std::vector<size_t>{data.size()}, data);
}

size_t Tensor::compute_strides() {
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
    return stride;
}

const std::vector<size_t>& Tensor::shape() const { return shape_; }
const std::vector<size_t>& Tensor::strides() const { return strides_; }
size_t Tensor::rank() const { return shape_.size(); }

size_t Tensor::numel() const {
    size_t n = 1;
    for (size_t s : shape_) n *= s;
    return n;
}

bool Tensor::is_contiguous() const {
    size_t expected = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

scalar_t& Tensor::at(std::initializer_list<size_t> indices) {
    assert(indices.size() == shape_.size() && "Number of indices must match tensor rank");
    size_t pos = offset_;
    size_t i = 0;
    for (auto index : indices) {
        assert(index < shape_[i] && "Index out of bounds");
        pos += index * strides_[i];
        i++;
    }
    return (*data_)[pos];
}

scalar_t Tensor::at(std::initializer_list<size_t> indices) const {
    assert(indices.size() == shape_.size() && "Number of indices must match tensor rank");
    size_t pos = offset_;
    size_t i = 0;
    for (auto index : indices) {
        assert(index < shape_[i] && "Index out of bounds");
        pos += index * strides_[i];
        i++;
    }
    return (*data_)[pos];
}

scalar_t& Tensor::at(const std::vector<size_t>& indices) {
    assert(indices.size() == shape_.size() && "Number of indices must match tensor rank");
    size_t pos = offset_;
    for (size_t i = 0; i < indices.size(); i++) {
        assert(indices[i] < shape_[i] && "Index out of bounds");
        pos += indices[i] * strides_[i];
    }
    return (*data_)[pos];
}

scalar_t Tensor::at(const std::vector<size_t>& indices) const {
    assert(indices.size() == shape_.size() && "Number of indices must match tensor rank");
    size_t pos = offset_;
    for (size_t i = 0; i < indices.size(); i++) {
        assert(indices[i] < shape_[i] && "Index out of bounds");
        pos += indices[i] * strides_[i];
    }
    return (*data_)[pos];
}

const std::vector<scalar_t>& Tensor::data() const { return *data_; }
std::vector<scalar_t>& Tensor::mutable_data() { return *data_; }

TensorPtr Tensor::reshape(std::vector<size_t> new_shape) const {
    assert(is_contiguous() && "Tensor must be contiguous to reshape");

    size_t new_numel = 1;
    for (size_t dim : new_shape) new_numel *= dim;
    assert(new_numel == numel() && "New shape must have the same number of elements as the original tensor");

    std::vector<size_t> new_strides(new_shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; i--) {
        new_strides[i] = stride;
        stride *= new_shape[i];
    }

    return TensorPtr(new Tensor(data_, new_shape, new_strides, offset_));
}

TensorPtr Tensor::permute(std::vector<size_t> axes) const {
    assert(axes.size() == shape_.size() && "Axes length must match tensor rank");
    for (size_t a : axes) assert(a < shape_.size() && "Axis index out of bounds");

    std::vector<bool> seen(shape_.size(), false);
    for (size_t a : axes) {
        assert(!seen[a] && "Axes must be unique");
        seen[a] = true;
    }

    std::vector<size_t> new_shape(shape_.size());
    std::vector<size_t> new_strides(strides_.size());
    for (size_t i = 0; i < axes.size(); i++) {
        new_shape[i] = shape_[axes[i]];
        new_strides[i] = strides_[axes[i]];
    }

    return TensorPtr(new Tensor(data_, new_shape, new_strides, offset_));
}

TensorPtr Tensor::contiguous() const {
    auto result = Tensor::create(shape_);
    if (numel() > 0) {
        std::vector<size_t> idx(rank(), 0);
        while (true) {
            result->at(idx) = at(idx);
            if (!increment_index(idx, shape_)) break;
        }
    }
    return result;
}

TensorPtr Tensor::transpose() const {
    assert(rank() == 2 && "transpose is for 2D tensors, use permute for higher ranks");
    return permute({1, 0});
}

TensorPtr Tensor::add(const TensorPtr& other) const {
    size_t out_rank = std::max(rank(), other->rank());
    std::vector<size_t> new_shape(out_rank);
    for (size_t i = 0; i < out_rank; i++) {
        size_t a = (i < rank()) ? shape_[i] : 1;
        size_t b = (i >= (out_rank - other->rank())) ? other->shape()[i - (out_rank - other->rank())] : 1;
        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for addition");
        new_shape[i] = std::max(a, b);
    }

    auto out = Tensor::create(new_shape);
    std::vector<size_t> cur_idx(out_rank, 0);
    for (size_t i = 0; i < out->numel(); i++) {
        std::vector<size_t> a_idx(rank());
        std::vector<size_t> b_idx(other->rank());
        for (size_t j = 0; j < rank(); j++)
            a_idx[j] = (shape_[j] == 1) ? 0 : cur_idx[out_rank - rank() + j];
        for (size_t j = 0; j < other->rank(); j++)
            b_idx[j] = (other->shape()[j] == 1) ? 0 : cur_idx[out_rank - other->rank() + j];
        out->at(cur_idx) = at(a_idx) + other->at(b_idx);
        increment_index(cur_idx, new_shape);
    }
    return out;
}

TensorPtr Tensor::sub(const TensorPtr& other) const {
    size_t out_rank = std::max(rank(), other->rank());
    std::vector<size_t> new_shape(out_rank);
    for (size_t i = 0; i < out_rank; i++) {
        size_t a = (i < rank()) ? shape_[i] : 1;
        size_t b = (i >= (out_rank - other->rank())) ? other->shape()[i - (out_rank - other->rank())] : 1;
        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for subtraction");
        new_shape[i] = std::max(a, b);
    }

    auto out = Tensor::create(new_shape);
    std::vector<size_t> cur_idx(out_rank, 0);
    for (size_t i = 0; i < out->numel(); i++) {
        std::vector<size_t> a_idx(rank());
        std::vector<size_t> b_idx(other->rank());
        for (size_t j = 0; j < rank(); j++)
            a_idx[j] = (shape_[j] == 1) ? 0 : cur_idx[out_rank - rank() + j];
        for (size_t j = 0; j < other->rank(); j++)
            b_idx[j] = (other->shape()[j] == 1) ? 0 : cur_idx[out_rank - other->rank() + j];
        out->at(cur_idx) = at(a_idx) - other->at(b_idx);
        increment_index(cur_idx, new_shape);
    }
    return out;
}

TensorPtr Tensor::mul(const TensorPtr& other) const {
    size_t out_rank = std::max(rank(), other->rank());
    std::vector<size_t> new_shape(out_rank);
    for (size_t i = 0; i < out_rank; i++) {
        size_t a = (i < rank()) ? shape_[i] : 1;
        size_t b = (i >= (out_rank - other->rank())) ? other->shape()[i - (out_rank - other->rank())] : 1;
        assert((a == b || a == 1 || b == 1) && "Tensors are not compatible for multiplication");
        new_shape[i] = std::max(a, b);
    }

    auto out = Tensor::create(new_shape);
    std::vector<size_t> cur_idx(out_rank, 0);
    for (size_t i = 0; i < out->numel(); i++) {
        std::vector<size_t> a_idx(rank());
        std::vector<size_t> b_idx(other->rank());
        for (size_t j = 0; j < rank(); j++)
            a_idx[j] = (shape_[j] == 1) ? 0 : cur_idx[out_rank - rank() + j];
        for (size_t j = 0; j < other->rank(); j++)
            b_idx[j] = (other->shape()[j] == 1) ? 0 : cur_idx[out_rank - other->rank() + j];
        out->at(cur_idx) = at(a_idx) * other->at(b_idx);
        increment_index(cur_idx, new_shape);
    }
    return out;
}

TensorPtr Tensor::operator+(const TensorPtr& other) const { return add(other); }
TensorPtr Tensor::operator-(const TensorPtr& other) const { return sub(other); }
TensorPtr Tensor::operator*(const TensorPtr& other) const { return mul(other); }

bool Tensor::allclose(const Tensor& other, scalar_t eps) const {
    assert(shape_ == other.shape_ && "Shapes must match for allclose");
    if (numel() == 0) return true;
    std::vector<size_t> idx(rank(), 0);
    while (true) {
        if (std::abs(at(idx) - other.at(idx)) > eps) return false;
        if (!increment_index(idx, shape_)) break;
    }
    return true;
}

void Tensor::transpose_buffer(const scalar_t* in, scalar_t* out, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            out[j * rows + i] = in[i * cols + j];
}

void Tensor::matmul_scalar(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            scalar_t sum = 0.0f;
            for (size_t k = 0; k < K; k++)
                sum += a[i * K + k] * b[k * N + j];
            out[i * N + j] = sum;
        }
    }
}

void Tensor::matmul_avx2(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N) {
    __m256 b_vec, prod_vec, a_broadcast, sum_vec;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j + 8 <= N; j += 8) {
            sum_vec = _mm256_setzero_ps();
            for (size_t k = 0; k < K; k++) {
                a_broadcast = _mm256_set1_ps(a[i * K + k]);
                b_vec = _mm256_loadu_ps(&b[k * N + j]);
                prod_vec = _mm256_mul_ps(a_broadcast, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod_vec);
            }
            _mm256_storeu_ps(&out[i * N + j], sum_vec);
        }
        for (size_t j = N - (N % 8); j < N; j++) {
            scalar_t sum = 0.0f;
            for (size_t k = 0; k < K; k++)
                sum += a[i * K + k] * b[k * N + j];
            out[i * N + j] = sum;
        }
    }
}

TensorPtr Tensor::matmul(const TensorPtr& other) const {
    assert(rank() == other->rank() && (rank() == 2 || rank() == 3) && "Both tensors must be rank 2 or rank 3, and match each other");
    assert(shape_[rank() - 1] == other->shape()[rank() - 2] && "Inner dimensions must match for matrix multiplication");

    TensorPtr a_contig = is_contiguous() ? nullptr : contiguous();
    TensorPtr b_contig = other->is_contiguous() ? nullptr : other->contiguous();
    const scalar_t* a_base = a_contig ? a_contig->data().data() : data_->data() + offset_;
    const scalar_t* b_base = b_contig ? b_contig->data().data() : other->data_->data() + other->offset_;

    if (rank() == 2) {
        size_t M = shape_[0], K = shape_[1], N = other->shape()[1];
        auto out = Tensor::create({M, N});
        Tensor::matmul_avx2(a_base, b_base, out->mutable_data().data(), M, K, N);
        return out;
    } else {
        assert(shape_[0] == other->shape()[0] && "Leading (head) dimension must match for batched matmul");
        size_t L = shape_[0], M = shape_[1], K = shape_[2], N = other->shape()[2];
        auto out = Tensor::create({L, M, N});
        for (size_t l = 0; l < L; l++) {
            Tensor::matmul_avx2(
                a_base + l * M * K,
                b_base + l * K * N,
                out->mutable_data().data() + l * M * N,
                M, K, N
            );
        }
        return out;
    }
}

TensorPtr Tensor::softmax(size_t dim) const {
    assert(dim < rank() && "Dimension out of bounds for softmax");
    assert((rank() == 2 || rank() == 3) && "softmax only supports rank 2 or rank 3");
    assert(dim == rank() - 1 && "softmax currently only supports normalizing the last axis");

    auto out = Tensor::create(shape_);

    if (rank() == 2) {
        for (size_t i = 0; i < shape_[0]; i++) {
            scalar_t row_max = at({i, 0});
            for (size_t j = 1; j < shape_[1]; j++) row_max = std::max(row_max, at({i, j}));

            scalar_t row_sum = 0.0f;
            for (size_t j = 0; j < shape_[1]; j++) {
                scalar_t e = std::exp(at({i, j}) - row_max);
                out->at({i, j}) = e;
                row_sum += e;
            }
            for (size_t j = 0; j < shape_[1]; j++) out->at({i, j}) /= row_sum;
        }
    } else {
        for (size_t i = 0; i < shape_[0]; i++) {
            for (size_t j = 0; j < shape_[1]; j++) {
                scalar_t row_max = at({i, j, 0});
                for (size_t k = 1; k < shape_[2]; k++) row_max = std::max(row_max, at({i, j, k}));

                scalar_t row_sum = 0.0f;
                for (size_t k = 0; k < shape_[2]; k++) {
                    scalar_t e = std::exp(at({i, j, k}) - row_max);
                    out->at({i, j, k}) = e;
                    row_sum += e;
                }
                for (size_t k = 0; k < shape_[2]; k++) out->at({i, j, k}) /= row_sum;
            }
        }
    }
    return out;
}
