#pragma once
#include <vector>
#include <memory>

using scalar_t = float;
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

//pulled from NNFRAME but no autograd 
class Tensor : public std::enable_shared_from_this<Tensor> {
    private:
        std::shared_ptr<std::vector<scalar_t>> data_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_;

        size_t compute_strides();

        Tensor(std::shared_ptr<std::vector<scalar_t>> data, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset);

    public:
        Tensor(std::vector<size_t> shape, scalar_t fill_value = 0.0f);
        Tensor(std::vector<size_t> shape, std::vector<scalar_t> data);

        const std::vector<size_t>& shape() const;
        const std::vector<size_t>& strides() const;
        size_t rank() const;
        size_t numel() const;
        bool is_contiguous() const;

        // Access
        scalar_t& at(std::initializer_list<size_t> indices);
        scalar_t at(std::initializer_list<size_t> indices) const;
        scalar_t& at(const std::vector<size_t>& indices);
        scalar_t at(const std::vector<size_t>& indices) const;

        const std::vector<scalar_t>& data() const;
        std::vector<scalar_t>& mutable_data();

        // Shape ops — views into the same buffer unless noted
        TensorPtr reshape(std::vector<size_t> new_shape) const;
        TensorPtr transpose() const;
        TensorPtr permute(std::vector<size_t> axes) const;
        TensorPtr contiguous() const; // materializes a fresh contiguous copy

        // Elementwise math (broadcasting, new tensor returned)
        TensorPtr add(const TensorPtr& other) const;
        TensorPtr sub(const TensorPtr& other) const;
        TensorPtr mul(const TensorPtr& other) const;

        TensorPtr operator+(const TensorPtr& other) const;
        TensorPtr operator-(const TensorPtr& other) const;
        TensorPtr operator*(const TensorPtr& other) const;

        TensorPtr matmul(const TensorPtr& other) const; // rank 2 or rank 3 (batched, e.g. per-head)
        TensorPtr softmax(size_t dim) const; // last-dim only

        bool allclose(const Tensor& other, scalar_t eps = 1e-5f) const;

        // Factory methods
        static TensorPtr create(std::vector<size_t> shape, scalar_t fill = 0.0f);
        static TensorPtr zeros(std::vector<size_t> shape);
        static TensorPtr from_vector(std::vector<scalar_t> data); // rank-1, shape inferred

        // Raw-buffer matmul kernels (row-major contiguous)
        static void matmul_avx2(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N);
        static void matmul_scalar(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N);
        static void transpose_buffer(const scalar_t* in, scalar_t* out, size_t rows, size_t cols);
};
