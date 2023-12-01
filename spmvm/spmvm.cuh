// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_SPMVM_SPMVM_CUH__
#define __SPPARK_SPMVM_SPMVM_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <cassert>

#include <util/rusterror.h>

// #ifndef WARP_SZ
// #define WARP_SZ 32
// #endif
// #ifdef __GNUC__
// #define asm __asm__ __volatile__
// #else
// #define asm asm volatile
// #endif

template <typename scalar_t>
class spmvm_t;

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < n)
    {
        y[idx] = x[idx] + y[idx];
        idx += gridDim.x * blockDim.x;
    }
}

int add_test(void)
{
    int N = 1 << 20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<512, 32>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    printf("Max error: %f", maxError);

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}

template <typename scalar_t>
__global__ void accumulate(scalar_t *d_out, spmvm_t<scalar_t> *csr, const scalar_t *d_scalars)
{

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < csr->row_size)
    {
        for (size_t i = csr->d_row_ptr[idx]; i < csr->d_row_ptr[idx + 1]; i++)
        {
            d_out[idx] = d_out[idx] + d_scalars[csr->d_col_idx[i]] * csr->d_data[i];
        }
        idx += gridDim.x * blockDim.x;
    }
}

// #ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
// template __global__ void accumulate(scalar_t *d_out, spmvm_t<scalar_t> *csr, const scalar_t *d_scalars);
// #endif

template <typename scalar_t>
struct csr_t_host
{
    const scalar_t *data;
    const size_t *col_idx;
    const size_t *row_ptr;

    size_t row_size;
    size_t col_size;
    size_t nnz;
};

template <typename scalar_t>
class spmvm_t
{
public:
    scalar_t *d_data;
    size_t *d_col_idx;
    size_t *d_row_ptr;

    size_t row_size;
    size_t col_size;
    size_t nnz;

public:
    spmvm_t(const csr_t_host<scalar_t> *csr)
        : d_data(nullptr), d_col_idx(nullptr), d_row_ptr(nullptr)
    {
        nnz = csr->nnz;
        col_size = csr->col_size;
        row_size = csr->row_size;

        cudaMalloc(&d_data, nnz * sizeof(scalar_t));
        cudaMalloc(&d_col_idx, nnz * sizeof(size_t));
        cudaMalloc(&d_row_ptr, (row_size + 1) * sizeof(size_t));

        cudaMemcpy(d_data, csr->data, nnz * sizeof(scalar_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx, csr->col_idx, nnz * sizeof(size_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_ptr, csr->row_ptr, (row_size + 1) * sizeof(size_t),
                   cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
    }

    ~spmvm_t()
    {
        if (d_data)
            cudaFree(d_data);
        if (d_col_idx)
            cudaFree(d_col_idx);
        if (d_row_ptr)
            cudaFree(d_row_ptr);

        cudaDeviceSynchronize();
    }

    RustError invoke(scalar_t out[], const scalar_t scalars[])
    {
        scalar_t *d_out;
        scalar_t *d_scalars;
        CUDA_OK(cudaMalloc(&d_out, row_size * sizeof(scalar_t)));
        CUDA_OK(cudaMalloc(&d_scalars, col_size * sizeof(scalar_t)));

        CUDA_OK(cudaMemcpy(d_out, out, row_size * sizeof(scalar_t),
                           cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_scalars, scalars, col_size * sizeof(scalar_t),
                           cudaMemcpyHostToDevice));

        accumulate<scalar_t><<<512, 32>>>(d_out, this, d_scalars);
        CUDA_OK(cudaGetLastError());

        cudaMemcpy(out, d_out, row_size * sizeof(scalar_t),
                   cudaMemcpyDeviceToHost);

        if (d_out)
            cudaFree(d_out);
        if (d_scalars)
            cudaFree(d_scalars);

        cudaDeviceSynchronize();
        return RustError{cudaSuccess};
    }
};

template <typename scalar_t>
static RustError spmvm(scalar_t out[], const csr_t_host<scalar_t> *csr, const scalar_t scalars[])
{
    try
    {
        spmvm_t<scalar_t> spmvm{csr};
        return spmvm.invoke(out, scalars);
    }
    catch (const cuda_error &e)
    {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

template <typename scalar_t>
static RustError spmvm_cpu(scalar_t out[], const csr_t_host<scalar_t> *csr, const scalar_t scalars[])
{
    for (size_t idx = 0; idx < csr->row_size; ++idx)
    {
        for (size_t i = csr->row_ptr[idx]; i < csr->row_ptr[idx + 1]; ++i)
        {
            out[idx] += scalars[csr->col_idx[i]] * csr->data[i];
        }
    }
    return RustError{cudaSuccess};
}

#endif /* __SPPARK_SPMVM_SPMVM_CUH__ */
