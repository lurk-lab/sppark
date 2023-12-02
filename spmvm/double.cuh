// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_SPMVM_DOUBLE_CUH__
#define __SPPARK_SPMVM_DOUBLE_CUH__

#include <cuda.h>
#include <cassert>

#ifndef WARP_SZ
#define WARP_SZ 32
#endif
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

template <typename scalar_t>
struct double_host_t
{
    const scalar_t *data;
    const size_t *col_idx;
    const size_t *row_ptr;

    size_t num_rows;
    size_t num_cols;
    size_t nnz;
};

// Kernel function to double the elements of array
template <typename scalar_t>
__global__ void device_double_scalars(scalar_t *d_scalars, size_t num_cols, scalar_t *out)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < num_cols)
    {
        out[idx] = d_scalars[idx] * d_scalars[idx] + d_scalars[idx];
        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
}

#undef asm

#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
template __global__ void device_double_scalars(scalar_t *d_scalars, size_t num_cols, scalar_t *out);
#endif

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

template <typename scalar_t>
class double_scalar_t
{
public:
    const gpu_t &gpu;

    // sparse matrix data fields
    scalar_t *d_data;
    size_t *d_col_idx;
    size_t *d_row_ptr;

    size_t num_rows;
    size_t num_cols;
    size_t nnz;

    scalar_t *d_scalars;
    scalar_t *d_out;

public:
    double_scalar_t(double_host_t<scalar_t> *csr, int device_id = -1)
        : gpu(select_gpu(device_id)), d_data(nullptr), d_col_idx(nullptr), d_row_ptr(nullptr), d_scalars(nullptr), d_out(nullptr)
    {
        this->num_rows = csr->num_rows;
        this->num_cols = csr->num_cols;
        this->nnz = csr->nnz;

        d_data = reinterpret_cast<decltype(d_data)>(gpu.Dmalloc(nnz * sizeof(scalar_t)));
        d_col_idx = reinterpret_cast<decltype(d_col_idx)>(gpu.Dmalloc(nnz * sizeof(size_t)));
        d_row_ptr = reinterpret_cast<decltype(d_row_ptr)>(gpu.Dmalloc((num_rows + 1) * sizeof(size_t)));
        d_scalars = reinterpret_cast<decltype(d_scalars)>(gpu.Dmalloc(num_cols * sizeof(scalar_t)));
        d_out = reinterpret_cast<decltype(d_out)>(gpu.Dmalloc(num_rows * sizeof(scalar_t)));
    }
    ~double_scalar_t()
    {
        gpu.sync();
        if (d_data)
            gpu.Dfree(d_data);
        if (d_col_idx)
            gpu.Dfree(d_col_idx);
        if (d_row_ptr)
            gpu.Dfree(d_row_ptr);
        if (d_scalars)
            gpu.Dfree(d_scalars);
        if (d_out)
            gpu.Dfree(d_out);
    }

public:
    RustError invoke(double_host_t<scalar_t> *csr, scalar_t scalars[], scalar_t out[])
    {
        assert(csr->num_rows == this->num_rows);
        assert(csr->num_cols == this->num_cols);
        assert(csr->nnz == this->nnz);

        try
        {
            if (scalars)
                gpu[2].HtoD(&d_scalars[0], &scalars[0], csr->num_cols);

            device_double_scalars<scalar_t><<<1, 32, 0, gpu[2]>>>(&d_scalars[0], csr->num_cols, &d_out[0]);
            CUDA_OK(cudaGetLastError());

            gpu[2].DtoH(&out[0], &d_out[0], csr->num_rows);
            gpu.sync();

            return RustError{cudaSuccess};
        }
        catch (const cuda_error &e)
        {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }
    }
};

template <typename scalar_t>
static RustError double_scalars(double_host_t<scalar_t> *csr, scalar_t *scalars, scalar_t *out)
{
    try
    {
        double_scalar_t<scalar_t> dbl{csr};
        return dbl.invoke(csr, scalars, out);
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

#endif
