// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_SPMVM_SPMVM_CUH__
#define __SPPARK_SPMVM_SPMVM_CUH__

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
class spmvm_t;

template <typename scalar_t>
struct spmvm_context_t
{
    scalar_t *d_data;
    size_t *d_col_idx;
    size_t *d_row_ptr;

    size_t num_rows;
    size_t num_cols;
    size_t nnz;

    // input scalars
    scalar_t *d_scalars;
    // output scalars
    scalar_t *d_out;
};

template <typename scalar_t>
void drop_spmvm_context(spmvm_context_t<scalar_t> &ref)
{
    CUDA_OK(cudaFree(ref.d_data));
    CUDA_OK(cudaFree(ref.d_col_idx));
    CUDA_OK(cudaFree(ref.d_row_ptr));
    CUDA_OK(cudaFree(ref.d_scalars));
    CUDA_OK(cudaFree(ref.d_out));
}

struct buckets_t
{
    const size_t *data;
    size_t len;
    size_t width;
};

// Kernel function to double the elements of array
template <typename scalar_t>
__global__ void csr_vector_mul(spmvm_context_t<scalar_t> *d_context)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < d_context->num_rows)
    {
        for (size_t i = d_context->d_row_ptr[idx]; i < d_context->d_row_ptr[idx + 1]; i++)
        {
            d_context->d_out[idx] = d_context->d_out[idx] + d_context->d_scalars[d_context->d_col_idx[i]] * d_context->d_data[i];
        }
        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
}

// Kernel function to double the elements of array
template <typename scalar_t>
__global__ void csr_vector_mul_buckets(spmvm_context_t<scalar_t> *d_context)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < d_context->num_rows)
    {
        for (size_t i = d_context->d_row_ptr[idx]; i < d_context->d_row_ptr[idx + 1]; i++)
        {
            d_context->d_out[idx] = d_context->d_out[idx] + d_context->d_scalars[d_context->d_col_idx[i]] * d_context->d_data[i];
        }
        idx += gridDim.x * blockDim.x;
    }
    __syncthreads();
}

#undef asm

#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
template __global__ void csr_vector_mul(spmvm_context_t<scalar_t> *d_context);
#endif

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

template <typename scalar_t>
struct spmvm_host_t
{
    const scalar_t *data;
    const size_t *col_idx;
    const size_t *row_ptr;

    size_t num_rows;
    size_t num_cols;
    size_t nnz;
};

template <typename scalar_t>
struct witness_t
{
    const scalar_t *W;
    const scalar_t *u;
    const scalar_t *U;

    size_t nW;
    size_t nU;
};

template <typename scalar_t>
class spmvm_t
{
public:
    const gpu_t &gpu;
    bool owned;

    spmvm_context_t<scalar_t> *context;
    spmvm_context_t<scalar_t> *d_context;

public:
    spmvm_t(size_t num_rows, size_t num_cols, size_t nnz, int device_id = -1)
        : gpu(select_gpu(device_id))
    {
        this->owned = true;

        // setup on host side
        this->context = reinterpret_cast<spmvm_context_t<scalar_t> *>(malloc(sizeof(spmvm_context_t<scalar_t>)));
        context->d_data = reinterpret_cast<scalar_t *>(gpu.Dmalloc(nnz * sizeof(scalar_t)));
        context->d_col_idx = reinterpret_cast<size_t *>(gpu.Dmalloc(nnz * sizeof(size_t)));
        context->d_row_ptr = reinterpret_cast<size_t *>(gpu.Dmalloc((num_rows + 1) * sizeof(size_t)));
        context->num_rows = num_rows;
        context->num_cols = num_cols;
        context->nnz = nnz;
        context->d_scalars = reinterpret_cast<scalar_t *>(gpu.Dmalloc(num_cols * sizeof(scalar_t)));
        context->d_out = reinterpret_cast<scalar_t *>(gpu.Dmalloc(num_rows * sizeof(scalar_t)));

        // copy over to device
        this->d_context = reinterpret_cast<spmvm_context_t<scalar_t> *>(gpu.Dmalloc(sizeof(spmvm_context_t<scalar_t>)));
        gpu[2].HtoD(&this->d_context[0], &context[0], 1);
    }
    spmvm_t(spmvm_host_t<scalar_t> *csr, spmvm_context_t<scalar_t> *spmvm_context, int device_id = -1) : gpu(select_gpu(device_id))
    {
        // only do setup if csr is not nullptr
        if (csr)
        {
            // setup on host side
            this->context = reinterpret_cast<spmvm_context_t<scalar_t> *>(malloc(sizeof(spmvm_context_t<scalar_t>)));
            context->d_data = reinterpret_cast<scalar_t *>(gpu.Dmalloc(csr->nnz * sizeof(scalar_t)));
            context->d_col_idx = reinterpret_cast<size_t *>(gpu.Dmalloc(csr->nnz * sizeof(size_t)));
            context->d_row_ptr = reinterpret_cast<size_t *>(gpu.Dmalloc((csr->num_rows + 1) * sizeof(size_t)));
            context->num_rows = csr->num_rows;
            context->num_cols = csr->num_cols;
            context->nnz = csr->nnz;
            context->d_scalars = reinterpret_cast<scalar_t *>(gpu.Dmalloc(csr->num_cols * sizeof(scalar_t)));
            context->d_out = reinterpret_cast<scalar_t *>(gpu.Dmalloc(csr->num_rows * sizeof(scalar_t)));

            // copy over to device, while passing the context out
            spmvm_context = reinterpret_cast<spmvm_context_t<scalar_t> *>(gpu.Dmalloc(sizeof(spmvm_context_t<scalar_t>)));
            gpu[2].HtoD(&spmvm_context[0], &context[0], 1);

            // move data into allocated memory
            if (csr->data)
                gpu[2].HtoD(&context->d_data[0], &csr->data[0], csr->nnz);
            if (csr->col_idx)
                gpu[2].HtoD(&context->d_col_idx[0], &csr->col_idx[0], csr->nnz);
            if (csr->row_ptr)
                gpu[2].HtoD(&context->d_row_ptr[0], &csr->row_ptr[0], csr->num_rows + 1);
        } else {
            this->context = reinterpret_cast<spmvm_context_t<scalar_t> *>(malloc(sizeof(spmvm_context_t<scalar_t>)));
            gpu[2].DtoH(&context[0], &spmvm_context[0], 1);
        }

        this->owned = false;
        this->d_context = spmvm_context;
    }
    ~spmvm_t()
    {
        gpu.sync();
        if (context->d_data && owned)
            gpu.Dfree(context->d_data);
        if (context->d_col_idx && owned)
            gpu.Dfree(context->d_col_idx);
        if (context->d_row_ptr && owned)
            gpu.Dfree(context->d_row_ptr);
        if (context->d_scalars && owned)
            gpu.Dfree(context->d_scalars);
        if (context->d_out && owned)
            gpu.Dfree(context->d_out);

        if (context)
            free(context);
        if (d_context)
            gpu.Dfree(d_context);
    }

public:
    RustError invoke(spmvm_host_t<scalar_t> *csr, const scalar_t scalars[], scalar_t out[], size_t nthreads)
    {
        assert(csr->num_rows == context->num_rows);
        assert(csr->num_cols == context->num_cols);
        assert(csr->nnz == context->nnz);

        try
        {
            if (csr->data)
                gpu[2].HtoD(&context->d_data[0], &csr->data[0], context->nnz);
            if (csr->col_idx)
                gpu[2].HtoD(&context->d_col_idx[0], &csr->col_idx[0], context->nnz);
            if (csr->row_ptr)
                gpu[2].HtoD(&context->d_row_ptr[0], &csr->row_ptr[0], context->num_rows + 1);

            if (scalars)
                gpu[2].HtoD(&context->d_scalars[0], &scalars[0], context->num_cols);

            cudaMemsetAsync(&context->d_out[0], 0, context->num_rows * sizeof(scalar_t), gpu[2]);
            csr_vector_mul<scalar_t><<<gpu.sm_count(), nthreads, 0, gpu[2]>>>(d_context);
            CUDA_OK(cudaGetLastError());

            gpu[2].DtoH(&out[0], &context->d_out[0], context->num_rows);
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

    RustError invoke_witness(spmvm_host_t<scalar_t> *csr, const witness_t<scalar_t> *witness, scalar_t out[], size_t nthreads)
    {
        assert(csr->num_rows == context->num_rows);
        assert(csr->num_cols == context->num_cols);
        assert(csr->nnz == context->nnz);

        assert(witness->nW + witness->nU + 1 == csr->num_cols);

        try
        {
            if (csr->data)
                gpu[2].HtoD(&context->d_data[0], &csr->data[0], context->nnz);
            if (csr->col_idx)
                gpu[2].HtoD(&context->d_col_idx[0], &csr->col_idx[0], context->nnz);
            if (csr->row_ptr)
                gpu[2].HtoD(&context->d_row_ptr[0], &csr->row_ptr[0], context->num_rows + 1);

            if (witness->W)
                gpu[2].HtoD(&context->d_scalars[0], &witness->W[0], witness->nW);
            gpu[2].HtoD(&context->d_scalars[witness->nW], witness->u, 1);
            if (witness->U)
                gpu[2].HtoD(&context->d_scalars[witness->nW + 1], &witness->U[0], witness->nU);

            cudaMemsetAsync(&context->d_out[0], 0, context->num_rows * sizeof(scalar_t), gpu[2]);
            csr_vector_mul<scalar_t><<<gpu.sm_count(), nthreads, 0, gpu[2]>>>(d_context);
            CUDA_OK(cudaGetLastError());

            gpu[2].DtoH(&out[0], &context->d_out[0], context->num_rows);
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

    RustError invoke_witness_with(const witness_t<scalar_t> *witness, scalar_t out[], size_t nblocks, size_t nthreads)
    {
        assert(witness->nW + witness->nU + 1 == context->num_cols);

        try
        {
            if (witness->W)
                gpu[2].HtoD(&context->d_scalars[0], &witness->W[0], witness->nW);
            gpu[2].HtoD(&context->d_scalars[witness->nW], witness->u, 1);
            if (witness->U)
                gpu[2].HtoD(&context->d_scalars[witness->nW + 1], &witness->U[0], witness->nU);

            cudaMemsetAsync(&context->d_out[0], 0, context->num_rows * sizeof(scalar_t), gpu[2]);
            csr_vector_mul<scalar_t><<<nblocks, nthreads, 0, gpu[2]>>>(d_context);
            CUDA_OK(cudaGetLastError());

            gpu[2].DtoH(&out[0], &context->d_out[0], context->num_rows);
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

    RustError invoke_witness_with_buckets(const witness_t<scalar_t> *witness, scalar_t out[],
                                          const buckets_t *buckets, size_t nblocks, size_t nthreads)
    {
        assert(witness->nW + witness->nU + 1 == context->num_cols);

        try
        {
            if (witness->W)
                gpu[2].HtoD(&context->d_scalars[0], &witness->W[0], witness->nW);
            gpu[2].HtoD(&context->d_scalars[witness->nW], witness->u, 1);
            if (witness->U)
                gpu[2].HtoD(&context->d_scalars[witness->nW + 1], &witness->U[0], witness->nU);

            cudaMemsetAsync(&context->d_out[0], 0, context->num_rows * sizeof(scalar_t), gpu[2]);
            csr_vector_mul<scalar_t><<<nblocks, nthreads, 0, gpu[2]>>>(d_context);
            CUDA_OK(cudaGetLastError());

            gpu[2].DtoH(&out[0], &context->d_out[0], context->num_rows);
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
static RustError sparse_matrix_mul(spmvm_host_t<scalar_t> *csr, const scalar_t *scalars, scalar_t *out, size_t nthreads)
{
    try
    {
        size_t num_rows = csr->num_rows;
        size_t num_cols = csr->num_cols;
        size_t nnz = csr->nnz;
        spmvm_t<scalar_t> spmvm{num_rows, num_cols, nnz};
        return spmvm.invoke(csr, scalars, out, nthreads);
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
static RustError sparse_matrix_witness_init(spmvm_host_t<scalar_t> *csr, spmvm_context_t<scalar_t> *spmvm_context)
{
    try
    {
        spmvm_t<scalar_t> spmvm{csr, spmvm_context};
        return RustError{cudaSuccess};
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
static RustError sparse_matrix_witness(
    spmvm_host_t<scalar_t> *csr, const witness_t<scalar_t> *witness, scalar_t *out, size_t nthreads)
{
    try
    {
        size_t num_rows = csr->num_rows;
        size_t num_cols = csr->num_cols;
        size_t nnz = csr->nnz;
        spmvm_t<scalar_t> spmvm{num_rows, num_cols, nnz};
        return spmvm.invoke_witness(csr, witness, out, nthreads);
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
static RustError sparse_matrix_witness_with(
    spmvm_context_t<scalar_t> *spmvm_context, const witness_t<scalar_t> *witness,
    scalar_t out[], size_t nblocks, size_t nthreads)
{
    try
    {
        spmvm_t<scalar_t> spmvm{nullptr, spmvm_context};
        return spmvm.invoke_witness_with(witness, out, nblocks, nthreads);
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
static RustError sparse_matrix_witness_cpu(
    spmvm_host_t<scalar_t> *csr, const witness_t<scalar_t> *witness, scalar_t *out)
{
    scalar_t *scalars = (scalar_t *)malloc(csr->num_cols * sizeof(scalar_t));

    if (witness->W)
        memcpy(&scalars[0], &witness->W[0], witness->nW * sizeof(scalar_t));
    memcpy(&scalars[witness->nW], witness->u, sizeof(scalar_t));
    if (witness->U)
        memcpy(&scalars[witness->nW + 1], &witness->U[0], witness->nU * sizeof(scalar_t));

    for (size_t idx = 0; idx < csr->num_rows; ++idx)
    {
        for (size_t i = csr->row_ptr[idx]; i < csr->row_ptr[idx + 1]; i++)
        {
            out[idx] = out[idx] + scalars[csr->col_idx[i]] * csr->data[i];
        }
    }

    free(scalars);
    return RustError{cudaSuccess};
}

#endif
