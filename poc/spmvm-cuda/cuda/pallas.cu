// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

#include <ff/pasta.hpp>

typedef jacobian_t<pallas_t> point_t;
typedef xyzz_t<pallas_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef vesta_t scalar_t;

#include <msm/pippenger.cuh>
#include <spmvm/spmvm.cuh>

#ifndef __CUDA_ARCH__

extern "C" void drop_spmvm_context_pallas(spmvm_context_t<scalar_t> &ref)
{
    drop_spmvm_context<scalar_t>(ref);
}

extern "C" RustError cuda_sparse_matrix_mul_pallas(spmvm_host_t<scalar_t> *csr, const scalar_t *scalars, scalar_t *out, size_t nthreads)
{
    return sparse_matrix_mul<scalar_t>(csr, scalars, out, nthreads);
}

extern "C" RustError cuda_sparse_matrix_witness_init_pallas(
    spmvm_host_t<scalar_t> *csr, spmvm_context_t<scalar_t> *context)
{
    return sparse_matrix_witness_init<scalar_t>(csr, context);
}

extern "C" RustError cuda_sparse_matrix_witness_pallas(
    spmvm_host_t<scalar_t> *csr, const witness_t<scalar_t> *witness, scalar_t *out, size_t nthreads)
{
    return sparse_matrix_witness<scalar_t>(csr, witness, out, nthreads);
}

extern "C" RustError cuda_sparse_matrix_witness_with_pallas(
    spmvm_context_t<scalar_t> *context, const witness_t<scalar_t> *witness, scalar_t *out, size_t nblocks, size_t nthreads)
{
    return sparse_matrix_witness_with<scalar_t>(context, witness, out, nblocks, nthreads);
}

extern "C" RustError cuda_sparse_matrix_witness_pallas_cpu(
    spmvm_host_t<scalar_t> *csr, const witness_t<scalar_t> *witness, scalar_t *out)
{
    return sparse_matrix_witness_cpu<scalar_t>(csr, witness, out);
}

extern "C" void drop_msm_context_pallas(msm_context_t<affine_t::mem_t> &ref)
{
    CUDA_OK(cudaFree(ref.d_points));
}

extern "C" RustError
cuda_pippenger_pallas_init(const affine_t points[], size_t npoints, msm_context_t<affine_t::mem_t> *msm_context)
{
    return mult_pippenger_init<bucket_t, point_t, affine_t, scalar_t>(points, npoints, msm_context);
}

extern "C" RustError cuda_pippenger_pallas(point_t *out, const affine_t points[], size_t npoints,
                                           const scalar_t scalars[])
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars);
}

extern "C" RustError cuda_pippenger_pallas_with(point_t *out, msm_context_t<affine_t::mem_t> *msm_context, size_t npoints,
                                                const scalar_t scalars[])
{
    return mult_pippenger_with<bucket_t, point_t, affine_t, scalar_t>(out, msm_context, npoints, scalars);
}
#endif