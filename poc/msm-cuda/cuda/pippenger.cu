// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ff/bls12-381.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

template <typename T>
struct msm_context_t
{
    T *d_points;
};

extern "C" RustError
mult_pippenger_init(const affine_t points[], size_t npoints, msm_context_t<affine_t::mem_t> *msm_context)
{
    msm_t<bucket_t, point_t, affine_t, scalar_t> msm{points, npoints};
    msm_context->d_points = msm.get_d_points();
    return RustError{cudaSuccess};
}

extern "C" RustError mult_pippenger(point_t *out, const affine_t points[], size_t npoints,
                                    const scalar_t scalars[])
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false);
}

extern "C" RustError mult_pippenger_with(point_t *out, const msm_context_t<affine_t::mem_t> *msm_context, size_t npoints,
                                         const scalar_t scalars[])
{
    return mult_pippenger_with<bucket_t>(out, msm_context->d_points, npoints, scalars, false);
}
#endif
