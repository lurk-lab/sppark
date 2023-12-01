// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#if defined(FEATURE_BLS12_381)
#include <ff/bls12-381-fp2.hpp>
#elif defined(FEATURE_BLS12_377)
#include <ff/bls12-377-fp2.hpp>
#elif defined(FEATURE_BN254)
#include <ff/alt_bn128.hpp>
#else
#error "no FEATURE"
#endif

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef fr_t scalar_t;

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <spmvm/spmvm.cuh>

extern "C" RustError::by_value spmvm(scalar_t out[], const csr_t_host<scalar_t> &csr, const scalar_t scalars[])
{
    return spmvm<scalar_t>(out, csr, scalars);
}
