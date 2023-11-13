// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

template <int intermediate_mul>
__launch_bounds__(768, 1) __global__
void _CT_NTT(const unsigned int radix, const unsigned int lg_domain_size,
             const unsigned int stage, const unsigned int iterations,
             fr_t* d_inout, const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
             const fr_t* d_radix6_twiddles, const fr_t* d_radixX_twiddles,
             const fr_t* d_intermediate_twiddles,
             const unsigned int intermediate_twiddle_shift,
             const bool is_intt, const fr_t d_domain_size_inverse)
{
#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
    __builtin_assume(radix <= 10);
    __builtin_assume(iterations <= radix);
    __builtin_assume(stage <= lg_domain_size - iterations);
#endif

    const index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    const index_t out_mask = ((index_t)1 << (stage + iterations - 1)) - 1;
#if 1
    const index_t thread_ntt_pos = (tid & out_mask) >> (iterations - 1);
#else
    const index_t inp_mask = ((index_t)1 << stage) - 1;
    const index_t thread_ntt_pos = (tid >> (iterations - 1)) & inp_mask;
#endif

    // rearrange |tid|'s bits
    index_t idx0 = (tid & ~out_mask) | ((tid << stage) & out_mask);
    idx0 = idx0 * 2 + thread_ntt_pos;
    index_t idx1 = idx0 + ((index_t)1 << stage);

    fr_t r0 = d_inout[idx0];
    fr_t r1 = d_inout[idx1];

    if (intermediate_mul == 1) {
        unsigned int diff_mask = (1 << (iterations - 1)) - 1;
        unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - stage;

        index_t root_idx0 = bit_rev(thread_ntt_idx, nbits) * thread_ntt_pos;
        index_t root_idx1 = thread_ntt_pos << (nbits - 1);

        fr_t first_root, second_root;
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);
        second_root *= first_root;

        r0 *= first_root;
        r1 *= second_root;
    } else if (intermediate_mul == 2) {
        unsigned int diff_mask = (1 << (iterations - 1)) - 1;
        unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
        unsigned int nbits = intermediate_twiddle_shift + iterations;

        index_t root_idx0 = bit_rev(thread_ntt_idx, nbits);
        index_t root_idx1 = bit_rev(thread_ntt_idx + 1, nbits);

        fr_t t0 = d_intermediate_twiddles[(thread_ntt_pos << radix) + root_idx0];
        fr_t t1 = d_intermediate_twiddles[(thread_ntt_pos << radix) + root_idx1];

        r0 *= t0;
        r1 *= t1;
    }

    {
        fr_t t = r1;
        r1 = r0 - t;
        r0 = r0 + t;
    }

    for (int s = 1; s < min(iterations, 6); s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

#ifdef __CUDA_ARCH__
        fr_t x = fr_t::csel(r1, r0, pos);
        shfl_bfly(x, laneMask);
        r0 = fr_t::csel(x, r0, !pos);
        r1 = fr_t::csel(x, r1, pos);
#endif

        fr_t t = d_radix6_twiddles[rank << (6 - (s + 1))];
        t *= r1;

        r1 = r0 - t;
        r0 = r0 + t;
    }

    for (int s = 6; s < iterations; s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t t = d_radixX_twiddles[rank << (radix - (s + 1))];

        // shfl_bfly through the shared memory
        extern __shared__ fr_t shared_exchange[];

#ifdef __CUDA_ARCH__
        fr_t x = fr_t::csel(r1, r0, pos);
        __syncthreads();
        shared_exchange[threadIdx.x] = x;
        __syncthreads();
        x = shared_exchange[threadIdx.x ^ laneMask];
        r0 = fr_t::csel(x, r0, !pos);
        r1 = fr_t::csel(x, r1, pos);
#endif

        t *= r1;

        r1 = r0 - t;
        r0 = r0 + t;
    }

    if (is_intt && (stage + iterations) == lg_domain_size) {
        r0 *= d_domain_size_inverse;
        r1 *= d_domain_size_inverse;
    }

    // rotate "iterations" bits in indices
    index_t mask = ((index_t)1 << (stage + iterations)) - ((index_t)1 << stage);
    index_t rotw = idx0 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

    d_inout[idx0] = r0;
    d_inout[idx1] = r1;
}

#define NTT_ARGUMENTS \
        unsigned int, unsigned int, unsigned int, unsigned int, fr_t*, \
        const fr_t (*)[WINDOW_SIZE], const fr_t*, const fr_t*, const fr_t*, \
        unsigned int, bool, fr_t

template __global__ void _CT_NTT<0>(NTT_ARGUMENTS);
template __global__ void _CT_NTT<1>(NTT_ARGUMENTS);
template __global__ void _CT_NTT<2>(NTT_ARGUMENTS);

#undef NTT_ARGUMENTS

#ifndef __CUDA_ARCH__

class CT_launcher {
    fr_t* d_inout;
    const int lg_domain_size;
    bool is_intt;
    int stage;
    const NTTParameters& ntt_parameters;
    const cudaStream_t& stream;

public:
    CT_launcher(fr_t* d_ptr, int lg_dsz, bool intt,
                const NTTParameters& params, const cudaStream_t& s)
      : d_inout(d_ptr), lg_domain_size(lg_dsz), is_intt(intt), stage(0),
        ntt_parameters(params), stream(s)
    {}

    void step(int iterations)
    {
        assert(iterations <= 10);

        const int radix = iterations < 6 ? 6 : iterations;

        index_t num_threads = (index_t)1 << (lg_domain_size - 1);
        index_t block_size = 1 << (radix - 1);
        index_t num_blocks;

        block_size = (num_threads <= block_size) ? num_threads : block_size;
        num_blocks = (num_threads + block_size - 1) / block_size;

        assert(num_blocks == (unsigned int)num_blocks);

        fr_t* d_radixX_twiddles = nullptr;
        fr_t* d_intermediate_twiddles = nullptr;
        unsigned int intermediate_twiddle_shift = 0;

        #define NTT_CONFIGURATION \
                num_blocks, block_size, sizeof(fr_t) * block_size, stream

        #define NTT_ARGUMENTS radix, lg_domain_size, stage, iterations, \
                d_inout, ntt_parameters.partial_twiddles, \
                ntt_parameters.radix6_twiddles, d_radixX_twiddles, \
                d_intermediate_twiddles, intermediate_twiddle_shift, \
                is_intt, domain_size_inverse[lg_domain_size]

        switch (radix) {
        case 6:
            switch (stage) {
            case 0:
                _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 6:
                intermediate_twiddle_shift = std::max(12 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix6_twiddles_6;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 12:
                intermediate_twiddle_shift = std::max(18 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix6_twiddles_12;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        case 7:
            d_radixX_twiddles = ntt_parameters.radix7_twiddles;
            switch (stage) {
            case 0:
                _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 7:
                intermediate_twiddle_shift = std::max(14 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix7_twiddles_7;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        case 8:
            d_radixX_twiddles = ntt_parameters.radix8_twiddles;
            switch (stage) {
            case 0:
                _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 8:
                intermediate_twiddle_shift = std::max(16 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix8_twiddles_8;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        case 9:
            d_radixX_twiddles = ntt_parameters.radix9_twiddles;
            switch (stage) {
            case 0:
                _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 9:
                intermediate_twiddle_shift = std::max(18 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix9_twiddles_9;
                _CT_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        case 10:
            d_radixX_twiddles = ntt_parameters.radix10_twiddles;
            switch (stage) {
            case 0:
                _CT_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _CT_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        default:
            assert(false);
        }

        #undef NTT_CONFIGURATION
        #undef NTT_ARGUMENTS

        CUDA_OK(cudaGetLastError());

        stage += radix;
    }
};

void CT_NTT(fr_t* d_inout, const int lg_domain_size, bool intt,
            const NTTParameters& ntt_parameters, const cudaStream_t& stream)
{
    CT_launcher params{d_inout, lg_domain_size, intt, ntt_parameters, stream};

    if (lg_domain_size <= 10) {
        params.step(lg_domain_size);
    } else if (lg_domain_size <= 17) {
        params.step(lg_domain_size / 2 + lg_domain_size % 2);
        params.step(lg_domain_size / 2);
    } else if (lg_domain_size <= 30) {
        int step = lg_domain_size / 3;
        int rem = lg_domain_size % 3;
        params.step(step);
        params.step(step + (lg_domain_size == 29 ? 1 : 0));
        params.step(step + (lg_domain_size == 29 ? 1 : rem));
    } else if (lg_domain_size <= 40) {
        int step = lg_domain_size / 4;
        int rem = lg_domain_size % 4;
        params.step(step);
        params.step(step + (rem > 2));
        params.step(step + (rem > 1));
        params.step(step + (rem > 0));
    } else {
        assert(false);
    }
}

#endif
