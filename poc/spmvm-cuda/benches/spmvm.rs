// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(non_snake_case)]

use std::io::Read;

use abomonation::Abomonation;
use criterion::{criterion_group, criterion_main, Criterion};

use pasta_curves::{
    group::ff::{Field, PrimeField},
    pallas,
};
use spmvm_cuda::{
    self,
    spmvm::{
        pallas::{
            sparse_matrix_witness_init_pallas, sparse_matrix_witness_pallas,
            sparse_matrix_witness_with_pallas,
        },
        CudaSparseMatrix, CudaWitness,
    },
    utils::SparseMatrix,
};
use rand::Rng;

const ROWS: usize = 99087;
const COLS: usize = 81500;
const NNZ: usize = 10751580;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

include!("../src/tests.rs");

fn read_abomonated<T: Abomonation + Clone>(name: String) -> std::io::Result<T> {
    use std::fs::OpenOptions;
    use std::io::BufReader;

    let arecibo = home::home_dir().unwrap().join(".arecibo");

    let data = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(arecibo.join(name))?;
    let mut reader = BufReader::new(data);
    let mut bytes = vec![];
    reader.read_to_end(&mut bytes)?;

    let (data, _) = unsafe { abomonation::decode::<T>(&mut bytes).unwrap() };

    Ok(data.clone())
}

fn criterion_benchmark(c: &mut Criterion) {
    // let bench_npow: usize = std::env::var("BENCH_NPOW")
    //     .unwrap_or("17".to_string())
    //     .parse()
    //     .unwrap();
    // let n = 1usize << (bench_npow + 1);
    // let m = 1usize << bench_npow;

    println!("generating random matrix and scalars, just hang on...");
    let csr = SparseMatrix::random(ROWS, COLS, NNZ);
    let cuda_csr =
        CudaSparseMatrix::new(&csr.data, &csr.indices, &csr.indptr, ROWS, COLS);

    // let witness_primary = read_abomonated::<
    //     Vec<<pallas::Scalar as PrimeField>::Repr>,
    // >("witness_primary".into())
    // .unwrap();
    // let mut witness_primary = unsafe {
    //     std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_primary)
    // };
    // witness_primary.push(pallas::Scalar::ZERO);
    // witness_primary.push(pallas::Scalar::from(37));
    // witness_primary.push(pallas::Scalar::from(42));

    // let num_vars = COLS - 3;
    // let witness = CudaWitness::new(
    //     &witness_primary[0..num_vars],
    //     &witness_primary[num_vars],
    //     &witness_primary[num_vars + 1..],
    // );

    let W = crate::tests::gen_scalars(COLS - 3);
    let U = crate::tests::gen_scalars(2);
    let witness = CudaWitness::new(&W, &pallas::Scalar::ONE, &U);
    // let scalars = [W.clone(), vec![pallas::Scalar::ONE], U.clone()].concat();

    #[cfg(feature = "cuda")]
    {
        unsafe { spmvm_cuda::CUDA_OFF = true };
    }

    // let mut group = c.benchmark_group("CPU");
    // group.sample_size(10);

    // group.bench_function(format!("{} points", csr.data.len()), |b| {
    //     b.iter(|| {
    //         let _ = csr.multiply_vec(&scalars);
    //     })
    // });

    // group.finish();

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { spmvm_cuda::CUDA_OFF = false };

        let mut group = c.benchmark_group("GPU");
        group.sample_size(20);

        let mut cuda_res = vec![pallas::Scalar::ONE; cuda_csr.num_rows];
        let context = sparse_matrix_witness_init_pallas(&cuda_csr);
        for nblocks in [64, 128, 256] {
            for nthreads in [64, 128, 256] {
                group.bench_function(
                    format!(
                        "{} points, {}, {}",
                        cuda_csr.nnz, nblocks, nthreads
                    ),
                    |b| {
                        b.iter(|| {
                            let _ = sparse_matrix_witness_with_pallas(
                                &context,
                                &witness,
                                &mut cuda_res,
                                nblocks,
                                nthreads,
                            );
                        })
                    },
                );
            }
        }

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
