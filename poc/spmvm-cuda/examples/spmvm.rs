#![allow(non_snake_case)]

use std::time::Instant;

use pasta_curves::{
    group::ff::Field,
    pallas,
};
use spmvm_cuda::{
    spmvm::{
        pallas::{
            sparse_matrix_witness_init_pallas,
            sparse_matrix_witness_with_pallas,
        },
        CudaSparseMatrix, CudaWitness,
    },
    utils::{gen_scalars, SparseMatrix},
};

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

/// cargo run --release --example spmvm
fn main() {
    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { spmvm_cuda::CUDA_OFF = false };
    }

    let csr = SparseMatrix::<pallas::Scalar>::random(200, 100, 2000);
    let cuda_csr = CudaSparseMatrix::from(&csr);
    let witness = gen_scalars(100);
    let cuda_witness =
        CudaWitness::new(&witness[0..97], &witness[97], &witness[98..]);

    let context = sparse_matrix_witness_init_pallas(&cuda_csr);
    let mut cuda_res = vec![pallas::Scalar::ZERO; 200];
    let start = Instant::now();
    sparse_matrix_witness_with_pallas(&context, &cuda_witness, &mut cuda_res, 84, 128);
    println!("time: {:?}", start.elapsed());
    println!("res: {:?}", cuda_res.len());
}
