#![allow(non_snake_case)]

use std::ffi::c_void;

use pasta_curves::{pallas, group::ff::Field};

use super::{CudaSparseMatrix, CudaWitness};


#[repr(C)]
#[derive(Debug, Clone)]
pub struct SpMVMContextPallas {
    pub d_context: *const c_void,
}

unsafe impl Send for SpMVMContextPallas {}
unsafe impl Sync for SpMVMContextPallas {}

impl Default for SpMVMContextPallas {
    fn default() -> Self {
        Self {
            d_context: core::ptr::null(),
        }
    }
}

// TODO: check for device-side memory leaks
impl Drop for SpMVMContextPallas {
    fn drop(&mut self) {
        extern "C" {
            fn drop_spmvm_context_pallas(by_ref: &SpMVMContextPallas);
        }
        unsafe {
            drop_spmvm_context_pallas(std::mem::transmute::<&_, &_>(self))
        };

        self.d_context = core::ptr::null();
    }
}

pub fn sparse_matrix_mul_pallas(
    csr: &CudaSparseMatrix<pallas::Scalar>,
    scalars: &[pallas::Scalar],
    nthreads: usize,
) -> Vec<pallas::Scalar> {
    extern "C" {
        fn cuda_sparse_matrix_mul_pallas(
            csr: *const CudaSparseMatrix<pallas::Scalar>,
            scalars: *const pallas::Scalar,
            out: *mut pallas::Scalar,
            nthreads: usize,
        ) -> sppark::Error;
    }

    let mut out = vec![pallas::Scalar::ZERO; csr.num_rows];
    let err = unsafe {
        cuda_sparse_matrix_mul_pallas(
            csr as *const _,
            scalars.as_ptr(),
            out.as_mut_ptr(),
            nthreads,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    out
}

pub fn sparse_matrix_witness_init_pallas(
    csr: &CudaSparseMatrix<pallas::Scalar>,
) -> SpMVMContextPallas {
    extern "C" {
        fn cuda_sparse_matrix_witness_init_pallas(
            csr: *const CudaSparseMatrix<pallas::Scalar>,
            context: *mut SpMVMContextPallas,
        ) -> sppark::Error;
    }

    let mut context = SpMVMContextPallas::default();
    let err = unsafe {
        cuda_sparse_matrix_witness_init_pallas(
            csr as *const _,
            &mut context as *mut _,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    context
}

pub fn sparse_matrix_witness_pallas(
    csr: &CudaSparseMatrix<pallas::Scalar>,
    witness: &CudaWitness<pallas::Scalar>,
    buffer: &mut [pallas::Scalar],
    nthreads: usize,
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_pallas(
            csr: *const CudaSparseMatrix<pallas::Scalar>,
            witness: *const CudaWitness<pallas::Scalar>,
            out: *mut pallas::Scalar,
            nthreads: usize,
        ) -> sppark::Error;
    }

    assert_eq!(
        witness.nW + witness.nU + 1,
        csr.num_cols,
        "invalid witness size"
    );

    let err = unsafe {
        cuda_sparse_matrix_witness_pallas(
            csr as *const _,
            witness as *const _,
            buffer.as_mut_ptr(),
            nthreads,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn sparse_matrix_witness_with_pallas(
    context: &SpMVMContextPallas,
    witness: &CudaWitness<pallas::Scalar>,
    buffer: &mut [pallas::Scalar],
    nblocks: usize,
    nthreads: usize,
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_with_pallas(
            context: *const SpMVMContextPallas,
            witness: *const CudaWitness<pallas::Scalar>,
            out: *mut pallas::Scalar,
            nblocks: usize,
            nthreads: usize,
        ) -> sppark::Error;
    }

    let err = unsafe {
        cuda_sparse_matrix_witness_with_pallas(
            context as *const _,
            witness as *const _,
            buffer.as_mut_ptr(),
            nblocks,
            nthreads,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn sparse_matrix_witness_pallas_cpu(
    csr: &CudaSparseMatrix<pallas::Scalar>,
    witness: &CudaWitness<pallas::Scalar>,
    buffer: &mut [pallas::Scalar],
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_pallas_cpu(
            csr: *const CudaSparseMatrix<pallas::Scalar>,
            witness: *const CudaWitness<pallas::Scalar>,
            out: *mut pallas::Scalar,
        ) -> sppark::Error;
    }

    assert_eq!(
        witness.nW + witness.nU + 1,
        csr.num_cols,
        "invalid witness size"
    );

    let err = unsafe {
        cuda_sparse_matrix_witness_pallas_cpu(
            csr as *const _,
            witness as *const _,
            buffer.as_mut_ptr(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}