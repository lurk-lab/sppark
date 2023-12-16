#![allow(non_snake_case)]

use std::ffi::c_void;

use pasta_curves::{vesta, group::ff::Field};

use super::{CudaSparseMatrix, CudaWitness};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct SpMVMContextVesta {
    pub d_data: *const c_void,
    pub d_col_idx: *const c_void,
    pub d_row_ptr: *const c_void,

    pub num_rows: usize,
    pub num_cols: usize,
    pub nnz: usize,

    pub d_scalars: *const c_void,
    pub d_out: *const c_void,
}

unsafe impl Send for SpMVMContextVesta {}
unsafe impl Sync for SpMVMContextVesta {}

impl Default for SpMVMContextVesta {
    fn default() -> Self {
        Self {
            d_data: core::ptr::null(),
            d_col_idx: core::ptr::null(),
            d_row_ptr: core::ptr::null(),
            num_rows: 0,
            num_cols: 0,
            nnz: 0,
            d_scalars: core::ptr::null(),
            d_out: core::ptr::null(),
        }
    }
}

// TODO: check for device-side memory leaks
impl Drop for SpMVMContextVesta {
    fn drop(&mut self) {
        extern "C" {
            fn drop_spmvm_context_vesta(by_ref: &SpMVMContextVesta);
        }
        unsafe {
            drop_spmvm_context_vesta(std::mem::transmute::<&_, &_>(self))
        };

        self.d_data = core::ptr::null();
        self.d_col_idx = core::ptr::null();
        self.d_row_ptr = core::ptr::null();

        self.num_rows = 0;
        self.num_cols = 0;
        self.nnz = 0;

        self.d_scalars = core::ptr::null();
        self.d_out = core::ptr::null();
    }
}

pub fn sparse_matrix_mul_vesta(
    csr: &CudaSparseMatrix<vesta::Scalar>,
    scalars: &[vesta::Scalar],
    nthreads: usize,
) -> Vec<vesta::Scalar> {
    extern "C" {
        fn cuda_sparse_matrix_mul_vesta(
            csr: *const CudaSparseMatrix<vesta::Scalar>,
            scalars: *const vesta::Scalar,
            out: *mut vesta::Scalar,
            nthreads: usize,
        ) -> sppark::Error;
    }

    let mut out = vec![vesta::Scalar::ZERO; csr.num_rows];
    let err = unsafe {
        cuda_sparse_matrix_mul_vesta(
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

pub fn sparse_matrix_witness_init_vesta(
    csr: &CudaSparseMatrix<vesta::Scalar>,
) -> SpMVMContextVesta {
    extern "C" {
        fn cuda_sparse_matrix_witness_init_vesta(
            csr: *const CudaSparseMatrix<vesta::Scalar>,
            context: *mut SpMVMContextVesta,
        ) -> sppark::Error;
    }

    let mut context = SpMVMContextVesta::default();
    let err = unsafe {
        cuda_sparse_matrix_witness_init_vesta(
            csr as *const _,
            &mut context as *mut _,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    context
}

pub fn sparse_matrix_witness_vesta(
    csr: &CudaSparseMatrix<vesta::Scalar>,
    witness: &CudaWitness<vesta::Scalar>,
    buffer: &mut [vesta::Scalar],
    nthreads: usize,
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_vesta(
            csr: *const CudaSparseMatrix<vesta::Scalar>,
            witness: *const CudaWitness<vesta::Scalar>,
            out: *mut vesta::Scalar,
            nthreads: usize,
        ) -> sppark::Error;
    }

    assert_eq!(
        witness.nW + witness.nU + 1,
        csr.num_cols,
        "invalid witness size"
    );

    let err = unsafe {
        cuda_sparse_matrix_witness_vesta(
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

pub fn sparse_matrix_witness_with_vesta(
    context: &SpMVMContextVesta,
    witness: &CudaWitness<vesta::Scalar>,
    buffer: &mut [vesta::Scalar],
    nblocks: usize,
    nthreads: usize,
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_with_vesta(
            context: *const SpMVMContextVesta,
            witness: *const CudaWitness<vesta::Scalar>,
            out: *mut vesta::Scalar,
            nblocks: usize,
            nthreads: usize,
        ) -> sppark::Error;
    }

    assert_eq!(
        witness.nW + witness.nU + 1,
        context.num_cols,
        "invalid witness size"
    );

    let err = unsafe {
        cuda_sparse_matrix_witness_with_vesta(
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

pub fn sparse_matrix_witness_vesta_cpu(
    csr: &CudaSparseMatrix<vesta::Scalar>,
    witness: &CudaWitness<vesta::Scalar>,
    buffer: &mut [vesta::Scalar],
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_vesta_cpu(
            csr: *const CudaSparseMatrix<vesta::Scalar>,
            witness: *const CudaWitness<vesta::Scalar>,
            out: *mut vesta::Scalar,
        ) -> sppark::Error;
    }

    assert_eq!(
        witness.nW + witness.nU + 1,
        csr.num_cols,
        "invalid witness size"
    );

    let err = unsafe {
        cuda_sparse_matrix_witness_vesta_cpu(
            csr as *const _,
            witness as *const _,
            buffer.as_mut_ptr(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}