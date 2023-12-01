// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bls12_377")]
use ark_bls12_377::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
#[cfg(feature = "bls12_381")]
use ark_bls12_381::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
#[cfg(feature = "bn254")]
use ark_bn254::G1Affine;
use ark_ec::AffineCurve;

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;

pub mod util;

// const scalar_t *data;
// const size_t *col_idx;
// const size_t *row_ptr;

// size_t row_size;
// size_t col_size;
// size_t nnz;
#[repr(C)]
pub struct SparseMatrix<G: AffineCurve> {
    data: *const c_void,
    col_idx: *const c_void,
    row_ptr: *const c_void,

    row_size: usize,
    col_size: usize,
    nnz: usize,

    _p: PhantomData<G>,
}

impl<G: AffineCurve> SparseMatrix<G> {
    fn new(data: Vec<G::ScalarField>, col_idx: Vec<usize>, row_ptr: Vec<usize>, row_size: usize, col_size: usize) -> Self {
        let data_ptr = data.as_ptr();
        let col_idx_ptr = col_idx.as_ptr();
        let row_ptr_ptr = row_ptr.as_ptr();
        let nnz = data.len();

        mem::forget(data);
        mem::forget(col_idx);
        mem::forget(row_ptr);

        SparseMatrix {
            data: data_ptr as *const c_void,
            col_idx: col_idx_ptr as *const c_void,
            row_ptr: row_ptr_ptr as *const c_void,
            row_size,
            col_size,
            nnz,
            _p: PhantomData,
        }
    }
}

// Ensure to add proper cleanup for SparseMatrix to avoid memory leaks.
impl<G: AffineCurve> Drop for SparseMatrix<G> {
    fn drop(&mut self) {
        unsafe {
            let _ = Vec::from_raw_parts(self.data as *mut G::ScalarField, self.nnz, self.nnz);
            let _ = Vec::from_raw_parts(self.col_idx as *mut usize, self.nnz, self.nnz);
            let _ = Vec::from_raw_parts(self.row_ptr as *mut usize, self.row_size + 1, self.row_size + 1);
        }
    }
}


pub fn sparse_matrix_vector_arkworks<G: AffineCurve>(
    csr: &SparseMatrix<G>,
    scalars: &[G::ScalarField],
) -> Vec<G::ScalarField> {
    #[cfg_attr(feature = "quiet", allow(improper_ctypes))]
    extern "C" {
        fn spmvm<G>(
            out: *mut G::ScalarField,
            csr: &SparseMatrix<G>,
            scalars: *const G::ScalarField,
        ) -> sppark::Error;
    }

    let ret = Vec::with_capacity(csr.row_size);
    let err = unsafe {
        spmvm(ret.as_ptr() as *mut _, csr, scalars.as_ptr() as *const _)
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}
