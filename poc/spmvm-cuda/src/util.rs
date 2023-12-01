// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use rand;
use rand::Rng;

use ark_ec::AffineCurve;
use ark_std::UniformRand;
use crate::SparseMatrix;

pub fn generate_random_csr<G: AffineCurve>(n: usize, m: usize) -> SparseMatrix<G> {
    let mut rng = rand::thread_rng();

    let mut data = Vec::new();
    let mut col_idx = Vec::new();
    let mut row_ptr = Vec::new();
    row_ptr.push(0);

    for _ in 0..n {
        let num_elements = rng.gen_range(5..=10); // Random number of elements between 5 to 10
        for _ in 0..num_elements {
            data.push(G::ScalarField::rand(&mut rng)); // Random data value
            col_idx.push(rng.gen_range(0..m)); // Random column index
        }
        row_ptr.push(data.len()); // Add the index of the next row start
    }

    SparseMatrix::new(data, col_idx, row_ptr, n, m)
}

pub fn generate_scalars<G: AffineCurve>(
    len: usize,
) -> Vec<G::ScalarField> {
    let mut rng = rand::thread_rng();

    let scalars = (0..len)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();

        scalars
}