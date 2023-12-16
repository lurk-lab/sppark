#![allow(non_snake_case)]

use std::{collections::HashMap, io::Read, time::Instant};

use abomonation::Abomonation;
use abomonation_derive::Abomonation;
use once_cell::sync::OnceCell;
use pasta_curves::{
    group::ff::{Field, PrimeField},
    pallas,
};
use pasta_msm::{
    spmvm::{
        pallas::{
            sparse_matrix_witness_init_pallas,
            sparse_matrix_witness_with_pallas,
        },
        CudaSparseMatrix, CudaWitness,
    },
    utils::SparseMatrix,
};

use plotters::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

#[derive(Clone, Debug, PartialEq, Eq, Abomonation)]
#[abomonation_bounds(where <F as PrimeField>::Repr: Abomonation)]
struct R1CSShape<F: PrimeField> {
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: SparseMatrix<F>,
    B: SparseMatrix<F>,
    C: SparseMatrix<F>,
    #[abomonate_with(F::Repr)]
    digest: OnceCell<F>,
}

fn plot_rows(
    scalars: &[usize],
    out_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new(out_file, (1280 * 2, 960 * 2)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .margin(20)
        .caption("Histogram Test", ("sans-serif", 50.0))
        .build_cartesian_2d(
            (0u32..100u32).into_segmented(),
            0u32..5000000u32,
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(scalars.iter().map(|x| (*x as u32, 1))),
    )?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", out_file);

    Ok(())
}

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

fn sparse_matrix_statistics(
    matrix: &SparseMatrix<pallas::Scalar>,
    witness: &[pallas::Scalar],
    out_file: &str,
    num_vars: usize,
    num_cons: usize,
) {
    println!("========      {} stats      ========", out_file);
    println!("(rows, cols): {}, {}", matrix.indptr.len() - 1, matrix.cols);
    let row_data = matrix
        .indptr
        .windows(2)
        .map(|ptr| ptr[1] - ptr[0])
        .collect::<Vec<_>>();
    let mut row_hist = HashMap::new();

    for i in row_data.iter() {
        let e = row_hist.entry(*i);
        *e.or_insert(0) += 1usize;
    }

    let mut row_vec: Vec<(usize, usize)> =
        row_hist.iter().map(|(x, y)| (*x, *y)).collect();

    row_vec.sort_by(|a, b| b.1.cmp(&a.1));

    let mut total = 0;
    for (key, value) in row_vec {
        total += value;
        println!("{:<5}: {:<10} ({:<10})", key, value, total);
    }

    println!("sum of rows: {:?}", row_data.par_iter().sum::<usize>());
    println!("row hist: {:?}", row_hist);
    plot_rows(&row_data, out_file).unwrap();

    let csr = CudaSparseMatrix::from(matrix);
    let context = sparse_matrix_witness_init_pallas(&csr);
    let witness = CudaWitness::new(
        &witness[0..num_vars],
        &witness[num_vars],
        &witness[num_vars + 1..],
    );
    let mut res = vec![pallas::Scalar::ZERO; num_cons];
    let start = Instant::now();
    sparse_matrix_witness_with_pallas(&context, &witness, &mut res, 128, 128);
    println!("time: {:?}", start.elapsed());
    println!("res: {:?}", res.len());

    println!("====================================");
    println!()
}

fn r1cs_shape_statistics(
    r1cs_shape: &R1CSShape<pallas::Scalar>,
    witness: &[pallas::Scalar],
) {
    sparse_matrix_statistics(
        &r1cs_shape.A,
        witness,
        "plots/row_data_A.png",
        r1cs_shape.num_vars,
        r1cs_shape.num_cons,
    );
    sparse_matrix_statistics(
        &r1cs_shape.B,
        witness,
        "plots/row_data_B.png",
        r1cs_shape.num_vars,
        r1cs_shape.num_cons,
    );
    sparse_matrix_statistics(
        &r1cs_shape.C,
        witness,
        "plots/row_data_C.png",
        r1cs_shape.num_vars,
        r1cs_shape.num_cons,
    );

    let random_matrix_A = SparseMatrix::<pallas::Scalar>::random(
        r1cs_shape.A.indptr.len() - 1,
        r1cs_shape.A.cols,
        r1cs_shape.A.data.len(),
    );
    sparse_matrix_statistics(
        &random_matrix_A,
        witness,
        "plots/random_A.png",
        r1cs_shape.num_vars,
        r1cs_shape.num_cons,
    );

    let random_matrix_B = SparseMatrix::<pallas::Scalar>::random(
        r1cs_shape.B.indptr.len() - 1,
        r1cs_shape.B.cols,
        r1cs_shape.B.data.len(),
    );
    sparse_matrix_statistics(
        &random_matrix_B,
        witness,
        "plots/random_B.png",
        r1cs_shape.num_vars,
        r1cs_shape.num_cons,
    );

    let random_matrix_C = SparseMatrix::<pallas::Scalar>::random(
        r1cs_shape.C.indptr.len() - 1,
        r1cs_shape.C.cols,
        r1cs_shape.C.data.len(),
    );
    sparse_matrix_statistics(
        &random_matrix_C,
        witness,
        "plots/random_C.png",
        r1cs_shape.num_vars,
        r1cs_shape.num_cons,
    );
}

/// cargo run --release --example abomonated_spmvm
fn main() {
    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };
    }

    let r1cs_primary =
        read_abomonated::<R1CSShape<pallas::Scalar>>("r1cs_primary".into())
            .unwrap();

    println!("data.len: {}", r1cs_primary.A.data.len());

    let witness_primary = read_abomonated::<
        Vec<<pallas::Scalar as PrimeField>::Repr>,
    >("witness_primary".into())
    .unwrap();
    let mut witness_primary = unsafe {
        std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_primary)
    };
    witness_primary.push(pallas::Scalar::ZERO);
    witness_primary.push(pallas::Scalar::from(37));
    witness_primary.push(pallas::Scalar::from(42));

    let npoints = witness_primary.len();
    println!("npoints: {}", npoints);
    let mut count = 0;
    for i in witness_primary.iter() {
        if i.is_zero_vartime() {
            count += 1;
        }
    }
    println!("zeros: {} ({}%)", count, count as f32 / npoints as f32);

    // let scalars = gen_scalars(npoints);
    r1cs_shape_statistics(&r1cs_primary, &witness_primary);

    let start = Instant::now();
    let res = r1cs_primary.A.multiply_vec(&witness_primary);
    println!("time: {:?}", start.elapsed());
    println!("res: {:?}", res.len());
}
