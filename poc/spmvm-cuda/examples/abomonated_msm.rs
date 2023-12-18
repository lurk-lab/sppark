use std::{io::Read, time::Instant, collections::HashMap};

use abomonation::Abomonation;
use num::{BigInt, Num, Zero};
use pasta_curves::{
    group::ff::PrimeField,
    pallas,
};
use spmvm_cuda::utils::{gen_scalars, CommitmentKey};

use plotters::prelude::*;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

fn plot_scalars(
    scalars: &[pallas::Scalar],
    out_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(out_file, (640, 480)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption(out_file, ("sans-serif", 50.0))
        .build_cartesian_2d((0u32..1000u32).into_segmented(), 0u32..50000u32)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("1000 * scalar / MODULUS")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    let field_size =
        BigInt::from_str_radix(&pallas::Scalar::MODULUS[2..], 16).unwrap();
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(scalars.iter().map(|x| {
                let x = format!("{:?}", x);
                let x = BigInt::from_str_radix(&x[2..], 16).unwrap();
                let res: BigInt = (1000 * x) / &field_size;
                (*res.to_u32_digits().1.get(0).unwrap_or(&0), 1)
            })),
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

/// cargo run --release --example abomonated
fn main() {
    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { spmvm_cuda::CUDA_OFF = false };
    }

    let ck_primary =
        read_abomonated::<CommitmentKey<pallas::Affine>>("ck_primary".into())
            .unwrap();
    let witness_primary = read_abomonated::<
        Vec<<pallas::Scalar as PrimeField>::Repr>,
    >("witness_primary".into())
    .unwrap();
    let witness_primary = unsafe {
        std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_primary)
    };

    let npoints = witness_primary.len();
    println!("npoints: {}", npoints);

    // println!("randomize");
    // let mut rng = rand::thread_rng();
    // for i in 0..npoints {
    //     if rng.next_u32() % 3 != 0 || witness_primary[i].is_zero_vartime()
    //     {
    //         witness_primary[i] = pallas::Scalar::random(&mut rng);
    //     }
    // }

    let scalars = gen_scalars(npoints);

    let mut total = BigInt::zero();
    let mut dist: HashMap<BigInt, usize> = HashMap::new();
    // let mut small_count = vec![0; 10];
    for i in 0..npoints {
        let wi = format!("{:?}", witness_primary[i]);
        let num = BigInt::from_str_radix(&wi[2..], 16).unwrap();
        total += &num;
        *dist.entry(num).or_insert(0) += 1;
        // for j in 0..10 {
        //     if witness_primary[i] == pallas::Scalar::from(j) {
        //         small_count[j as usize] += 1;
        //     }
        // }
    }

    let average = total / npoints;
    println!("avg: {:?}", average);
    println!("dist len: {:?}", dist.len());
    // println!("smalls: {:?}", small_count);

    let start = Instant::now();
    let res = spmvm_cuda::pallas(&ck_primary.ck[..npoints], &witness_primary);
    println!("time: {:?}", start.elapsed());
    println!("res: {:?}", res);

    plot_scalars(&witness_primary, "plots/witness_primary.png").unwrap();

    let mut total = BigInt::zero();
    for i in 0..npoints {
        let wi = format!("{:?}", scalars[i]);
        total += BigInt::from_str_radix(&wi[2..], 16).unwrap();
    }

    let average = total / npoints;
    println!("avg: {:?}", average);

    let start = Instant::now();
    let res = spmvm_cuda::pallas(&ck_primary.ck[..npoints], &scalars);
    println!("time: {:?}", start.elapsed());
    println!("res: {:?}", res);

    plot_scalars(&scalars, "plots/scalars.png").unwrap();
}