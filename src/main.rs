use cust::prelude::*;
use libm::expf;
use libm::logf;
use libm::powf;
use libm::sqrtf;
use ndarray::Array1;
use rand::Rng;
use rayon::prelude::*;
use std::error::Error;
use std::f32::consts::TAU;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;

const N: usize = 1024;
const STEPS: usize = 1000;
const DT: f32 = 0.01;
const M_S: f32 = 1.0;
const G: f32 = 39.5;
const ATOL: f32 = 1.49e-12;
const RTOL: f32 = 1.49e-12;
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

const SIGMA: f32 = 10.0;
const RHO: f32 = 28.0;
const BETA: f32 = 8.0 / 3.0;

#[inline(always)]
fn lorenz_derivatives(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let dx = SIGMA * (y - x);
    let dy = -x * z + RHO * x - y;
    let dz = x * y - BETA * z;
    (dx, dy, dz)
}

fn compute_energies(state_out: &[f32], n: usize, steps: usize, g: f32, mass: f32) -> Vec<f32> {
    let mut energies = Vec::with_capacity(steps);

    for s in 0..steps {
        let mut ke = 0.0;
        let mut pe = 0.0;

        // kinetic
        for i in 0..n {
            let offset = (s * n + i) * 6;
            let vx = state_out[offset + 3];
            let vy = state_out[offset + 4];
            let vz = state_out[offset + 5];
            ke += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
        }

        // potential (pairwise)
        let base_offset = s * n * 6;

        for i in 0..n {
            let xi = state_out[base_offset + i * 6 + 0];
            let yi = state_out[base_offset + i * 6 + 1];
            let zi = state_out[base_offset + i * 6 + 2];

            for j in 0..n {
                if i == j {
                    continue;
                }

                let xj = state_out[base_offset + j * 6 + 0];
                let yj = state_out[base_offset + j * 6 + 1];
                let zj = state_out[base_offset + j * 6 + 2];

                let dx = xi - xj;
                let dy = yi - yj;
                let dz = zi - zj;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                pe -= g * mass * mass / r;
            }
        }

        energies.push(ke + pe);
    }

    energies
}

fn compute_energies_parallel(
    state_out: &[f32],
    n: usize,
    steps: usize,
    g: f32,
    mass: f32,
) -> Vec<f32> {
    (0..steps)
        .into_par_iter()
        .map(|s| {
            let mut ke = 0.0;
            let mut pe = 0.0;

            // kinetic
            for i in 0..n {
                let offset = (s * n + i) * 6;
                let vx = state_out[offset + 3];
                let vy = state_out[offset + 4];
                let vz = state_out[offset + 5];
                ke += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
            }

            // potential (pairwise)
            let base_offset = s * n * 6;

            for i in 0..n {
                let xi = state_out[base_offset + i * 6 + 0];
                let yi = state_out[base_offset + i * 6 + 1];
                let zi = state_out[base_offset + i * 6 + 2];

                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let xj = state_out[base_offset + j * 6 + 0];
                    let yj = state_out[base_offset + j * 6 + 1];
                    let zj = state_out[base_offset + j * 6 + 2];

                    let dx = xi - xj;
                    let dy = yi - yj;
                    let dz = zi - zj;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();

                    pe -= g * mass * mass / r;
                }
            }

            ke + pe
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    // CUDA setup
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    let mut state_out = vec![0.0f32; N * STEPS * 6];

    let mut rng = rand::rng();
    // initial cond
    // for i in 0..N {
    //     let x = rng.random_range(-10.0..10.0);
    //     let y = rng.random_range(-10.0..10.0);
    //     let z = rng.random_range(10.0..30.0);
    //     let (vx, vy, vz) = lorenz_derivatives(x, y, z);

    //     let offset = i * 6;
    //     state_out[offset + 0] = x;
    //     state_out[offset + 1] = y;
    //     state_out[offset + 2] = z;
    //     state_out[offset + 3] = vx;
    //     state_out[offset + 4] = vy;
    //     state_out[offset + 5] = vz;
    // }

    for i in 0..N {
        // circular for testing
        let r = rng.gen_range(1.0..5.0);
        let theta = rng.gen_range(0.0..TAU);
        let x = r * theta.cos();
        let y = r * theta.sin();
        let z = 0.0;
        let v = (G * M_S / r).sqrt();
        let vx = -v * theta.sin();
        let vy = v * theta.cos();
        let vz = 0.0;

        let offset = i * 6;
        state_out[offset + 0] = x;
        state_out[offset + 1] = y;
        state_out[offset + 2] = z;
        state_out[offset + 3] = vx;
        state_out[offset + 4] = vy;
        state_out[offset + 5] = vz;
    }

    let mut dev_state_out = state_out.as_slice().as_dbuf()?;
    let mut dev_rk_err = DeviceBuffer::<f32>::zeroed(N * STEPS)?; // just directly to avoid copy from host

    // let kernel_euler_step = module.get_function("euler_step")?;
    let kernel_dopr54_step = module.get_function("dopr54_step")?;

    for step in 1..STEPS {
        unsafe {
            // launch!(
            //     kernel_euler_step<<<(N as u32 + 127) / 128, 128, 0, stream>>>(
            //         dev_state_out.as_device_ptr(),
            //         N,
            //         STEPS,
            //         step,
            //         DT
            //     )
            // )?;
            let t0 = step as f32 * DT;
            launch!(
                kernel_dopr54_step<<<(N as u32 + 127) / 128, 128, 0, stream>>>(
                    dev_state_out.as_device_ptr(),
                    N,
                    STEPS,
                    step,
                    DT,
                    t0,
                    ATOL,
                    RTOL,
                    dev_rk_err.as_device_ptr()
                )
            )?;
        }
    }

    stream.synchronize()?;

    dev_state_out.copy_to(&mut state_out)?;
    let mut rk_err = vec![0.0f32; N * STEPS];
    dev_rk_err.copy_to(&mut rk_err)?;

    println!("Compute finished");

    // let energies = compute_energies(&state_out, N, STEPS, G, M_S);
    // let mut ef = File::create("energy.csv")?;
    // writeln!(ef, "step,energy")?;
    // for (i, e) in energies.iter().enumerate() {
    //     writeln!(ef, "{},{}", i, e)?;
    // }

    let energies_parallel = compute_energies_parallel(&state_out, N, STEPS, G, M_S);
    let mut ef = File::create("energy_parallel.csv")?;
    writeln!(ef, "step,energy")?;
    for (i, e) in energies_parallel.iter().enumerate() {
        writeln!(ef, "{},{}", i, e)?;
    }

    let final_offset = ((STEPS - 1) * N + 0) * 6;
    println!("Final x[0]: {}", state_out[final_offset + 0]);

    let x0: Vec<f32> = (0..STEPS).map(|s| state_out[(s * N + 0) * 6 + 0]).collect();
    let y0: Vec<f32> = (0..STEPS).map(|s| state_out[(s * N + 0) * 6 + 1]).collect();
    let z0: Vec<f32> = (0..STEPS).map(|s| state_out[(s * N + 0) * 6 + 2]).collect();
    let mut file = File::create("particle_orbit.csv")?;
    writeln!(file, "x,y,z")?;
    for ((x, y), z) in x0.iter().zip(y0.iter()).zip(z0.iter()) {
        writeln!(file, "{},{},{}", x, y, z)?;
    }

    // rk_err csv
    let mut erf = File::create("rk_error.csv")?;
    writeln!(erf, "step,particle,rk_err")?;
    for s in 0..STEPS {
        let row = &rk_err[s * N..(s + 1) * N];
        for (i, &e) in row.iter().enumerate() {
            writeln!(erf, "{},{},{}", s, i, e)?;
        }
    }

    let radii: Array1<f32> = Array1::linspace(0.0, 4.0, 200);
    let mut file_s = File::create("sphericalcutoff.csv")?;
    let mut file_mn = File::create("miyamoto_nagai.csv")?;
    let mut file_nfw = File::create("navarro_frenk_white.csv")?;
    writeln!(file_s, "x,ax")?;
    writeln!(file_mn, "x,ax")?;
    writeln!(file_nfw, "x,ax")?;

    let y = 0.0;
    let z = 0.0;
    let alpha = 1.8;
    let r1 = 1.0;
    let c2 = powf(1.9 / 8., 2.);
    let a_mn = 3. / 8.;
    let b_mn = 0.28 / 8.;
    let a_nfw = 16. / 8.;

    for x in radii.iter().copied() {
        let (ax_s, _, _) = sphericalcutoff_force(x, y, z, AMP_S, alpha, r1, c2);
        let (ax_mn, _, _) = miyamoto_nagai_force(x, y, z, AMP_MN, a_mn, b_mn);
        let (ax_nfw, _, _) = navarro_frenk_white_force(x, y, z, AMP_NFW, a_nfw);

        writeln!(file_s, "{},{}", x, ax_s)?;
        writeln!(file_mn, "{},{}", x, ax_mn)?;
        writeln!(file_nfw, "{},{}", x, ax_nfw)?;
    }

    Ok(())
}

// TEMP

const PI: f32 = 3.14159265358979323846264338327950288;

const AMP_S: f32 = 67168.3897826272;
const AMP_MN: f32 = 0.7574802019;
const AMP_NFW: f32 = 4.852230533528;

#[inline(always)]
fn gamma_p(a: f32, x: f32) -> f32 {
    println!("{a} {x}");
    let af: f64 = a as f64;
    let xf: f64 = x as f64;
    let mut sum: f64 = 1.;
    let mut term: f64 = 1.;
    let mut denom: f64 = 1.;
    for n in 1..40 {
        denom = af + (n as f64);
        term *= xf / denom;
        sum += term;
        println!("{denom} {term} {sum}");
    }
    sum as f32
}
#[inline(always)]
fn sphericalcutoff_force(
    x: f32,
    y: f32,
    z: f32,
    amp: f32,
    alpha: f32,
    r1: f32,
    c2: f32,
) -> (f32, f32, f32) {
    let r2 = powf(x, 2.) + powf(y, 2.) + powf(z, 2.);
    let r = sqrtf(r2);
    let ar = 2. * PI * 1.48919 * gamma_p(0.6, r2 / c2);
    let ax = ar * (x / r);
    let ay = ar * (y / r);
    let az = ar * (z / r);
    (ax, ay, az)
}
fn navarro_frenk_white_force(x: f32, y: f32, z: f32, amp: f32, a: f32) -> (f32, f32, f32) {
    let r2 = powf(x, 2.) + powf(y, 2.) + powf(z, 2.);
    let r = sqrtf(r2);
    let ar = amp * (logf(1. + r / a) - r / (a + r)) / r2;
    let ax = ar * (x / r);
    let ay = ar * (y / r);
    let az = ar * (z / r);
    (ax, ay, az)
}
fn miyamoto_nagai_force(x: f32, y: f32, z: f32, amp: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let R2 = powf(x, 2.) + powf(y, 2.);
    let R = sqrtf(R2);
    let z2 = powf(z, 2.);
    let b2 = powf(b, 2.);
    let sqrtz2b2 = sqrtf(z2 + b2);
    let pyth = powf(a + sqrtz2b2, 2.);
    let denom = powf(pyth + R2, 3. / 2.);
    let aR = -amp * (R / denom);
    let ax = aR * (x / R);
    let ay = aR * (y / R);
    let az = -z * (a + sqrtz2b2) / (sqrtz2b2 * denom);
    (ax, ay, az)
}
