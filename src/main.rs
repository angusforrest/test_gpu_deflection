use cust::prelude::*;
use libm::{pow, sqrt};
use rand::Rng;
use std::error::Error;
use std::f64::consts::TAU;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::f64::consts::PI;
use statrs::function::gamma::{gamma, gamma_lr};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

const N: usize = 1024;
const STEPS_CAP: usize = 200000;

const T_END: f64 = 40.0;
const DT0: f64 = 0.002; // initial dt per particle
const ATOL: f64 = 1.0e-8;
const RTOL: f64 = 1.0e-8;

// FIXME: double check in MPA impl
const FAC_MIN: f64 = 0.33;
const FAC_MAX: f64 = 6.0;
const SAFETY: f64 = 0.9;
const DT_MIN: f64 = 1.0e-12;
const DT_MAX: f64 = 0.25;

const M_S: f64 = 1.0;
const G: f64 = 1.0;

const BLOCK_SIZE: u32 = 128;

fn grid_size(n: usize, block: u32) -> (u32, u32) {
    let blocks = ((n as u32) + block - 1) / block;
    (blocks, block)
}

const N_AR: usize = 10000;
const R_MIN: f64 = 1e-4;
const R_MAX: f64 = 100.0;

pub fn mass(r2: f64, alpha: f64, rc: f64) -> f64 {
    2.0 * PI * pow(rc, 3.0 - alpha)
        * gamma(1.5 - 0.5 * alpha)
        * gamma_lr(1.5 - 0.5 * alpha, r2 / (rc * rc))
}

fn build_sphericalcutoff_force_table(
    amp: f64,
    alpha: f64,
    r1: f64,
    rc: f64,
) -> (Vec<f64>, f64, f64) {
    let mut table = Vec::with_capacity(N_AR);
    let dr = (R_MAX - R_MIN) / (N_AR as f64 - 1.0);
    for i in 0..N_AR {
        let r = R_MIN + i as f64 * dr;
        let r2 = r * r;
        let m = amp * pow(r1, alpha) * mass(r2, alpha, rc);
        let ar = -m / r2;
        table.push(ar);
    }
    (table, R_MIN, dr)
}

fn main() -> Result<(), Box<dyn Error>> {
    // CUDA setup
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    let kernel = module.get_function("dopr54_adaptive")?;

    // keeping these separate for now but vec7 would be ideal
    let mut state_out = vec![0.0f64; STEPS_CAP * N * 6];
    let mut time_out = vec![0.0f64; STEPS_CAP * N];

    // conds for circular orbits
    let mut rng = rand::rng();
    for i in 0..N {
        let r = rng.gen_range(1.0..5.0);
        let theta = rng.gen_range(0.0..TAU);
        // let x = r * theta.cos();
        // let y = r * theta.sin();
        // let z = 0.0;
        // let v = (G * M_S / r).sqrt();
        // let vx = -v * theta.sin();
        // let vy =  v * theta.cos();
        // let vz = 0.0;
        let x = 1.0;
        let y = 0.0;
        let z = 0.0;
        let vx = 0.0;
        let vy = 1.0;
        let vz = 0.0;

        let offset0 = (0 * N + i) * 6;
        state_out[offset0 + 0] = x;
        state_out[offset0 + 1] = y;
        state_out[offset0 + 2] = z;
        state_out[offset0 + 3] = vx;
        state_out[offset0 + 4] = vy;
        state_out[offset0 + 5] = vz;
    }

    // per-particle scalars
    let mut t_host    = vec![0.0f64; N];
    let mut dt_host   = vec![DT0; N];
    let mut w_host    = vec![0u32; N];
    let mut done_host = vec![0u8; N];
    let mut err_host  = vec![0.0f64; N];

    // device buffers
    let mut dev_state_out = DeviceBuffer::<f64>::from_slice(&state_out)?;
    let mut dev_t         = DeviceBuffer::<f64>::from_slice(&t_host)?;
    let mut dev_dt        = DeviceBuffer::<f64>::from_slice(&dt_host)?;
    let mut dev_w         = DeviceBuffer::<u32>::from_slice(&w_host)?;
    let mut dev_done      = DeviceBuffer::<u8>::from_slice(&done_host)?;
    let mut dev_err       = DeviceBuffer::<f64>::from_slice(&err_host)?;
    let mut dev_time_out   = DeviceBuffer::<f64>::zeroed(STEPS_CAP * N)?;

    // we launch until all threads are done (or we hit capacity)
    let (grid, block) = grid_size(N, BLOCK_SIZE);
    let mut iter = 0usize;
    let max_outer_iters = 200_000usize; // guard against infinite loops; can raise

    // create table
    let bulge_amp = 0.029994597188218296;
    let bulge_alpha = 1.8;
    let bulge_r1 = 1.0;
    let bulge_rc = 1.9 / 8.0;

    let (ar_table_host, r_min, dr) = build_sphericalcutoff_force_table(bulge_amp, bulge_alpha, bulge_r1, bulge_rc);
    let dev_ar_table = DeviceBuffer::from_slice(&ar_table_host)?;

    loop {
        unsafe {
            launch!(
                kernel<<<grid, block, 0, stream>>>(
                    dev_state_out.as_device_ptr(),
                    dev_time_out.as_device_ptr(),
                    N,
                    STEPS_CAP,
                    dev_t.as_device_ptr(),
                    dev_dt.as_device_ptr(),
                    dev_w.as_device_ptr(),
                    dev_done.as_device_ptr(),
                    T_END,
                    ATOL,
                    RTOL,
                    FAC_MIN,
                    FAC_MAX,
                    SAFETY,
                    DT_MIN,
                    DT_MAX,
                    dev_err.as_device_ptr(),
                    dev_ar_table.as_device_ptr(),
                    r_min,
                    dr,
                    N_AR as u32
                )
            )?;
        }

        stream.synchronize()?;
        iter += 1;

        // copy back "done" each iteration. Maybe we collapse this on device or do it less frequently?
        dev_done.copy_to(&mut done_host)?;

        // stop if all done
        let any_active = done_host.iter().any(|&d| d == 0);
        if !any_active {
            break;
        }

        // stop if cap reached
        if iter >= max_outer_iters {
            eprintln!(
                "Reached max iterations ({}) before all particles finished; stopping.",
                max_outer_iters
            );
            break;
        }
    }

    dev_state_out.copy_to(&mut state_out)?;
    dev_time_out.copy_to(&mut time_out)?;
    dev_t.copy_to(&mut t_host)?;
    dev_dt.copy_to(&mut dt_host)?;
    dev_w.copy_to(&mut w_host)?;
    dev_err.copy_to(&mut err_host)?;

    let w0 = w_host[0] as usize;
    if w0 >= STEPS_CAP - 1 {
        eprintln!("WARNING: particle 0 hit STEPS_CAP-1; last step may have been overwritten multiple times.");
    }

    println!("Integration finished after {} kernel launches.", iter);

    // a few diagnostics
    let final_timestep = w_host[0] as usize;
    let final_off = (final_timestep * N + 0) * 6;
    println!("Particle 0 finished at t = {:.12}, timestep = {}", t_host[0], final_timestep);
    println!(
        "Final state (x,y,z) = ({:.12}, {:.12}, {:.12})",
        state_out[final_off + 0],
        state_out[final_off + 1],
        state_out[final_off + 2],
    );
    println!(
        "Last normalized RK error (particle 0) = {:.3e}",
        err_host[0]
    );


    let w0 = w_host[0] as usize;
    let traj_len = (w0 + 1).min(STEPS_CAP);
    let mut file = File::create("particle0_steps_with_time.csv")?;
    writeln!(file, "timestep,time,x,y,z,vx,vy,vz")?;

    for s in 0..traj_len {
        let off = (s * N + 0) * 6;
        let time = time_out[s * N + 0];
        let x  = state_out[off + 0];
        let y  = state_out[off + 1];
        let z  = state_out[off + 2];
        let vx = state_out[off + 3];
        let vy = state_out[off + 4];
        let vz = state_out[off + 5];
        writeln!(
            file,
            "{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            s, time, x, y, z, vx, vy, vz
        )?;
    }
    println!("Wrote {} rows to particle0_steps_with_time.csv", traj_len);

    let accepts0 = w_host[0] as usize;  // steps advanced for particle 0
    let accept_rate = if iter > 0 { 100.0 * (accepts0 as f64) / (iter as f64) } else { 0.0 };
    println!("Kernel launches (outer iterations): {}", iter);
    println!("Particle 0 accepted steps: {}", accepts0);
    println!("Approx overall accept rate (p0): {:.2}%", accept_rate);
    println!("Final t[0] = {:.12} (target T_END = {:.12})", t_host[0], T_END);

    // TODO: How to compute total energy when we have diverging times per particle?

    Ok(())
}
