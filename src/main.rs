use cust::prelude::*;
use std::error::Error;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use rand::Rng;
use std::f32::consts::TAU;

const N: usize = 1024;
const STEPS: usize = 1000;
const DT: f32 = 0.01;
const M_S: f32 = 1.0;
const G: f32 = 39.5;
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    // CUDA setup
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    let mut state_out = vec![0.0f32; N * STEPS * 6];

    let mut rng = rand::thread_rng();
    // initial cond
    for i in 0..N {
        let x = rng.gen_range(-10.0..10.0);
        let y = rng.gen_range(-10.0..10.0);
        let z = rng.gen_range(10.0..30.0);
        let vx = 0.0;
        let vy = 0.0;
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

    let kernel_euler_step = module.get_function("euler_step")?;

    for step in 1..STEPS {
        unsafe {
            launch!(
                kernel_euler_step<<<(N as u32 + 127) / 128, 128, 0, stream>>>(
                    dev_state_out.as_device_ptr(),
                    N,
                    STEPS,
                    step,
                    DT
                )
            )?;
        }
    }

    stream.synchronize()?;

    dev_state_out.copy_to(&mut state_out)?;

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


    Ok(())
}
