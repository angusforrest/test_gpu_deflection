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
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    // CUDA setup
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    let total_size = N * STEPS;

    let mut x_out = vec![0.0f32; total_size];
    let mut y_out = vec![0.0f32; total_size];
    let mut vx_out = vec![0.1f32; total_size];
    let mut vy_out = vec![0.0f32; total_size];

    let mut rng = rand::thread_rng();
    // initial cond
    for i in 0..N {

        let r = rng.gen_range(1.0..5.0);
        // [0, 2pi)
        let theta = rng.gen_range(0.0..TAU);

        let x = r * theta.cos();
        let y = r * theta.sin();
        x_out[i] = x;
        y_out[i] = y;

        // v = sqrt(GM / r)
        let v = (39.5 / r).sqrt();
        let vx = -v * theta.sin();
        let vy =  v * theta.cos();
        vx_out[i] = vx;
        vy_out[i] = vy;
    }

    let mut dev_x_out = x_out.as_slice().as_dbuf()?;
    let mut dev_y_out = y_out.as_slice().as_dbuf()?;
    let mut dev_vx_out = vx_out.as_slice().as_dbuf()?;
    let mut dev_vy_out = vy_out.as_slice().as_dbuf()?;

    let func_vx = module.get_function("euler_integration_vx")?;
    let func_x = module.get_function("euler_integration_x")?;

    for step in 1..STEPS {
        unsafe {
            launch!(
                func_vx<<<(N as u32 + 127) / 128, 128, 0, stream>>>(
                    dev_x_out.as_device_ptr(),
                    dev_y_out.as_device_ptr(),
                    dev_vx_out.as_device_ptr(),
                    N,
                    STEPS,
                    step,
                    DT
                )
            )?;

            launch!(
                func_vx<<<(N as u32 + 127) / 128, 128, 0, stream>>>(
                    dev_y_out.as_device_ptr(),
                    dev_x_out.as_device_ptr(),
                    dev_vy_out.as_device_ptr(),
                    N,
                    STEPS,
                    step,
                    DT
                )
            )?;

            launch!(
                func_x<<<(N as u32 + 127) / 128, 128, 0, stream>>>(
                    dev_x_out.as_device_ptr(),
                    dev_vx_out.as_device_ptr(),
                    N,
                    STEPS,
                    step,
                    DT
                )
            )?;

            launch!(
                func_x<<<(N as u32 + 127) / 128, 128, 0, stream>>>(
                    dev_y_out.as_device_ptr(),
                    dev_vy_out.as_device_ptr(),
                    N,
                    STEPS,
                    step,
                    DT
                )
            )?;
        }
    }

    stream.synchronize()?;

    dev_x_out.copy_to(&mut x_out)?;
    dev_y_out.copy_to(&mut y_out)?;
    dev_vx_out.copy_to(&mut vx_out)?;
    dev_vy_out.copy_to(&mut vy_out)?;

    println!("Final x[0]: {}", x_out[(STEPS - 1) * N]);

    let x0: Vec<f32> = (0..STEPS).map(|s| x_out[s * N + 0]).collect();
    let y0: Vec<f32> = (0..STEPS).map(|s| y_out[s * N + 0]).collect();
    let mut file = File::create("particle_orbit.csv")?;
    writeln!(file, "x,y")?;
    for (x, y) in x0.iter().zip(y0.iter()) {
        writeln!(file, "{},{}", x, y)?;
    }


    Ok(())
}
