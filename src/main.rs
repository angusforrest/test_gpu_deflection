use cust::prelude::*;
use std::error::Error;
use std::ffi::CString;

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

    // step 0 positions
    for i in 0..N {
        x_out[i] = 1.0 + i as f32 * 0.001;
        y_out[i] = 0.0;
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

    Ok(())
}
