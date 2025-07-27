use cuda_std::{kernel, thread};
use libm::powf;

const M_S: f32 = 1.0;
const G: f32 = 39.5;

#[inline(always)]
fn differential_system(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    return (10(y - x), -x * z + 28x - y, x * y - 8 * z / 3);
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn euler_update_velocity_component(
    x_out: *const f32,
    y_out: *const f32,
    vx_out: *mut f32,
    n: usize,
    steps: usize,
    current_step: usize,
    dt: f32,
) {
    let tid = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    let step_offset = current_step * n;
    let prev_offset = (current_step - 1) * n;

    if tid < n {
        let x = *x_out.add(prev_offset + tid);
        let y = *y_out.add(prev_offset + tid);
        let prev_vx = *vx_out.add(prev_offset + tid);
        *vx_out.add(step_offset + tid) = prev_vx - potential_thingy(x, y, z) * dt;
    }
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn euler_update_position_component(
    x_out: *mut f32,
    vx_out: *const f32,
    n: usize,
    steps: usize,
    current_step: usize,
    dt: f32,
) {
    let tid = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    let step_offset = current_step * n;
    let prev_offset = (current_step - 1) * n;

    if tid < n {
        let prev_x = *x_out.add(prev_offset + tid);
        let vx = *vx_out.add(step_offset + tid);
        *x_out.add(step_offset + tid) = prev_x + vx * dt;
    }
}
