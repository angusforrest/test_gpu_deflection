use cuda_std::{kernel, thread};
use libm::powf;

const M_S: f32 = 1.0;
const G: f32 = 39.5;

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

#[inline(always)]
fn compute_acceleration(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    // newtonian grav
    let r2 = x * x + y * y + z * z;
    let r32 = powf(r2, 1.5);
    let a = -G * M_S / r32;
    (a * x, a * y, a * z)
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn euler_step(
    state_out: *mut f32,
    n: usize,
    steps: usize,
    current_step: usize,
    dt: f32,
) {
    let tid = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    if tid >= n {
        return;
    }

    let prev_offset = ((current_step - 1) * n + tid) * 6;
    let curr_offset = (current_step * n + tid) * 6;

    let x = *state_out.add(prev_offset + 0);
    let y = *state_out.add(prev_offset + 1);
    let z = *state_out.add(prev_offset + 2);
    let vx_old = *state_out.add(prev_offset + 3);
    let vy_old = *state_out.add(prev_offset + 4);
    let vz_old = *state_out.add(prev_offset + 5);

    let (dx, dy, dz) = lorenz_derivatives(x, y, z);

    // normal euler step
    let x_new = x + dx * dt;
    let y_new = y + dy * dt;
    let z_new = z + dz * dt;

    *state_out.add(curr_offset + 0) = x_new;
    *state_out.add(curr_offset + 1) = y_new;
    *state_out.add(curr_offset + 2) = z_new;
    *state_out.add(curr_offset + 3) = dx;
    *state_out.add(curr_offset + 4) = dy;
    *state_out.add(curr_offset + 5) = dz;
}