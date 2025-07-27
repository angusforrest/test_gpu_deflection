use cuda_std::{kernel, thread};
use libm::powf;

mod butcher;
use butcher::{ButcherTableau, DormandPrince54 as Coeffs};

const M_S: f32 = 1.0;
const G: f32 = 39.5;

#[inline(always)]
fn compute_acceleration(t: f32, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let r2 = x * x + y * y + z * z;
    let r32 = powf(r2, 1.5);
    let a = -G * M_S / r32;
    (a * x, a * y, a * z)
}

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn dopr54_step(
    state_out: *mut f32,
    n: usize,
    steps: usize,
    current_step: usize,
    dt: f32,
    t0: f32,
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
    let vx = *state_out.add(prev_offset + 3);
    let vy = *state_out.add(prev_offset + 4);
    let vz = *state_out.add(prev_offset + 5);

    // intermediate rk stage values
    let mut rk_x = [0.0f32; Coeffs::STAGES];
    let mut rk_y = [0.0f32; Coeffs::STAGES];
    let mut rk_z = [0.0f32; Coeffs::STAGES];
    let mut rk_vx = [0.0f32; Coeffs::STAGES];
    let mut rk_vy = [0.0f32; Coeffs::STAGES];
    let mut rk_vz = [0.0f32; Coeffs::STAGES];

    // hopefully these will be unrolled...
    for i in 0..Coeffs::STAGES {
        let mut xi = x;
        let mut yi = y;
        let mut zi = z;
        let mut vxi = vx;
        let mut vyi = vy;
        let mut vzi = vz;

        // contributions from prev stages using tableau A
        for j in 0..i {
            let aij = Coeffs::A[i][j];
            xi += dt * aij * rk_x[j];
            yi += dt * aij * rk_y[j];
            zi += dt * aij * rk_z[j];
            vxi += dt * aij * rk_vx[j];
            vyi += dt * aij * rk_vy[j];
            vzi += dt * aij * rk_vz[j];
        }

        // tableau C can be unused here, if acceleration is time-independent
        let t_stage = t0 + dt * Coeffs::C[i];

        let (axi, ayi, azi) = compute_acceleration(t_stage, xi, yi, zi);

        rk_x[i] = vxi;
        rk_y[i] = vyi;
        rk_z[i] = vzi;
        rk_vx[i] = axi;
        rk_vy[i] = ayi;
        rk_vz[i] = azi;
    }

    // output with initial values
    let mut x_new = x;
    let mut y_new = y;
    let mut z_new = z;
    let mut vx_new = vx;
    let mut vy_new = vy;
    let mut vz_new = vz;

    // combine stages using tableau B to compute final state
    for i in 0..Coeffs::STAGES {
        let b = Coeffs::B[i];
        x_new += dt * b * rk_x[i];
        y_new += dt * b * rk_y[i];
        z_new += dt * b * rk_z[i];
        vx_new += dt * b * rk_vx[i];
        vy_new += dt * b * rk_vy[i];
        vz_new += dt * b * rk_vz[i];
    }

    // note: because we aren't doing adaptive stepping this is DOP5 not DOP54

    *state_out.add(curr_offset + 0) = x_new;
    *state_out.add(curr_offset + 1) = y_new;
    *state_out.add(curr_offset + 2) = z_new;
    *state_out.add(curr_offset + 3) = vx_new;
    *state_out.add(curr_offset + 4) = vy_new;
    *state_out.add(curr_offset + 5) = vz_new;
}