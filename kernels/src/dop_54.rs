use cuda_std::{kernel, thread};
use libm::{powf, sqrtf};
use crate::butcher::{ButcherTableau, DormandPrince54 as Coeffs};

const M_S: f32 = 1.0;
const G: f32 = 39.5;

#[inline(always)]
fn compute_acceleration(t: f32, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let r2 = x * x + y * y + z * z;
    let r32 = powf(r2, 1.5);
    let a = -G * M_S / r32;
    (a * x, a * y, a * z)
}

#[inline(always)]
pub fn rk_norm(
    x: f32, x_new: f32, err_x: f32,
    y: f32, y_new: f32, err_y: f32,
    z: f32, z_new: f32, err_z: f32,
    vx: f32, vx_new: f32, err_vx: f32,
    vy: f32, vy_new: f32, err_vy: f32,
    vz: f32, vz_new: f32, err_vz: f32,
    atol: f32, rtol: f32
) -> f32 {
    let sc_x  = atol + rtol * f32::max(x.abs(), x_new.abs());
    let sc_y  = atol + rtol * f32::max(y.abs(), y_new.abs());
    let sc_z  = atol + rtol * f32::max(z.abs(), z_new.abs());
    let sc_vx = atol + rtol * f32::max(vx.abs(), vx_new.abs());
    let sc_vy = atol + rtol * f32::max(vy.abs(), vy_new.abs());
    let sc_vz = atol + rtol * f32::max(vz.abs(), vz_new.abs());

    // might need to guard against div by 0
    let sum = powf(err_x / sc_x, 2.0)
            + powf(err_y / sc_y, 2.0)
            + powf(err_z / sc_z, 2.0)
            + powf(err_vx / sc_vx, 2.0)
            + powf(err_vy / sc_vy, 2.0)
            + powf(err_vz / sc_vz, 2.0);

    sqrtf(sum / 6.0)
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
    atol: f32,
    rtol: f32,
    error_out: *mut f32,
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

    // now we've completed 5th order, we do 4th:

    let mut x_hat = x;
    let mut y_hat = y;
    let mut z_hat = z;
    let mut vx_hat = vx;
    let mut vy_hat = vy;
    let mut vz_hat = vz;

    for i in 0..Coeffs::STAGES {
        let b_hat = Coeffs::B_HAT[i];
        x_hat += dt * b_hat * rk_x[i];
        y_hat += dt * b_hat * rk_y[i];
        z_hat += dt * b_hat * rk_z[i];
        vx_hat += dt * b_hat * rk_vx[i];
        vy_hat += dt * b_hat * rk_vy[i];
        vz_hat += dt * b_hat * rk_vz[i];
    }

    // and compute truncation error est per component
    let err_x = x_new - x_hat;
    let err_y = y_new - y_hat;
    let err_z = z_new - z_hat;
    let err_vx = vx_new - vx_hat;
    let err_vy = vy_new - vy_hat;
    let err_vz = vz_new - vz_hat;

    let rk_err = rk_norm(
        x, x_new, err_x,
        y, y_new, err_y,
        z, z_new, err_z,
        vx, vx_new, err_vx,
        vy, vy_new, err_vy,
        vz, vz_new, err_vz,
        atol, rtol
    );

    if !error_out.is_null() {
        *error_out.add(current_step * n + tid) = rk_err;
    }

    // later:
    // if rk_err <= 1.0 {
    // accept step
    // } else {
    // reject and reduce dt
    // }

    *state_out.add(curr_offset + 0) = x_new;
    *state_out.add(curr_offset + 1) = y_new;
    *state_out.add(curr_offset + 2) = z_new;
    *state_out.add(curr_offset + 3) = vx_new;
    *state_out.add(curr_offset + 4) = vy_new;
    *state_out.add(curr_offset + 5) = vz_new;
}