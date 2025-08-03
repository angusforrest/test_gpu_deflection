use libm::expf;
use libm::powf;
use libm::sqrtf;

const PI: f32 = 3.14159265358979323846264338327950288;

#[inline(always)]
fn sphericalcutoff_force(r: f32, amp: f32, alpha: f32, r1: f32, c2: f32) -> f32 {
    let r2 = powf(r, 2);
    -amp * (powf(r1 / r, alpha) * (alpha * c2 + 2 * r2) * expf(-r2 / c2)) / (r * c2)
}
fn navarro_frenk_white_force(r: f32, amp: f32, a: f32) -> f32 {
    let ar3 = powf((a + r), 3);
    let r2 = powf(r, 2);
    -amp * (1 / (4 * PI)) * ((a + 3 * r) / (r2 * ar3))
}
fn miyamoto_nagai_force(R: f32, z: f32, amp: f32, a: f32, b: f32) -> (f32, f32) {
    let z2 = powf(z, 2);
    let b2 = powf(b, 2);
    let sqrtz2b2 = sqrtf(z2 + b2);
    let pyth = powf(a + sqrtz2b2, 2);
    let R2 = powf(R, 2);
    let denom = powf(pyth + R2, 3 / 2);
    amp * (R / denom, z * (a + sqrtz2b2) / (sqrtz2b2 * denom))
}
