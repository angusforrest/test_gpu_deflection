use std::fs::File;
use std::io::{BufWriter, Write};
use std::f64::consts::PI;
use statrs::function::gamma::{gamma, gamma_lr};
use libm::pow;

pub fn mass(r2: f64, alpha: f64, rc: f64) -> f64 {
    2.0 * PI * pow(rc, 3.0 - alpha)
        * gamma(1.5 - 0.5 * alpha)
        * gamma_lr(1.5 - 0.5 * alpha, r2 / (rc * rc))
}

pub fn precompute_sphericalcutoff_force(
    filename: &str,
    amp: f64,
    alpha: f64,
    r1: f64,
    rc: f64,
    r_min: f64,
    r_max: f64,
    n_points: usize,
) -> std::io::Result<()> {
    let dr = (r_max - r_min) / (n_points as f64 - 1.0);
    let mut writer = BufWriter::new(File::create(filename)?);
    writeln!(writer, "r,ar")?;

    for i in 0..n_points {
        let r = r_min + i as f64 * dr;
        let r2 = r * r;
        let m = amp * pow(r1, alpha) * mass(r2, alpha, rc);
        let ar = -m / r2;
        writeln!(writer, "{},{}", r, ar)?;
    }

    Ok(())
}
