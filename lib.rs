#![cfg_attr(
    target_os = "cuda",
    no_std,
    register_attr(nvvm_internal)
)]

use cuda_std::*;
