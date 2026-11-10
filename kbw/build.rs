// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

fn main() {
    build_info_build::build_script();

    let mut accelerators = Vec::new();

    if std::env::var("CARGO_FEATURE_GPU").is_ok() {
        accelerators.push("GPU");
    }

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        accelerators.push("CUDA");
    }

    if accelerators.is_empty() {
        println!("cargo:rustc-env=KBW_ACCELERATORS=", );
    } else {
        println!(
            "cargo:rustc-env=KBW_ACCELERATORS=[{}]",
            accelerators.join(", ")
        );
    }
}
