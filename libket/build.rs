// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

fn main() {
    if std::env::var("CARGO_FEATURE_C_API").is_ok() {
        build_info_build::build_script();
    }
    #[cfg(feature = "qasm")]
    lalrpop::process_root().unwrap();
}
