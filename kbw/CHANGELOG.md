<!--
SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>

SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## 0.4.0

- Update Libket to version 0.6.0
- Moved expectation value calculation and sampling to the `QuantumExecution` trait.

## 0.3.0

- Update Libket to version 0.5.0.
- Added DenseV2.

## 0.2.1

- Update Libket to version 0.4.1.
- Added option to enable quantum gate decomposition.

## 0.2.0

- Updated Libket to version 0.4.0, adding the `batch` and `live` execution modes.

## 0.1.7

- Fixed for Libket version 0.3.1.

## 0.1.6

- Updated Libket to version 0.3.0.

## 0.1.5

- Fixed plugin `pown` for the Dense simulator.
- Updated Libket to version 0.2.3.

## 0.1.4

- Added the ability to select dump type using environment variables:
  - `KBW_DUMP_TYPE`: Choose from `vector`, `probability`, or `shots`.
  - `KBW_SHOTS`: Specify the number of shots.
- Added support for seeding the RNG with the `KBW_SEED` environment variable.

## 0.1.3

- Updated Libket to version 0.2.2.

## 0.1.2

- Updated Libket to version 0.2.0.

## 0.1.1

- Fixed an issue with measure probability.

## 0.1.0

- Ported KBW from C++ to Rust.
- Added Dense simulator.
