<!--
SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>

SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## 0.6.0

- Added support for execution with gradient calculation and gradients via parameter-shift.
- Updated the decomposition algorithms according to https://arxiv.org/abs/2504.20291.
- Added calculation of expected values via classical shadows and direct measurement with samples.
- Added the `LiveCExecution` C struct to enable implementing live execution targets in other programming languages.
- Added an auxiliary qubit allocation interface for programmers.

## 0.5.1

- Fixed a bug related to inverse gate application.

## 0.5.0

- Added linear-time gate decomposition using auxiliary qubit, as presented in https://arxiv.org/abs/2406.05581.
- Implemented qubit mapping using Dynamic Look-Ahead, as presented in https://doi.org/10.1109/TCAD.2020.2970594.
- Added the necessaries for represent QPU coupling graph and gate set. 

## 0.4.1

- Implemented multi-controlled single qubit gate decomposition, using the algorithm presented in https://doi.org/10.1103/PhysRevA.106.042602.
- Introduced preliminary support for ZX calculus-based circuit optimization.

## 0.4.0

- Introduced the `live` execution mode, enabling iterative execution of quantum operations. The previous behavior is now referred to as `batch` execution mode.
- Moved measurement results and dumps to be stored directly in the `Process` structure instead of being referenced by a shared pointer.
- Reverted the `dump` functionality to its original form, now exclusively storing information in the `vector` type.
- Added expected value calculations to the process capabilities.
- Modified the process to handle the quantum execution call, requiring the quantum executor to be passed in the constructor.
- Removed classical operations and control flows from the process.

## 0.3.1

- Added documentation for the process, objects, and gates modules.
- Modified the gate decomposition process to exclusively use the CNOT gate as the multi-qubit gate.
- Added examples.

## 0.3.0

- Added support for decomposition of multi-controlled quantum gates.
- Added documentation for the C API.

## 0.2.3

- Introduced the `plugin` function in the `gates` module.

## 0.2.2

- Refactored the implementation of `Quant` and gate functions for improved usability.

## 0.2.1

- Included the `Quant` type and gate functions to enhance the functionality.

## 0.2.0

- Expanded the available dump types with `"shots"` and `"probability"`, in addition to `"vector"`.

## 0.1.1

- Fixed a bug related to inverse gate application.

## 0.1.0

- Ported the Libket library from C++ to Rust for increased usability and performance.
