# opencl3

[![crates.io](https://img.shields.io/crates/v/opencl3.svg)](https://crates.io/crates/opencl3)
[![docs.io](https://docs.rs/opencl3/badge.svg)](https://docs.rs/opencl3/)
[![OpenCL 3.0](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/registry/OpenCL/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://github.com/kenba/opencl3/workflows/Rust/badge.svg)](https://github.com/kenba/opencl3/actions)

A Rust implementation of the Khronos [OpenCL](https://www.khronos.org/registry/OpenCL/) API.

# Description

This crate provides a relatively simple, object based model of the OpenCL 3.0
[API](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html).  
It is built upon the [cl3](https://crates.io/crates/cl3) crate, which
provides a functional interface to the OpenCL API.  

**OpenCL** (Open Computing Language) is framework for general purpose
parallel programming across heterogeneous devices including: CPUs, GPUs,
DSPs, FPGAs and other processors or hardware accelerators.

It is often considered as an open-source alternative to Nvidia's proprietary
Compute Unified Device Architecture [CUDA](https://developer.nvidia.com/cuda-zone)
for performing General-purpose computing on GPUs, see
[GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units).

The [OpenCL Specification](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_the_opencl_architecture)
has evolved over time and not all device vendors support all OpenCL features.

[OpenCL 3.0](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html)
is a unified specification that adds little new functionality to previous OpenCL versions.  
It specifies that all **OpenCL 1.2** features are **mandatory**, while all
OpenCL 2.x and 3.0 features are now optional.

OpenCL 2.x and 3.0 optional features include:
* Shared Virtual Memory (SVM),
* nested parallelism,
* pipes
* atomics
* and a generic address space,

## Design

See the crate [documentation](https://docs.rs/opencl3/).

## Use

Ensure that an OpenCL Installable Client Driver (ICD) and the appropriate OpenCL
hardware driver(s) are installed, see [cl3](https://crates.io/crates/cl3).

`opencl3` supports OpenCL 1.2 and 2.0 ICD loaders by default. If you have an
OpenCL 2.0 ICD loader then add the following to your project's `Cargo.toml`:

```toml
[dependencies]
opencl3 = "0.1"
```

If your OpenCL ICD loader supports higher versions of OpenCL then add the
appropriate features to opencl3, e.g. for an OpenCL 2.2 ICD loader add the
following to your project's `Cargo.toml` instead:

```toml
[dependencies.opencl3]
version = "0.1"
features = ["CL_VERSION_2_1", "CL_VERSION_2_2"]
```

See [OpenCL Description](https://github.com/kenba/opencl3/blob/main/docs/opencl_description.md)

## License

Licensed under the Apache License, Version 2.0, as per Khronos Group OpenCL.  
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

OpenCL and the OpenCL logo are trademarks of Apple Inc. used under license by Khronos.
