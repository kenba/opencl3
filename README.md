# opencl3

[![crates.io](https://img.shields.io/crates/v/opencl3.svg)](https://crates.io/crates/opencl3)
[![docs.io](https://docs.rs/opencl3/badge.svg)](https://docs.rs/opencl3/)
[![OpenCL 3.0](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/registry/OpenCL/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://github.com/kenba/opencl3/workflows/Rust/badge.svg)](https://github.com/kenba/opencl3/actions)

A Rust implementation of the Khronos [OpenCL](https://www.khronos.org/registry/OpenCL/) API.

## Description

A relatively simple, object based model of the OpenCL 3.0
[API](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html).  
It is built upon the [cl3](https://crates.io/crates/cl3) crate, which
provides a functional interface to the OpenCL [C API](https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h).  

[OpenCL](https://www.khronos.org/opencl/) (Open Computing Language) is framework for general purpose parallel programming across heterogeneous devices including: CPUs, GPUs, DSPs, FPGAs and other processors or hardware accelerators. It is often considered as an open-source alternative to Nvidia's proprietary
Compute Unified Device Architecture [CUDA](https://developer.nvidia.com/cuda-zone)
for performing General-purpose computing on GPUs, see
[GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units).

[OpenCL 3.0](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html)
is a unified specification that adds little new functionality to previous OpenCL versions.  
It specifies that all **OpenCL 1.2** features are **mandatory**, while all
OpenCL 2.x and 3.0 features are now optional.

### Features

This library has:

* A simple API, enabling most OpenCL objects to be created with a single function call.
* Automatic OpenCL resource management using the [Drop trait](https://doc.rust-lang.org/book/ch15-03-drop.html) to implement [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization).
* Support for [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) OpenCL control flow execution using event wait lists.
* Support for Shared Virtual Memory (SVM) with an [SvmVec](src/svm.rs) object that can be serialized by [serde](https://crates.io/crates/serde).
* Support for OpenCL extensions, see [OpenCL Extensions](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html).
* Support for multithreading with [Send and Sync](https://doc.rust-lang.org/nomicon/send-and-sync.html) traits.

## Design

The library is object based with most OpenCL objects represented by rust structs.
For example, an OpenCL `cl_device_id` is represented by [Device](src/device.rs) with methods to get information about the device instead of calling `clGetDeviceInfo` with the relevant `cl_device_info` value.  

The struct methods are simpler to use than their equivalent standalone functions in [cl3](https://github.com/kenba/cl3) because they convert the `InfoType` enum into the correct underlying type returned by the `clGetDeviceInfo` call for the `cl_device_info` value.

Nearly all the structs implement the `Drop` trait to release their corresponding
OpenCL objects. The exceptions are `Platform` and `Device` which don't need to be released. See the crate [documentation](https://docs.rs/opencl3/).

## Use

Ensure that an OpenCL Installable Client Driver (ICD) and the appropriate OpenCL
hardware driver(s) are installed, see 
[OpenCL Installation](https://github.com/kenba/cl3/tree/main/docs/opencl_installation.md).

`opencl3` supports OpenCL 1.2 and 2.0 ICD loaders by default. If you have an
OpenCL 2.0 ICD loader then just add the following to your project's `Cargo.toml`:

```toml
[dependencies]
opencl3 = "0.6"
```

If your OpenCL ICD loader supports higher versions of OpenCL then add the
appropriate features to opencl3, e.g. for an OpenCL 3.0 ICD loader add the
following to your project's `Cargo.toml` instead:

```toml
[dependencies.opencl3]
version = "0.6"
features = ["CL_VERSION_2_1", "CL_VERSION_2_2", "CL_VERSION_3_0"]
```

OpenCL extensions can also be enabled by adding their features, e.g.:

```toml
[dependencies.opencl3]
version = "0.6"
features = ["cl_khr_gl_sharing", "cl_khr_dx9_media_sharing"]
```

For examples on how to use the library see the examples in the [examples](examples) directory and the integration tests in the [tests](tests) directory.

See the [OpenCL Guide ](https://github.com/KhronosGroup/OpenCL-Guide) and [OpenCL Description](https://github.com/kenba/opencl3/tree/main/docs/opencl_description.md) for background on using OpenCL.

## Recent changes

The API has changed considerably since version `0.1` of the library, with the
aim of making the library more consistent and easier to use.

The most recent change was to remove the Info enums from the underlying [cl3](https://crates.io/crates/cl3) crate and this crate so that data can be read from OpenCL devices in the future using new values that are currently undefined.

[SvmVec](src/svm.rs) was changed recently to provide better support for
coarse grain buffer Shared Virtual Memory now that Nvidia has started supporting it,
see [Nvidia OpenCL](https://developer.nvidia.com/opencl).

For information on other changes, see [Releases](RELEASES.md).

## Tests

The crate contains unit, documentation and integration tests.  
The tests run the platform and device info functions (among others) so they
can provide useful information about OpenCL capabilities of the system.

It is recommended to run the tests in single-threaded mode, since some of
them can interfere with each other when run multi-threaded, e.g.:

```shell
cargo test -- --test-threads=1 --show-output
```

The integration tests are marked `ignore` so use the following command to
run them:

```shell
cargo test -- --test-threads=1 --show-output --ignored
```

## Contribution

If you want to contribute through code or documentation, the [Contributing](CONTRIBUTING.md) guide is the best place to start. If you have any questions, please feel free to ask.
Just please abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under the Apache License, Version 2.0, as per Khronos Group OpenCL.  
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

Any contribution intentionally submitted for inclusion in the work by you shall be licensed  as defined in the Apache-2.0 license above, without any additional terms or conditions, unless you explicitly state otherwise.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.
