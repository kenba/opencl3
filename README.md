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
hardware driver(s) are installed, see 
[OpenCL Installation](https://github.com/kenba/cl3/tree/main/docs/opencl_installation.md).

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

For examples on how to use the library see the integration tests in
[integration_test.rs](https://github.com/kenba/opencl3/tree/main/tests/integration_test.rs)

See [OpenCL Description](https://github.com/kenba/opencl3/tree/main/docs/opencl_description.md) for background on using OpenCL.

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

## Examples

The tests provide examples of how the crate may be used, e.g. see:
[platform](https://github.com/kenba/opencl3/tree/main/src/platform.rs),
[device](https://github.com/kenba/opencl3/tree/main/src/device.rs),
[context](https://github.com/kenba/opencl3/tree/main/src/context.rs),
[integration_test](https://github.com/kenba/opencl3/tree/main/tests/integration_test.rs) and
[opencl2_kernel_test](https://github.com/kenba/opencl3/tree/main/tests/opencl2_kernel_test.rs).

The library is designed to support events and OpenCL 2 features such as Shared Virtual Memory (SVM) and kernel built-in work-group functions, e.g.:

```rust no-run
// Create OpenCL context from the OpenCL 2 device
// and an OpenCL command queue for the device
let mut context = Context::from_device(device).unwrap();
context.create_command_queues_with_properties(0, 0).unwrap();

// Build the OpenCL 2.0 program source and create the kernels.
let src = CString::new(PROGRAM_SOURCE).unwrap();
let options = CString::new(PROGRAM_BUILD_OPTIONS).unwrap();
context.build_program_from_source(&src, &options).unwrap();

// Get the svm capability of all the devices in the context.
let svm_capability = context.get_svm_mem_capability();

// The input data
const ARRAY_SIZE: usize = 8;
let value_array: [cl_int; ARRAY_SIZE] = [3, 2, 5, 9, 7, 1, 4, 2];

// Copy input data into an OpenCL SVM vector
let mut test_values = SvmVec::<cl_int>::with_capacity(&context, svm_capability, ARRAY_SIZE);
for &val in value_array.iter() {
    test_values.push(val);
}

// The output data, an OpenCL SVM vector
let mut results =
    SvmVec::<cl_int>::with_capacity_zeroed(&context, svm_capability, ARRAY_SIZE);
unsafe { results.set_len(ARRAY_SIZE) };

// Get the command queue for the device
let queue = context.default_queue();

// Run the inclusive scan kernel on the input data
let inclusive_scan_kernel_name = CString::new(INCLUSIVE_SCAN_KERNEL_NAME).unwrap();
if let Some(inclusive_scan_kernel) = context.get_kernel(&inclusive_scan_kernel_name) {
    // No need to write or map the data before enqueueing the kernel
    let kernel_event = ExecuteKernel::new(inclusive_scan_kernel)
        .set_arg_svm(results.as_mut_ptr())
        .set_arg_svm(test_values.as_ptr())
        .set_global_work_size(ARRAY_SIZE)
        .enqueue_nd_range(&queue)
        .unwrap();

    // Wait for the kernel to complete execution on the device
    kernel_event.wait().unwrap();

    // Can access OpenCL SVM directly, no need to map or read the results
    println!("sum results: {:?}", results);
}
```

## License

Licensed under the Apache License, Version 2.0, as per Khronos Group OpenCL.  
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

OpenCL and the OpenCL logo are trademarks of Apple Inc. used under license by Khronos.
