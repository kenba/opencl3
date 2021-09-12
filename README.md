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

## Example

The tests provide examples of how the crate may be used, e.g. see:
[platform](https://github.com/kenba/opencl3/tree/main/src/platform.rs),
[device](https://github.com/kenba/opencl3/tree/main/src/device.rs),
[context](https://github.com/kenba/opencl3/tree/main/src/context.rs),
[integration_test](https://github.com/kenba/opencl3/tree/main/tests/integration_test.rs) and
[opencl2_kernel_test](https://github.com/kenba/opencl3/tree/main/tests/opencl2_kernel_test.rs).

The library is designed to support events and OpenCL 2 features such as Shared Virtual Memory (SVM) and kernel built-in work-group functions, e.g.:

```rust no-run
const PROGRAM_SOURCE: &str = r#"
kernel void inclusive_scan_int (global int* output,
                                global int const* values)
{
    int sum = 0;
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);

    size_t num_groups = get_num_groups(0);
    for (size_t i = 0u; i < num_groups; ++i)
    {
        size_t lidx = i * lsize + lid;
        int value = work_group_scan_inclusive_add(values[lidx]);
        output[lidx] = sum + value;

        sum += work_group_broadcast(value, lsize - 1);
    }
}"#;

const KERNEL_NAME: &str = "inclusive_scan_int";

// Create a Context on an OpenCL device
let context = Context::from_device(&device).expect("Context::from_device failed");

// Build the OpenCL program source and create the kernel.
let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_STD_2_0)
    .expect("Program::create_and_build_from_source failed");
let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

// Create a command_queue on the Context's device
let queue = CommandQueue::create_with_properties(
    &context,
    context.default_device(),
    CL_QUEUE_PROFILING_ENABLE,
    0,
)
.expect("CommandQueue::create_with_properties failed");

// The input data
const ARRAY_SIZE: usize = 8;
let value_array: [cl_int; ARRAY_SIZE] = [3, 2, 5, 9, 7, 1, 4, 2];

// Create an OpenCL SVM vector
let mut test_values =SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE)
    .expect("SVM allocation failed");

// Map test_values if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
if !test_values.is_fine_grained() {
    queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut test_values, &[])?;
}

// Copy input data into the OpenCL SVM vector
test_values.clone_from_slice(&value_array);

// Make test_values immutable
let test_values = test_values;

// Unmap test_values if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
if !test_values.is_fine_grained() {
    let unmap_test_values_event = queue.enqueue_svm_unmap(&test_values, &[])?;
    unmap_test_values_event.wait()?;
}

// The output data, an OpenCL SVM vector
let mut results = SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE)
    .expect("SVM allocation failed");

// Run the kernel on the input data
let kernel_event = ExecuteKernel::new(kernel)
    .set_arg_svm(results.as_mut_ptr())
    .set_arg_svm(test_values.as_ptr())
    .set_global_work_size(ARRAY_SIZE)
    .enqueue_nd_range(&queue)?;

// Wait for the kernel to complete execution on the device
kernel_event.wait()?;

// Map results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
if !results.is_fine_grained() {
    queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, &mut results, &[])?;
}

// Can access OpenCL SVM directly, no need to map or read the results
println!("sum results: {:?}", results);

// Unmap results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
if !results.is_fine_grained() {
    let unmap_results_event = queue.enqueue_svm_unmap(&results, &[])?;
    unmap_results_event.wait()?;
}
```

## Use

Ensure that an OpenCL Installable Client Driver (ICD) and the appropriate OpenCL
hardware driver(s) are installed, see 
[OpenCL Installation](https://github.com/kenba/cl3/tree/main/docs/opencl_installation.md).

`opencl3` supports OpenCL 1.2 and 2.0 ICD loaders by default. If you have an
OpenCL 2.0 ICD loader then just add the following to your project's `Cargo.toml`:

```toml
[dependencies]
opencl3 = "0.5"
```

If your OpenCL ICD loader supports higher versions of OpenCL then add the
appropriate features to opencl3, e.g. for an OpenCL 3.0 ICD loader add the
following to your project's `Cargo.toml` instead:

```toml
[dependencies.opencl3]
version = "0.5"
features = ["CL_VERSION_2_1", "CL_VERSION_2_2", "CL_VERSION_3_0"]
```

For examples on how to use the library see the integration tests in
[integration_test.rs](https://github.com/kenba/opencl3/tree/main/tests/integration_test.rs)

See [OpenCL Description](https://github.com/kenba/opencl3/tree/main/docs/opencl_description.md) for background on using OpenCL.

## Recent changes

The API has changed considerably since version `0.1` of the library, with the
aim of making the library more consistent and easier to use.

The most recent change is to [SvmVec](src/svm.rs) to provide better support for
coarse grain buffer Shared Virtual Memory now that Nvidia has started supporting it,
see [Nvidia OpenCL](https://developer.nvidia.com/opencl).

[Context](src/context.rs) no longer contains: Programs, Kernels and Command Queues.
They must now be built separately, as shown in the example above.

It is now recommended to call the `Program::create_and_build_from_*` methods
to build programs since they will return the build log if there is a build failure.

The OpenCL function calls now return an error type with a Display trait that
shows the *name* of the OpenCL error, not just its number.

The `Event` API now returns `CommandExecutionStatus` and `EventCommandType`
which also use the Display trait to display their names.

The API for `memory` structs: `Buffer`, `Image` and `Pipe` have been unified
using the `ClMem` trait object.

## Design

Nearly all the structs implement the `Drop` trait to release their corresponding
OpenCL objects, see the crate [documentation](https://docs.rs/opencl3/).

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

## License

Licensed under the Apache License, Version 2.0, as per Khronos Group OpenCL.  
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

OpenCL and the OpenCL logo are trademarks of Apple Inc. used under license by Khronos.
