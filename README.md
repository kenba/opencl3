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

```rust
use opencl3::types::CL_TRUE;
use opencl3::memory::{CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, Buffer};
use opencl3::error_codes::ClError;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::program::{CL_STD_3_0, Program};
use opencl3::context::Context;
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};

const PROGRAM_SOURCE: &str = r#"
kernel void vector_add(global int *a,
                       global int *b,
                       global int *c)
{
    uint i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"#;

const KERNEL_NAME: &str = "vector_add";

const LENGTH: usize = 8;
const A: [i32; LENGTH] = [1,2,3,4,5,6,7,8];
const B: [i32; LENGTH] = [2,3,4,5,6,7,8,9];

fn main() -> Result<(), ClError> {
    // Find a usable platform and device for this application
    let platform = opencl3::platform::get_platforms()?.pop()
        .expect("get_platforms failed");
    let device = platform.get_devices(CL_DEVICE_TYPE_GPU)?.pop()
        .expect("get_devices failed");
    let device = Device::new(device);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device)
        .expect("Context::from_device failed");

    // Compile the OpenCL program source and create the kernel
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_STD_3_0)
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME)
        .expect("Kernel::create failed");

    // Create a command_queue on the Context's only device
    let queue = CommandQueue::create_with_properties(
        &context,
        context.default_device(),
        CL_QUEUE_PROFILING_ENABLE,
        0)
        .expect("CommandQueue::create_with_properties failed");

    // Create input and output Buffers
    println!("  {:?}\n  {:?}\n+ ------------------------", A, B);
    let mut a = Buffer::create(&context, CL_MEM_READ_ONLY, LENGTH, std::ptr::null_mut())
        .expect("Buffer::create failed (a)");
    let mut b = Buffer::create(&context, CL_MEM_READ_ONLY, LENGTH, std::ptr::null_mut())
        .expect("Buffer::create failed (b)");
    let mut c = Buffer::create(&context, CL_MEM_WRITE_ONLY, LENGTH, std::ptr::null_mut())
        .expect("Buffer::create failed (c)");

    // Write to the input buffers
    queue.enqueue_write_buffer(&mut a, CL_TRUE, 0, &A, &[])
        .expect("enqueue_write_buffer failed (a)");
    queue.enqueue_write_buffer(&mut b, CL_TRUE, 0, &B, &[])
        .expect("enqueue_write_buffer failed (b)");

    // Launch the kernel with the IO buffers
    let kernel_event = ExecuteKernel::new(&kernel)
        .set_arg(&a)
        .set_arg(&b)
        .set_arg(&c)
        .set_global_work_size(LENGTH)
        .enqueue_nd_range(&queue)?;

    // Block until work items are processed
    kernel_event.wait()?;

    // Read from output buffer
    let mut output = [0; LENGTH];
    queue.enqueue_read_buffer(&mut c, CL_TRUE, 0, &mut output, &[])
        .expect("enqueue_read_buffer failed (c)");
    println!("= {:?}", output);

    Ok(())
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
