// Copyright (c) 2020 Via Technology Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! [![crates.io](https://img.shields.io/crates/v/opencl3.svg)](https://crates.io/crates/opencl3)
//! [![docs.io](https://docs.rs/opencl3/badge.svg)](https://docs.rs/opencl3/)
//! [![OpenCL 3.0](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/registry/OpenCL/)
//! [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
//!
//! A Rust implementation of the Khronos [OpenCL](https://www.khronos.org/registry/OpenCL/)
//! API.
//!
//! # Description
//!
//! This crate provides a relatively simple, object based model of the OpenCL 3.0
//! [API](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html).  
//! It is built upon the [cl3](https://crates.io/crates/cl3) crate, which
//! provides a functional interface to the OpenCL API.  
//!
//! **OpenCL** (Open Computing Language) is framework for general purpose
//! parallel programming across heterogeneous devices including: CPUs, GPUs,
//! DSPs, FPGAs and other processors or hardware accelerators.
//!
//! It is often considered as an open-source alternative to Nvidia's proprietary
//! Compute Unified Device Architecture [CUDA](https://developer.nvidia.com/cuda-zone)
//! for performing General-purpose computing on GPUs, see
//! [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units).
//!
//! The [OpenCL Specification](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_the_opencl_architecture)
//! has evolved over time and not all device vendors support all OpenCL features.
//!
//! [OpenCL 3.0](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html)
//! is a unified specification that adds little new functionality to previous OpenCL versions.  
//! It specifies that all **OpenCL 1.2** features are **mandatory**, while all
//! OpenCL 2.x and OpenCL 3.0 features are now optional.
//!
//! # OpenCL Architecture
//!
//! The [OpenCL Specification](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_the_opencl_architecture)
//! considers OpenCL as four models:
//!
//! * **Platform Model**  
//! The physical OpenCL hardware: a *host* containing one or more OpenCL [platform]s,
//! each connected to one or more OpenCL [device]s.  
//! An OpenCL application running on the *host*, creates an OpenCL environment
//! called a [context] on a single [platform] to process data on one or more
//! of the OpenCL [device]s connected to the [platform].
//!
//! * **Programming Model**  
//! An OpenCL [program] consists of OpenCL [kernel] functions that can run
//! on OpenCL [device]s within a [context].  
//! OpenCL [program]s must be created (and most must be built) for a [context]
//! before their OpenCL [kernel] functions can be created from them,
//! the exception being "built-in" [kernel]s which don't need to be built
//! (or compiled and linked).  
//! OpenCL [kernel]s are controlled by an OpenCL application that runs on the
//! *host*, see **Execution Model**.
//!
//! * **Memory Model**  
//! **OpenCL 1.2** memory is divided into two fundamental memory regions:
//! **host memory** and **device memory**.  
//! OpenCL [kernel]s run on **device memory**; an OpenCL application must write
//! **host memory** to **device memory** for OpenCL [kernel]s to process.
//! An OpenCL application must also read results from **device memory** to
//! **host memory** after a [kernel] has completed execution.  
//! **OpenCL 2.0** shared virtual memory ([svm]) is shared between the host
//! and device(s) and synchronised by OpenCL; eliminating the explicit transfer
//! of memory between host and device(s) memory regions.
//!
//! * **Execution Model**  
//! An OpenCL application creates at least one OpenCL [command_queue] for each
//! OpenCL [device] (or *sub-device*) within it's OpenCL [context].  
//! OpenCL [kernel] executions and **OpenCL 1.2** memory reads and writes are
//! "enqueued" by the OpenCL application on each [command_queue].
//! An application can wait for all "enqueued" commands to finish on a
//! [command_queue] or it can wait for specific [event]s to complete.
//! TODO describe out-of-order execution.
//!
//! # OpenCL Objects
//!
//! ## Platform Model
//!
//! The platform model has thee objects:
//! * [platform]
//! * [device]
//! * [context]
//!
//! Of these three objects, the OpenCL [context] is by *far* the most important.
//! Each application must create a [context] from the most appropriate [device]s
//! available on one of [platform]s on the *host* system that the application
//! is running on.
//!
//! Most example OpenCL applications just choose the first available [platform]
//! and [device] for their [context]. However, since many systems have multiple
//! platforms and devices, the first [platform] and [device] are unlikely to
//! provide the best performance.  
//! For example, on a system with an APU (combined CPU and GPU, e.g. Intel i7)
//! and a discrete graphics card (e.g. Nvidia GTX 1070) OpenCL may find the
//! either the integrated GPU or the GPU on the graphics card first.
//!
//! OpenCL applications often require the performance of discrete graphics cards
//! or specific OpenCL features, such as [svm] or double/half floating point
//! precision. In such cases, it is necessary to query the [platform]s and
//! [device]s to choose the most appropriate [device]s for the application before
//! creating the [context].
//!
//! The [platform] and [device] modules contain structures and methods to simplify
//! querying the host system [platform]s and [device]s to create a [context].
//!
//! ## Programming Model
//!
//! The OpenCL programming model has two objects:
//! * [program]
//! * [kernel]
//!
//! OpenCL [kernel] functions are contained in OpenCL [program]s.  
//!
//! Kernels are usually defined as functions in OpenCL program source code,
//! however OpenCL devices may contain built-in kernels,
//! e.g.: some Intel GPUs have built-in motion estimation kernels.
//!
//! OpenCL [program] objects can be created from OpenCL source code,
//! built-in kernels, binaries and intermediate language binaries.
//! Depending upon how an OpenCL [program] object was created, it may need to
//! be built (or complied and linked) before the [kernel]s in them can be
//! created.
//!
//! All the [kernel]s in an [program] can be created together or they can be
//! created individually, by name.
//!
//! ## Memory Model
//!
//! The OpenCL memory model consists of five objects:
//! * `buffer`
//! * `image`
//! * [sampler]
//! * [svm]
//! * `pipe`
//!
//! `buffer`, `image` and [sampler] are OpenCL 1.2 (i.e. **mandatory**) objects,  
//! [svm] and `pipe` are are OpenCL 2.0 (i.e. optional) objects.
//!
//! A `buffer` is a contiguous block of memory used for general purpose data.  
//! An `image` holds data for one, two or three dimensional images.  
//! A [sampler] describes how a [kernel] is to sample an `image`, see
//! [Sampler objects](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_sampler_objects).  
//!
//! [Shared Virtual Memory](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_shared_virtual_memory)
//! enables the host and kernels executing on devices to directly share data
//! without explicitly transferring it.
//!
//! [Pipes](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_pipes)
//! store memory as FIFOs between [kernel]s. Pipes are not accessible from the host.
//!
//! ## Execution Model
//!
//! The OpenCL execution model has two objects:
//! * [command_queue]
//! * [event]
//!
//! OpenCL commands to transfer memory and execute kernels on devices are
//! performed via [command_queue]s.
//!
//! Each OpenCL device (and sub-device) must have at least one command_queue
//! associated with it, so that commands may be enqueued on to the device.
//!
//! There are several OpenCL command_queue "enqueue_" methods to transfer
//! data between host and device memory, map SVM memory and execute kernels.
//! All the "enqueue_" methods accept an event_wait_list parameter and return
//! an [event] that can be used to monitor and control *out-of-order* execution
//! of kernels on a command_queue, see
//! [Event Objects](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#event-objects).

extern crate cl3;

pub mod command_queue;
pub mod context;
pub mod device;
pub mod event;
pub mod kernel;
pub mod platform;
pub mod program;
pub mod sampler;
pub mod svm;
pub mod memory;

pub mod error_codes {
    pub use cl3::error_codes::*;
}
pub mod types {
    pub use cl3::types::*;
}
