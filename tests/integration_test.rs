// Copyright (c) 2020-2021 Via Technology Ltd. All Rights Reserved.
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

extern crate opencl3;

use cl3::device::CL_DEVICE_TYPE_GPU;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;

const PROGRAM_SOURCE: &str = r#"
kernel void saxpy_float (global float* z,
    global float const* x,
    global float const* y,
    float a)
{
size_t i = get_global_id(0);
z[i] = a*x[i] + y[i];
}"#;

const KERNEL_NAME: &str = "saxpy_float";

#[test]
#[ignore]
fn test_opencl_1_2_example() -> Result<()> {
    let platforms = get_platforms()?;
    assert!(0 < platforms.len());

    // Get the first platform
    let platform = &platforms[0];

    let devices = platform
        .get_devices(CL_DEVICE_TYPE_GPU)
        .expect("Platform::get_devices failed");
    assert!(0 < devices.len());

    let platform_name = platform.name()?;
    println!("Platform Name: {:?}", platform_name);

    // Create OpenCL context from the first device
    let device = Device::new(devices[0]);
    let vendor = device.vendor().expect("Device.vendor failed");
    let vendor_id = device.vendor_id().expect("Device.vendor_id failed");
    println!("OpenCL device vendor name: {}", vendor);
    println!("OpenCL device vendor id: {:X}", vendor_id);

    /////////////////////////////////////////////////////////////////////
    // Initialise OpenCL compute environment

    // Create a Context on the OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");

    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    /////////////////////////////////////////////////////////////////////
    // Compute data

    // The input data
    const ARRAY_SIZE: usize = 1000;
    let ones: [cl_float; ARRAY_SIZE] = [1.0; ARRAY_SIZE];
    let mut sums: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
    for i in 0..ARRAY_SIZE {
        sums[i] = 1.0 + 1.0 * i as cl_float;
    }

    // Create OpenCL device buffers
    let mut x = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())?
    };
    let mut y = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())?
    };
    let z = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())?
    };

    // Blocking write
    let _x_write_event = unsafe { queue.enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &ones, &[])? };

    // Non-blocking write, wait for y_write_event
    let y_write_event =
        unsafe { queue.enqueue_write_buffer(&mut y, CL_NON_BLOCKING, 0, &sums, &[])? };

    // a value for the kernel function
    let a: cl_float = 300.0;

    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&z)
            .set_arg(&x)
            .set_arg(&y)
            .set_arg(&a)
            .set_global_work_size(ARRAY_SIZE)
            .set_wait_event(&y_write_event)
            .enqueue_nd_range(&queue)?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.
    let mut results: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
    let _event =
        unsafe { queue.enqueue_read_buffer(&z, CL_NON_BLOCKING, 0, &mut results, &events)? };

    // Block until all commands on the queue have completed
    queue.finish()?;

    assert_eq!(1300.0, results[ARRAY_SIZE - 1]);
    println!("results back: {}", results[ARRAY_SIZE - 1]);

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}", duration);

    Ok(())
}

#[cfg(feature = "CL_VERSION_2_0")]
#[test]
#[ignore]
fn test_opencl_svm_example() -> Result<()> {
    use cl3::device::{CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, CL_DEVICE_SVM_FINE_GRAIN_BUFFER};
    use opencl3::command_queue::CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE};
    use opencl3::svm::SvmVec;

    let platforms = get_platforms()?;
    assert!(0 < platforms.len());

    /////////////////////////////////////////////////////////////////////
    // Query OpenCL compute environment
    let opencl_2: &str = "OpenCL 2";
    let opencl_3: &str = "OpenCL 3";

    // Find an OpenCL SVM, platform and device
    let mut device_id = ptr::null_mut();
    let mut is_svm_capable: bool = false;
    for p in platforms {
        let platform_version = p.version()?;
        if platform_version.contains(&opencl_2) || platform_version.contains(&opencl_3) {
            let devices = p
                .get_devices(CL_DEVICE_TYPE_GPU)
                .expect("Platform::get_devices failed");

            for dev_id in devices {
                let device = Device::new(dev_id);
                let svm_mem_capability = device.svm_mem_capability();
                is_svm_capable = 0 < svm_mem_capability
                    & (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
                if is_svm_capable {
                    device_id = dev_id;
                    break;
                }
            }
        }
    }

    if is_svm_capable {
        // Create OpenCL context from the OpenCL svm device
        let device = Device::new(device_id);
        let vendor = device.vendor().expect("Device.vendor failed");
        let vendor_id = device.vendor_id().expect("Device.vendor_id failed");
        println!("OpenCL device vendor name: {}", vendor);
        println!("OpenCL device vendor id: {:X}", vendor_id);

        /////////////////////////////////////////////////////////////////////
        // Initialise OpenCL compute environment

        // Create a Context on the OpenCL svm device
        let context = Context::from_device(&device).expect("Context::from_device failed");

        // Build the OpenCL program source and create the kernel.
        let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
            .expect("Program::create_and_build_from_source failed");

        let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

        // Create a command_queue on the Context's device
        let queue = CommandQueue::create_default_with_properties(
            &context,
            CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            0,
        )
        .expect("CommandQueue::create_default_with_properties failed");

        /////////////////////////////////////////////////////////////////////
        // Compute data

        // Get the svm capability of all the devices in the context.
        let svm_capability = context.get_svm_mem_capability();
        assert!(0 < svm_capability);

        let is_fine_grained_svm: bool = 0 < svm_capability & CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
        println!("OpenCL SVM is fine grained: {}", is_fine_grained_svm);

        // Create SVM vectors for the data

        // The SVM vectors
        const ARRAY_SIZE: usize = 1000;
        let mut ones =
            SvmVec::<cl_float>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");
        let mut sums =
            SvmVec::<cl_float>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");
        let mut results =
            SvmVec::<cl_float>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");

        let a: cl_float = 300.0;
        if is_fine_grained_svm {
            // The input data
            for i in 0..ARRAY_SIZE {
                ones[i] = 1.0;
            }

            for i in 0..ARRAY_SIZE {
                sums[i] = 1.0 + 1.0 * i as cl_float;
            }

            // Make ones and sums immutable
            let ones = ones;
            let sums = sums;

            // Use the ExecuteKernel builder to set the kernel buffer and
            // cl_float value arguments, before setting the one dimensional
            // global_work_size for the call to enqueue_nd_range.
            // Unwraps the Result to get the kernel execution event.
            let kernel_event = unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg_svm(results.as_mut_ptr())
                    .set_arg_svm(ones.as_ptr())
                    .set_arg_svm(sums.as_ptr())
                    .set_arg(&a)
                    .set_global_work_size(ARRAY_SIZE)
                    .enqueue_nd_range(&queue)?
            };

            // Wait for the kernel_event to complete
            kernel_event.wait()?;

            assert_eq!(1300.0, results[ARRAY_SIZE - 1]);
            println!("results back: {}", results[ARRAY_SIZE - 1]);

            // Calculate the kernel duration, from the kernel_event
            let start_time = kernel_event.profiling_command_start()?;
            let end_time = kernel_event.profiling_command_end()?;
            let duration = end_time - start_time;
            println!("kernel execution duration (ns): {}", duration);
        } else {
            // !is_fine_grained_svm
            // unsafe { ones.set_len(ARRAY_SIZE) };
            // unsafe { sums.set_len(ARRAY_SIZE) };

            // Map the input SVM vectors, before setting their data
            unsafe {
                queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut ones, &[])?;
                queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut sums, &[])?;
            }
            // The input data
            for i in 0..ARRAY_SIZE {
                ones[i] = 1.0;
            }

            for i in 0..ARRAY_SIZE {
                sums[i] = 1.0 + 1.0 * i as cl_float;
            }

            // Make ones and sums immutable
            let ones = ones;
            let sums = sums;

            let mut events: Vec<cl_event> = Vec::default();
            let unmap_sums_event = unsafe { queue.enqueue_svm_unmap(&sums, &[])? };
            let unmap_ones_event = unsafe { queue.enqueue_svm_unmap(&ones, &[])? };
            events.push(unmap_sums_event.get());
            events.push(unmap_ones_event.get());

            // Use the ExecuteKernel builder to set the kernel buffer and
            // cl_float value arguments, before setting the one dimensional
            // global_work_size for the call to enqueue_nd_range.
            // Unwraps the Result to get the kernel execution event.
            let kernel_event = unsafe {
                ExecuteKernel::new(&kernel)
                    .set_arg_svm(results.as_mut_ptr())
                    .set_arg_svm(ones.as_ptr())
                    .set_arg_svm(sums.as_ptr())
                    .set_arg(&a)
                    .set_global_work_size(ARRAY_SIZE)
                    .set_event_wait_list(&events)
                    .enqueue_nd_range(&queue)?
            };

            // Wait for the kernel_event to complete
            kernel_event.wait()?;

            // Map SVM results before reading them
            let _map_results_event =
                unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, &mut results, &[])? };

            assert_eq!(1300.0, results[ARRAY_SIZE - 1]);
            println!("results back: {}", results[ARRAY_SIZE - 1]);

            // Calculate the kernel duration from the kernel_event
            let start_time = kernel_event.profiling_command_start()?;
            let end_time = kernel_event.profiling_command_end()?;
            let duration = end_time - start_time;
            println!("kernel execution duration (ns): {}", duration);

            /////////////////////////////////////////////////////////////////////
            // Clean up
            let unmap_results_event = unsafe { queue.enqueue_svm_unmap(&results, &[])? };
            unmap_results_event.wait()?;
            println!("SVM buffers unmapped");
        }
    } else {
        println!("OpenCL SVM capable device not found")
    }

    Ok(())
}
