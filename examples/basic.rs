// Copyright (c) 2021 Via Technology Ltd. All Rights Reserved.
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

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
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
    const size_t i = get_global_id(0);
    z[i] = a*x[i] + y[i];
}"#;

const KERNEL_NAME: &str = "saxpy_float";

fn main() -> Result<()> {
    // Find a usable device for this application
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

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
    let read_event =
        unsafe { queue.enqueue_read_buffer(&z, CL_NON_BLOCKING, 0, &mut results, &events)? };

    // Wait for the read_event to complete.
    read_event.wait()?;

    // Output the first and last results
    println!("results front: {}", results[0]);
    println!("results back: {}", results[ARRAY_SIZE - 1]);

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}", duration);

    Ok(())
}
