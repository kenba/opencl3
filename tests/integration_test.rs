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

use cl3::device::{CL_DEVICE_SVM_FINE_GRAIN_BUFFER, CL_DEVICE_TYPE_GPU};
use opencl3::command_queue::{CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::event;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::platform::get_platforms;
use opencl3::types::{cl_event, cl_float, CL_FALSE, CL_TRUE};
use std::ffi::CString;
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
fn test_opencl_1_2_example() {
    let platforms = get_platforms().unwrap();
    assert!(0 < platforms.len());

    // Get the first platform
    let platform = &platforms[0];

    let devices = platform
        .get_devices(CL_DEVICE_TYPE_GPU)
        .expect("Platform::get_devices failed");
    assert!(0 < devices.len());

    // Create OpenCL context from the first device
    let device = Device::new(devices[0]);
    let vendor = device.vendor().expect("Device.vendor failed");
    let vendor_id = device.vendor_id().expect("Device.vendor_id failed");
    println!("OpenCL device vendor name: {:?}", vendor);
    println!("OpenCL device vendor id: {:X}", vendor_id);

    // Create a Context and a queue on the device
    let mut context = Context::from_device(device).expect("Context::from_device failed");

    // Create a command_queue on the the device
    context
        .create_command_queues(CL_QUEUE_PROFILING_ENABLE)
        .expect("Context::create_command_queues failed");

    // Build the OpenCL program source and create the kernel.
    let src = CString::new(PROGRAM_SOURCE).unwrap();
    let options = CString::default();
    context
        .build_program_from_source(&src, &options)
        .expect("Context::build_program_from_source failed");

    assert!(!context.kernels().is_empty());
    for kernel_name in context.kernels().keys() {
        println!("Kernel name: {:?}", kernel_name);
    }

    // The input data
    const ARRAY_SIZE: usize = 1000;
    let ones: [cl_float; ARRAY_SIZE] = [1.0; ARRAY_SIZE];
    let mut sums: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
    for i in 0..ARRAY_SIZE {
        sums[i] = 1.0 + 1.0 * i as cl_float;
    }

    // Create OpenCL device buffers
    let x = context
        .create_buffer::<cl_float>(CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())
        .unwrap();
    let y = context
        .create_buffer::<cl_float>(CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())
        .unwrap();
    let z = context
        .create_buffer::<cl_float>(CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())
        .unwrap();

    let queue = context.default_queue();

    let mut events: Vec<cl_event> = Vec::default();

    // Blocking write
    let _x_write_event = queue
        .enqueue_write_buffer(x, CL_TRUE, 0, &ones, &events)
        .unwrap();

    // Non-blocking write, wait for y_write_event
    let y_write_event = queue
        .enqueue_write_buffer(y, CL_FALSE, 0, &sums, &events)
        .unwrap();
    events.push(y_write_event.get());

    // Convert to CString for get_kernel function
    let kernel_name = CString::new(KERNEL_NAME).unwrap();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        // a value for the kernel function
        let a: cl_float = 300.0;

        // Use the ExecuteKernel builder to set the kernel buffer and
        // cl_float value arguments, before setting the one dimensional
        // global_work_size for the call to enqueue_nd_range.
        // Unwraps the Result to get the kernel execution event.
        let kernel_event = ExecuteKernel::new(kernel)
            .set_arg(&z)
            .set_arg(&x)
            .set_arg(&y)
            .set_arg(&a)
            .set_global_work_size(ARRAY_SIZE)
            .enqueue_nd_range(&queue, &events)
            .unwrap();
        events.clear();
        events.push(kernel_event.get());

        // Create a results array to hold the results from the OpenCL device
        // and enqueue a read command to read the device buffer into the array
        // after the kernel event completes.
        let mut results: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
        let _event = queue
            .enqueue_read_buffer(z, CL_FALSE, 0, &mut results, &events)
            .unwrap();
        events.clear();

        // Block until all commands on the queue have completed
        queue.finish().unwrap();

        assert_eq!(1300.0, results[ARRAY_SIZE - 1]);
        println!("results back: {}", results[ARRAY_SIZE - 1]);
    }
}

#[test]
#[ignore]
fn test_opencl_2_0_example() {
    let platforms = get_platforms().unwrap();
    assert!(0 < platforms.len());

    let opencl_2: String = "OpenCL 2".to_string();
    let mut device_id = ptr::null_mut();

    // Find an OpenCL 2, platform and device
    let mut is_opencl_2: bool = false;
    for p in platforms {
        let platform_version = p.version().unwrap().into_string().unwrap();
        if platform_version.contains(&opencl_2) {
            let devices = p
                .get_devices(CL_DEVICE_TYPE_GPU)
                .expect("Platform::get_devices failed");

            for d in devices {
                let dev = Device::new(d);
                let device_version = dev.version().unwrap().into_string().unwrap();
                is_opencl_2 = device_version.contains(&opencl_2);
                if is_opencl_2 {
                    device_id = d;
                    break;
                }
            }
        }
    }

    if !is_opencl_2 {
        assert!(false, "OpenCL 2 device not found")
    }

    // Create OpenCL context from the OpenCL 2 device
    let device = Device::new(device_id);
    let vendor = device.vendor().expect("Device.vendor failed");
    let vendor_id = device.vendor_id().expect("Device.vendor_id failed");
    let svm_mem_capability = device.svm_mem_capability();
    let is_fine_grained_svm: bool = 0 < svm_mem_capability & CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
    println!("OpenCL device vendor name: {:?}", vendor);
    println!("OpenCL device vendor id: {:X}", vendor_id);
    println!("OpenCL SVM is fine grained: {}", is_fine_grained_svm);

    let mut context = Context::from_device(device).expect("Context::from_device failed");
    context
        .create_command_queues_with_properties(
            CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            0,
        )
        .expect("Context::create_command_queues_with_properties failed");

    // Build the OpenCL program source and create the kernel.
    let src = CString::new(PROGRAM_SOURCE).unwrap();
    let options = CString::default();
    context
        .build_program_from_source(&src, &options)
        .expect("Context::build_program_from_source failed");

    assert!(!context.kernels().is_empty());
    for kernel_name in context.kernels().keys() {
        println!("Kernel name: {:?}", kernel_name);
    }

    let svm_capability = context.get_svm_mem_capability();
    assert!(0 < svm_capability);

    // The input data
    const ARRAY_SIZE: usize = 1000;
    let mut ones = context.create_svm_vec::<cl_float>(svm_capability);
    ones.reserve(ARRAY_SIZE);
    for _ in 0..ARRAY_SIZE {
        ones.push(1.0);
    }

    let mut sums = context.create_svm_vec::<cl_float>(svm_capability);
    sums.reserve(ARRAY_SIZE);
    for i in 0..ARRAY_SIZE {
        sums.push(1.0 + 1.0 * i as cl_float);
    }

    let queue = context.default_queue();

    // Convert to CString for get_kernel function
    let kernel_name = CString::new(KERNEL_NAME).unwrap();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let mut results = context.create_svm_vec::<cl_float>(svm_capability);
        results.reserve(ARRAY_SIZE);
        for i in 0..ARRAY_SIZE {
            results.push(i as cl_float);
        }

        let mut events: Vec<cl_event> = Vec::default();
        if is_fine_grained_svm {
            let a: cl_float = 300.0;

            let kernel_event = ExecuteKernel::new(kernel)
                .set_arg_svm(results.as_mut_ptr())
                .set_arg_svm(ones.as_ptr())
                .set_arg_svm(sums.as_ptr())
                .set_arg(&a)
                .set_global_work_size(ARRAY_SIZE)
                .enqueue_nd_range(&queue, &events)
                .unwrap();
            events.clear();
            events.push(kernel_event.get());
            event::wait_for_events(&events).unwrap();
            events.clear();

            assert_eq!(1300.0, results[ARRAY_SIZE - 1]);
            println!("results back: {}", results[ARRAY_SIZE - 1]);
        } else {
            // !is_fine_grained_svm
            // Map the SVM
            let map_ones_event = queue
                .enqueue_svm_map(CL_FALSE, CL_MAP_WRITE, &mut ones, &events)
                .unwrap();
            let map_sums_event = queue
                .enqueue_svm_map(CL_FALSE, CL_MAP_WRITE, &mut sums, &events)
                .unwrap();

            events.push(map_ones_event.get());
            events.push(map_sums_event.get());

            let a: cl_float = 300.0;

            let kernel_event = ExecuteKernel::new(kernel)
                .set_arg_svm(results.as_mut_ptr())
                .set_arg_svm(ones.as_ptr())
                .set_arg_svm(sums.as_ptr())
                .set_arg(&a)
                .set_global_work_size(ARRAY_SIZE)
                .enqueue_nd_range(&queue, &events)
                .unwrap();
            events.clear();

            events.push(kernel_event.get());
            event::wait_for_events(&events).unwrap();

            events.clear();

            let _map_results_event = queue
                .enqueue_svm_map(CL_TRUE, CL_MAP_READ, &mut results, &events)
                .unwrap();

            let unmap_results_event = queue.enqueue_svm_unmap(&results, &events).unwrap();
            let unmap_sums_event = queue.enqueue_svm_unmap(&sums, &events).unwrap();
            let unmap_ones_event = queue.enqueue_svm_unmap(&ones, &events).unwrap();
            events.push(unmap_results_event.get());
            events.push(unmap_sums_event.get());
            events.push(unmap_ones_event.get());

            assert_eq!(1300.0, results[ARRAY_SIZE - 1]);
            println!("results back: {}", results[ARRAY_SIZE - 1]);

            event::wait_for_events(&events).unwrap();
            println!("SVM buffers unmapped");
        }
    }
}
