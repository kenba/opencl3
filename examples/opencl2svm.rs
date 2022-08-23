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
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::error_codes::cl_int;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE};
use opencl3::program::{Program, CL_STD_2_0};
use opencl3::svm::SvmVec;
use opencl3::types::CL_BLOCKING;
use opencl3::Result;

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

fn main() -> Result<()> {
    // Find a usable platform and device for this application
    let platforms = opencl3::platform::get_platforms()?;
    let platform = platforms.first().expect("no OpenCL platforms");
    let device = *platform
        .get_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_STD_2_0)
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    // Create a command_queue on the Context's device
    let queue = unsafe {
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
            .expect("CommandQueue::create_default_with_properties failed")
    };

    // The input data
    const ARRAY_SIZE: usize = 8;
    let value_array: [cl_int; ARRAY_SIZE] = [3, 2, 5, 9, 7, 1, 4, 2];

    // Create an OpenCL SVM vector
    let mut test_values =
        SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");

    // Map test_values if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
    if !test_values.is_fine_grained() {
        unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut test_values, &[])? };
    }

    // Copy input data into the OpenCL SVM vector
    test_values.clone_from_slice(&value_array);

    // Make test_values immutable
    let test_values = test_values;

    // Unmap test_values if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
    if !test_values.is_fine_grained() {
        let unmap_test_values_event = unsafe { queue.enqueue_svm_unmap(&test_values, &[])? };
        unmap_test_values_event.wait()?;
    }

    // The output data, an OpenCL SVM vector
    let mut results =
        SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");

    // Run the kernel on the input data
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg_svm(results.as_mut_ptr())
            .set_arg_svm(test_values.as_ptr())
            .set_global_work_size(ARRAY_SIZE)
            .enqueue_nd_range(&queue)?
    };

    // Wait for the kernel to complete execution on the device
    kernel_event.wait()?;

    // Map results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
    if !results.is_fine_grained() {
        unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, &mut results, &[])? };
    }

    // Can access OpenCL SVM directly, no need to map or read the results
    println!("sum results: {:?}", results);

    // Unmap results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
    if !results.is_fine_grained() {
        let unmap_results_event = unsafe { queue.enqueue_svm_unmap(&results, &[])? };
        unmap_results_event.wait()?;
    }

    Ok(())
}
