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

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::error_codes::cl_int;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE};
use opencl3::program::{Program, CL_STD_2_0};
use opencl3::svm::{ExtendSvmVec, SvmVec};
use opencl3::types::CL_BLOCKING;
use opencl3::Result;
use serde::de::DeserializeSeed;
use std::ptr;

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
    // Find a suitable device for this application
    let devices = get_all_devices(CL_DEVICE_TYPE_GPU)?;
    assert!(0 < devices.len());

    // Find an OpenCL SVM device
    let mut device_id = ptr::null_mut();
    let mut is_svm_capable: bool = false;
    for dev_id in devices {
        let device = Device::new(dev_id);
        let svm_mem_capability = device.svm_mem_capability();
        is_svm_capable = 0 < svm_mem_capability;
        if is_svm_capable {
            device_id = dev_id;
            break;
        }
    }

    if is_svm_capable {
        // Create OpenCL context from the OpenCL svm device
        let device = Device::new(device_id);
        let vendor = device.vendor()?;
        let vendor_id = device.vendor_id()?;
        println!("OpenCL device vendor name: {}", vendor);
        println!("OpenCL device vendor id: {:X}", vendor_id);

        /////////////////////////////////////////////////////////////////////
        // Initialise OpenCL compute environment

        // Create a Context on the OpenCL svm device
        let context = Context::from_device(&device).expect("Context::from_device failed");

        // Build the OpenCL program source and create the kernel.
        let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_STD_2_0)
            .expect("Program::create_and_build_from_source failed");

        let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

        // Create a command_queue on the Context's device
        let queue = CommandQueue::create_default_with_properties(&context, 0, 0)
            .expect("CommandQueue::create_default_with_properties failed");

        // The input data
        const ARRAY_SIZE: usize = 8;
        const VALUE_ARRAY: &str = "[3,2,5,9,7,1,4,2]";

        // Deserialize into an OpenCL SVM vector
        let mut test_values = SvmVec::<cl_int>::new(&context);

        let mut deserializer = serde_json::Deserializer::from_str(&VALUE_ARRAY);

        // Handle test_values if device only supports CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
        if !test_values.is_fine_grained() {
            // SVM_COARSE_GRAIN_BUFFER needs to know the size of the data to allocate the SVM
            test_values =
                SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");
            // Map the SVM for a SVM_COARSE_GRAIN_BUFFER
            unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut test_values, &[])? };
            // Clear the SVM for the deserializer
            test_values.clear();
        }

        ExtendSvmVec(&mut test_values)
            .deserialize(&mut deserializer)
            .expect("Error deserializing the VALUE_ARRAY JSON string.");

        // Make test_values SVM vector immutable
        let test_values = test_values;

        // Unmap test_values if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
        if !test_values.is_fine_grained() {
            let unmap_test_values_event = unsafe { queue.enqueue_svm_unmap(&test_values, &[])? };
            unmap_test_values_event.wait()?;
        }

        // The output data, an OpenCL SVM vector
        let mut results =
            SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");

        // Run the sum kernel on the input data
        let sum_kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg_svm(results.as_mut_ptr())
                .set_arg_svm(test_values.as_ptr())
                .set_global_work_size(ARRAY_SIZE)
                .enqueue_nd_range(&queue)?
        };

        // Wait for the kernel to complete execution on the device
        sum_kernel_event.wait()?;

        // Map results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
        if !results.is_fine_grained() {
            unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, &mut results, &[])? };
        }

        // Convert SVM results to json
        let json_results = serde_json::to_string(&results).unwrap();
        println!("json results: {}", json_results);

        // Unmap results if not a CL_MEM_SVM_FINE_GRAIN_BUFFER
        if !results.is_fine_grained() {
            let unmap_results_event = unsafe { queue.enqueue_svm_unmap(&results, &[])? };
            unmap_results_event.wait()?;
        }
    } else {
        println!("OpenCL fine grained system SVM device not found")
    }

    Ok(())
}
