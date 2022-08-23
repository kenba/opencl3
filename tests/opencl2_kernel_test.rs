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

#![cfg(feature = "CL_VERSION_2_0")]

extern crate opencl3;

use cl3::device::{
    CL_DEVICE_SVM_FINE_GRAIN_BUFFER, CL_DEVICE_SVM_FINE_GRAIN_SYSTEM, CL_DEVICE_TYPE_ALL,
    CL_DEVICE_TYPE_GPU,
};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{create_program_kernels, ExecuteKernel, Kernel};
use opencl3::platform::get_platforms;
use opencl3::program::{Program, CL_STD_2_0};
use opencl3::svm::SvmVec;
use opencl3::types::cl_int;
use opencl3::Result;
use std::ptr;

// The OpenCL kernels in PROGRAM_SOURCE below use built-in work-group functions:
// work_group_reduce_add, work_group_scan_inclusive_add and work_group_broadcast
// which were introduced in OpenCL 2.0.
const PROGRAM_SOURCE: &str = r#"
kernel void sum_int (global int* sums,
                    global int const* values)
{
    int value = work_group_reduce_add(values[get_global_id(0)]);

    if (0u == get_local_id(0))
        sums[get_group_id(0)] = value;
}

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

const SUM_KERNEL_NAME: &str = "sum_int";
const INCLUSIVE_SCAN_KERNEL_NAME: &str = "inclusive_scan_int";

#[test]
#[ignore]
fn test_opencl_2_kernel_example() -> Result<()> {
    let platforms = get_platforms()?;
    assert!(0 < platforms.len());

    /////////////////////////////////////////////////////////////////////
    // Query OpenCL compute environment
    let opencl_2: &str = "OpenCL 2";
    let opencl_3: &str = "OpenCL 3";

    // Find an OpenCL fine grained SVM, platform and device
    let mut device_id = ptr::null_mut();
    let mut is_fine_grained_svm: bool = false;
    for p in platforms {
        let platform_version = p.version()?;
        if platform_version.contains(&opencl_2) || platform_version.contains(&opencl_3) {
            let devices = p
                .get_devices(CL_DEVICE_TYPE_GPU)
                .expect("Platform::get_devices failed");

            for dev_id in devices {
                let device = Device::new(dev_id);
                let svm_mem_capability = device.svm_mem_capability();
                is_fine_grained_svm = 0 < svm_mem_capability & CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
                if is_fine_grained_svm {
                    device_id = dev_id;
                    break;
                }
            }
        }
    }

    if is_fine_grained_svm {
        // Create OpenCL context from the OpenCL svm device
        let device = Device::new(device_id);
        let vendor = device.vendor()?;
        let vendor_id = device.vendor_id()?;
        println!("OpenCL device vendor name: {}", vendor);
        println!("OpenCL device vendor id: {:X}", vendor_id);

        /////////////////////////////////////////////////////////////////////
        // Initialise OpenCL compute environment

        // Create a Context on the OpenCL device
        let context = Context::from_device(&device).expect("Context::from_device failed");

        // Build the OpenCL program source.
        let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_STD_2_0)
            .expect("Program::create_and_build_from_source failed");

        // Create the kernels from the OpenCL program source.
        let kernels = create_program_kernels(&program)?;
        assert!(0 < kernels.len());

        let kernel_0_name = kernels[0].function_name()?;
        println!("OpenCL kernel_0_name: {}", kernel_0_name);

        let sum_kernel = if SUM_KERNEL_NAME == kernel_0_name {
            &kernels[0]
        } else {
            &kernels[1]
        };

        let inclusive_scan_kernel = if INCLUSIVE_SCAN_KERNEL_NAME == kernel_0_name {
            &kernels[0]
        } else {
            &kernels[1]
        };

        // Create a command_queue on the Context's device
        let queue = CommandQueue::create_default_with_properties(&context, 0, 0)
            .expect("CommandQueue::create_with_properties failed");

        // Get the svm capability of all the devices in the context.
        let svm_capability = context.get_svm_mem_capability();
        assert!(0 < svm_capability);

        // Create SVM vectors for the input and output data

        // The input data
        const ARRAY_SIZE: usize = 8;
        let value_array: [cl_int; ARRAY_SIZE] = [3, 2, 5, 9, 7, 1, 4, 2];

        // Copy into an OpenCL SVM vector
        let mut test_values =
            SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");
        test_values.copy_from_slice(&value_array);

        // Make test_values immutable
        let test_values = test_values;

        // The output data, an OpenCL SVM vector
        let mut results =
            SvmVec::<cl_int>::allocate_zeroed(&context, ARRAY_SIZE).expect("SVM allocation failed");

        // Run the sum kernel on the input data
        let sum_kernel_event = unsafe {
            ExecuteKernel::new(sum_kernel)
                .set_arg_svm(results.as_mut_ptr())
                .set_arg_svm(test_values.as_ptr())
                .set_global_work_size(ARRAY_SIZE)
                .enqueue_nd_range(&queue)?
        };

        // Wait for the kernel to complete execution on the device
        sum_kernel_event.wait()?;

        // Can access OpenCL SVM directly, no need to map or read the results
        println!("sum results: {:?}", results);
        assert_eq!(33, results[0]);
        assert_eq!(0, results[ARRAY_SIZE - 1]);

        // Run the inclusive scan kernel on the input data
        let kernel_event = unsafe {
            ExecuteKernel::new(inclusive_scan_kernel)
                .set_arg_svm(results.as_mut_ptr())
                .set_arg_svm(test_values.as_ptr())
                .set_global_work_size(ARRAY_SIZE)
                .enqueue_nd_range(&queue)?
        };

        kernel_event.wait()?;

        println!("inclusive_scan results: {:?}", results);
        assert_eq!(value_array[0], results[0]);
        assert_eq!(33, results[ARRAY_SIZE - 1]);
    } else {
        println!("OpenCL fine grained SVM capable device not found");
    }

    Ok(())
}

#[test]
#[ignore]
fn test_opencl_2_system_svm_example() -> Result<()> {
    let platforms = get_platforms()?;
    assert!(0 < platforms.len());

    /////////////////////////////////////////////////////////////////////
    // Query OpenCL compute environment
    let opencl_2: &str = "OpenCL 2";
    let opencl_3: &str = "OpenCL 3";

    // Find an OpenCL fine grained SVM, platform and device
    let mut device_id = ptr::null_mut();
    let mut is_fine_grained_system_svm: bool = false;
    for p in platforms {
        let platform_version = p.version()?;

        if platform_version.contains(&opencl_2) || platform_version.contains(&opencl_3) {
            let devices = p
                .get_devices(CL_DEVICE_TYPE_ALL)
                .expect("Platform::get_devices failed");

            for dev_id in devices {
                let device = Device::new(dev_id);
                let svm_mem_capability = device.svm_mem_capability();
                is_fine_grained_system_svm =
                    0 < svm_mem_capability & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;
                if is_fine_grained_system_svm {
                    device_id = dev_id;
                    break;
                }
            }
        }
    }

    if is_fine_grained_system_svm {
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

        let kernel = Kernel::create(&program, SUM_KERNEL_NAME).expect("Kernel::create failed");

        // Create a command_queue on the Context's device
        let queue = CommandQueue::create_default_with_properties(&context, 0, 0)
            .expect("CommandQueue::create_default_with_properties failed");

        // The input data
        const ARRAY_SIZE: usize = 8;
        let value_array: [cl_int; ARRAY_SIZE] = [3, 2, 5, 9, 7, 1, 4, 2];

        // Copy into an OpenCL SVM vector
        let mut test_values =
            SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE).expect("SVM allocation failed");
        test_values.copy_from_slice(&value_array);

        // Make test_values immutable
        let test_values = test_values;

        // The output data, an OpenCL SVM vector
        let mut results =
            SvmVec::<cl_int>::allocate_zeroed(&context, ARRAY_SIZE).expect("SVM allocation failed");

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

        // Can access OpenCL SVM directly, no need to map or read the results
        println!("sum results: {:?}", results);
        assert_eq!(33, results[0]);
        assert_eq!(0, results[ARRAY_SIZE - 1]);
    } else {
        println!("OpenCL fine grained system SVM device not found")
    }

    Ok(())
}
