// Copyright (c) 2021-2023 Via Technology Ltd. All Rights Reserved.
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

use cl3::ext::{
    CL_DEVICE_IMAGE_SUPPORT, CL_DEVICE_MAX_READ_IMAGE_ARGS, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
    CL_DEVICE_MAX_SAMPLERS, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, CL_IMAGE_FORMAT_NOT_SUPPORTED,
};
use cl3::memory::{
    CL_MEM_OBJECT_IMAGE2D, CL_MEM_WRITE_ONLY, CL_RGBA, CL_UNSIGNED_INT8,
};
use cl3::types::{cl_image_desc, cl_image_format, CL_NON_BLOCKING};
use libc::c_void;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::Image;
use opencl3::program::{Program, CL_STD_2_0};
use opencl3::types::cl_event;
use opencl3::Result;

const PROGRAM_SOURCE: &str = r#"
kernel void colorize(write_only image2d_t image)
{
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    write_imageui(image, (int2)(x, y), (uint4)(x, y, 0, 255));
}"#;

const KERNEL_NAME: &str = "colorize";

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

    // Print some information about the device
    let result = device.get_data(CL_DEVICE_IMAGE_SUPPORT)?;
    println!(
        "CL_DEVICE_IMAGE_SUPPORT: {:?}",
        ((result[3] as u32) << 24)
            | ((result[2] as u32) << 16)
            | ((result[1] as u32) << 8)
            | (result[0] as u32)
    );
    let result = device.get_data(CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS)?;
    println!(
        "CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: {:?}",
        ((result[3] as u32) << 24)
            | ((result[2] as u32) << 16)
            | ((result[1] as u32) << 8)
            | (result[0] as u32)
    );
    let result = device.get_data(CL_DEVICE_MAX_READ_IMAGE_ARGS)?;
    println!(
        "CL_DEVICE_MAX_READ_IMAGE_ARGS: {:?}",
        ((result[3] as u32) << 24)
            | ((result[2] as u32) << 16)
            | ((result[1] as u32) << 8)
            | (result[0] as u32)
    );
    let result = device.get_data(CL_DEVICE_MAX_WRITE_IMAGE_ARGS)?;
    println!(
        "CL_DEVICE_MAX_WRITE_IMAGE_ARGS: {:?}",
        ((result[3] as u32) << 24)
            | ((result[2] as u32) << 16)
            | ((result[1] as u32) << 8)
            | (result[0] as u32)
    );
    let result = device.get_data(CL_DEVICE_MAX_SAMPLERS)?;
    println!(
        "CL_DEVICE_MAX_SAMPLERS: {:?}",
        ((result[3] as u32) << 24)
            | ((result[2] as u32) << 16)
            | ((result[1] as u32) << 8)
            | (result[0] as u32)
    );
    let supported_formats =
        context.get_supported_image_formats(CL_MEM_WRITE_ONLY, CL_MEM_OBJECT_IMAGE2D)?;
    if supported_formats
        .iter()
        .filter(|f| {
            f.image_channel_order == CL_RGBA && f.image_channel_data_type == CL_UNSIGNED_INT8
        })
        .count()
        <= 0
    {
        println!("Device does not support CL_RGBA with CL_UNSIGNED_INT8 for CL_MEM_WRITE_ONLY!");
        return Err(CL_IMAGE_FORMAT_NOT_SUPPORTED.into());
    }

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_STD_2_0)
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    // Create a command_queue on the Context's device
    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
            .expect("CommandQueue::create_default_with_properties failed");

    // Create a set of images
    let image = unsafe {
        Image::create(
            &context,
            CL_MEM_WRITE_ONLY,
            &cl_image_format {
                image_channel_order: CL_RGBA,
                image_channel_data_type: CL_UNSIGNED_INT8,
            },
            &cl_image_desc {
                image_type: CL_MEM_OBJECT_IMAGE2D,
                image_width: 10 as usize,
                image_height: 10 as usize,
                image_depth: 1,
                image_array_size: 1,
                image_row_pitch: 0,
                image_slice_pitch: 0,
                num_mip_levels: 0,
                num_samples: 0,
                buffer: std::ptr::null_mut(),
            },
            std::ptr::null_mut(),
        )
        .expect("Image::create failed")
    };

    // Run the kernel on the input data
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&image)
            .set_global_work_sizes(&[10usize, 10usize])
            .enqueue_nd_range(&queue)?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Read the image data from the device
    let mut image_data = [0u8; 10 * 10 * 4];
    let read_event = unsafe {
        queue.enqueue_read_image(
            &image,
            CL_NON_BLOCKING,
            &[0usize, 0usize, 0usize] as *const usize,
            &[10usize, 10usize, 1usize] as *const usize,
            0,
            0,
            image_data.as_mut_ptr() as *mut c_void,
            &events,
        )?
    };

    // Wait for the read_event to complete.
    read_event.wait()?;

    // Print the image data
    println!("image_data: ");
    for y in 0..10 {
        for x in 0..10 {
            let offset = (y * 10 + x) * 4;
            print!(
                "({:>3}, {:>3}, {:>3}, {:>3}) ",
                image_data[offset],
                image_data[offset + 1],
                image_data[offset + 2],
                image_data[offset + 3]
            );
        }
        println!();
    }

    Ok(())
}
