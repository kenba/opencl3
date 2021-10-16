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

use opencl3::device::{device_type_text, vendor_id_text, Device, CL_DEVICE_TYPE_ALL};
use opencl3::Result;

/// Finds all the OpenCL platforms and devices on a system.
///
/// It displays OpenCL platform information from `clGetPlatformInfo` and
/// OpenCL device information from `clGetDeviceInfo` for all the platforms and
/// devices.
fn main() -> Result<()> {
    let platforms = opencl3::platform::get_platforms()?;
    println!("Number of platforms: {}", platforms.len());

    for platform in platforms {
        println!("CL_PLATFORM_VENDOR: {}", platform.vendor()?);
        println!("CL_PLATFORM_NAME: {}", platform.name()?);
        println!("CL_PLATFORM_VERSION: {}", platform.version()?);
        println!("CL_PLATFORM_PROFILE: {}", platform.profile()?);
        println!("CL_PLATFORM_EXTENSIONS: {}", platform.extensions()?);

        let devices = platform.get_devices(CL_DEVICE_TYPE_ALL)?;
        println!("Number of devices: {}", devices.len());
        println!();

        for device_id in devices {
            let device = Device::new(device_id);
            println!("\tCL_DEVICE_VENDOR: {}", device.vendor()?);
            let vendor_id = device.vendor_id()?;
            println!(
                "\tCL_DEVICE_VENDOR_ID: {:X}, {}",
                vendor_id,
                vendor_id_text(vendor_id)
            );
            println!("\tCL_DEVICE_NAME: {}", device.name()?);
            println!("\tCL_DEVICE_VERSION: {}", device.version()?);
            let device_type = device.dev_type()?;
            println!(
                "\tCL_DEVICE_TYPE: {:X}, {}",
                device_type,
                device_type_text(device_type)
            );
            println!("\tCL_DEVICE_PROFILE: {}", device.profile()?);
            println!("\tCL_DEVICE_EXTENSIONS: {}", device.extensions()?);
            println!(
                "\tCL_DEVICE_OPENCL_C_VERSION: {:?}",
                device.opencl_c_version()?
            );
            println!(
                "\tCL_DEVICE_BUILT_IN_KERNELS: {}",
                device.built_in_kernels()?
            );
            println!(
                "\tCL_DEVICE_SVM_CAPABILITIES: {:X}",
                device.svm_mem_capability()
            );
            println!();
        }
    }

    Ok(())
}
