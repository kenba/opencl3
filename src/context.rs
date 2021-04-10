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

pub use cl3::context::{CL_CONTEXT_INTEROP_USER_SYNC, CL_CONTEXT_PLATFORM};

use super::device::Device;
use super::Result;

use cl3::context;
use cl3::types::{
    cl_context, cl_context_properties, cl_device_id, cl_device_svm_capabilities, cl_image_format,
    cl_mem_flags, cl_mem_object_type, cl_uint,
};
use libc::{c_char, c_void, intptr_t, size_t};
use std::ptr;

/// An OpenCL context object.
/// Implements the Drop trait to call release_context when the object is dropped.
pub struct Context {
    context: cl_context,
    devices: Vec<cl_device_id>,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.devices.clear();
        context::release_context(self.context).expect("Error: clReleaseContext");
    }
}

impl Context {
    fn new(context: cl_context, devices: &[cl_device_id]) -> Context {
        Context {
            context,
            devices: devices.to_vec(),
        }
    }

    /// Get the underlying OpenCL cl_context.
    pub fn get(&self) -> cl_context {
        self.context
    }

    /// Create a Context from a vector of cl_device_ids.  
    /// Note: the vector of cl_device_ids are moved into Context
    ///
    /// * `devices` - a vector of cl_device_ids for an OpenCL Platform.
    /// * `properties` - a null terminated list of cl_context_properties, see
    /// [Context Properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#context-properties-table).
    /// * `pfn_notify` - an optional callback function that can be registered by the application.
    /// * `user_data` - passed as the user_data argument when pfn_notify is called.
    ///
    /// returns a Result containing the new OpenCL context
    /// or the error code from the OpenCL C API function.
    pub fn from_devices(
        devices: &[cl_device_id],
        properties: &[cl_context_properties],
        pfn_notify: Option<extern "C" fn(*const c_char, *const c_void, size_t, *mut c_void)>,
        user_data: *mut c_void,
    ) -> Result<Context> {
        let properties_ptr = if 0 < properties.len() {
            properties.as_ptr()
        } else {
            ptr::null()
        };
        let context = context::create_context(&devices, properties_ptr, pfn_notify, user_data)?;
        Ok(Context::new(context, devices))
    }

    /// Create a Context from a [Device].
    ///
    /// * `device` - a [Device].
    ///
    /// returns a Result containing the new OpenCL context
    /// or the error code from the OpenCL C API function.
    pub fn from_device(device: Device) -> Result<Context> {
        let devices: Vec<cl_device_id> = vec![device.id()];
        let properties = Vec::<cl_context_properties>::default();
        Context::from_devices(&devices, &properties, None, ptr::null_mut())
    }

    // TODO from_device_type call create_context_from_type

    /// Get the common Shared Virtual Memory (SVM) capabilities of the
    /// devices in the Context.
    pub fn get_svm_mem_capability(&self) -> cl_device_svm_capabilities {
        let device = Device::new(self.devices[0]);
        let mut svm_capability = device.svm_mem_capability();

        for index in 1..self.devices.len() {
            let device = Device::new(self.devices[index]);
            svm_capability &= device.svm_mem_capability();
        }

        svm_capability
    }

    /// Get the list of image formats supported by the Context for an image type,
    /// and allocation information.  
    /// Calls clGetSupportedImageFormats to get the desired information about the context.
    ///
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `image_type` - describes the image type.
    ///
    /// returns a Result containing the desired information in an InfoType enum
    /// or the error code from the OpenCL C API function.
    pub fn get_supported_image_formats(
        &self,
        flags: cl_mem_flags,
        image_type: cl_mem_object_type,
    ) -> Result<Vec<cl_image_format>> {
        Ok(cl3::memory::get_supported_image_formats(
            self.context,
            flags,
            image_type,
        )?)
    }

    pub fn devices(&self) -> &[cl_device_id] {
        &self.devices
    }

    pub fn default_device(&self) -> cl_device_id {
        self.devices[0]
    }

    pub fn num_devices(&self) -> cl_uint {
        self.devices.len() as cl_uint
    }

    #[cfg(feature = "CL_VERSION_3_0")]
    #[inline]
    pub fn set_destructor_callback(
        &self,
        pfn_notify: extern "C" fn(cl_context, *const c_void),
        user_data: *mut c_void,
    ) -> Result<()> {
        set_context_destructor_callback(self.context, pfn_notify, user_data)
    }

    pub fn reference_count(&self) -> Result<cl_uint> {
        Ok(context::get_context_info(
            self.context,
            context::ContextInfo::CL_CONTEXT_REFERENCE_COUNT,
        )?
        .to_uint())
    }

    pub fn properties(&self) -> Result<Vec<intptr_t>> {
        Ok(
            context::get_context_info(self.context, context::ContextInfo::CL_CONTEXT_PROPERTIES)?
                .to_vec_intptr(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::platform::get_platforms;
    use cl3::device::CL_DEVICE_TYPE_GPU;
    // use cl3::memory::{CL_MEM_OBJECT_IMAGE2D, CL_MEM_READ_WRITE};

    #[test]
    fn test_context() {
        let platforms = get_platforms().unwrap();
        assert!(0 < platforms.len());

        // Get the first platform
        let platform = &platforms[0];

        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
        assert!(0 < devices.len());

        // Get the first device
        let device = Device::new(devices[0]);
        let context = Context::from_device(device).unwrap();

        println!(
            "CL_DEVICE_SVM_CAPABILITIES: {:X}",
            context.get_svm_mem_capability()
        );

        // println!(
        //     "clGetSupportedImageFormats: {:?}",
        //     context
        //         .get_supported_image_formats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D)
        //         .unwrap()
        // );

        println!(
            "CL_CONTEXT_REFERENCE_COUNT: {}",
            context.reference_count().unwrap()
        );

        println!("CL_CONTEXT_PROPERTIES: {:?}", context.properties().unwrap());
    }
}
