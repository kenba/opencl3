// Copyright (c) 2020-2022 Via Technology Ltd.
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

pub use cl3::context;

use super::device::Device;
#[cfg(feature = "CL_VERSION_1_2")]
use super::device::SubDevice;
use super::Result;

#[allow(unused_imports)]
use cl3::dx9_media_sharing;

#[cfg(feature = "cl_khr_d3d10_sharing")]
use cl3::d3d10;

#[cfg(feature = "cl_khr_d3d11_sharing")]
use cl3::d3d11;

#[allow(unused_imports)]
use cl3::egl;
#[allow(unused_imports)]
use cl3::ext;
#[allow(unused_imports)]
use cl3::gl;
#[allow(unused_imports)]
use cl3::types::{
    cl_context, cl_context_info, cl_context_properties, cl_device_id, cl_device_svm_capabilities,
    cl_device_type, cl_event, cl_image_format, cl_mem, cl_mem_flags, cl_mem_object_type, cl_uint,
};
use libc::{c_char, c_void, intptr_t, size_t};
use std::ptr;

/// Get the current device used by an OpenGL context.
///
/// * `properties` - the OpenCL context properties.
///
/// returns a Result containing the device
/// or the error code from the OpenCL C API function.
#[cfg(feature = "cl_khr_gl_sharing")]
pub fn get_current_device_for_gl_context_khr(
    properties: &[cl_context_properties],
) -> Result<cl_device_id> {
    let device = intptr_t::from(gl::get_gl_context_info_khr(
        properties.as_ptr() as *mut cl_context_properties,
        gl::GlCL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
    )?) as cl_device_id;
    Ok(device)
}

/// Get the devices for an OpenGL context.
///
/// * `properties` - the OpenCL context properties.
///
/// returns a Result containing the devices
/// or the error code from the OpenCL C API function.
#[cfg(feature = "cl_khr_gl_sharing")]
pub fn get_devices_for_gl_context_khr(
    properties: &[cl_context_properties],
) -> Result<Vec<cl_device_id>> {
    let dev_ptrs: Vec<intptr_t> = gl::get_gl_context_info_khr(
        properties.as_ptr() as *mut cl_context_properties,
        gl::GlCL_DEVICES_FOR_GL_CONTEXT_KHR,
    )?
    .into();
    let devices = dev_ptrs
        .iter()
        .map(|ptr| *ptr as cl_device_id)
        .collect::<Vec<cl_device_id>>();
    Ok(devices)
}

/// An OpenCL context object.
/// Implements the Drop trait to call release_context when the object is dropped.
#[derive(Debug)]
pub struct Context {
    context: cl_context,
    devices: Vec<cl_device_id>,
}

impl From<Context> for cl_context {
    fn from(value: Context) -> Self {
        value.context
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.devices.clear();
        unsafe { context::release_context(self.context).expect("Error: clReleaseContext") };
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

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

    /// Create a Context from a slice of cl_device_ids.  
    ///
    /// * `devices` - a slice of cl_device_ids for an OpenCL Platform.
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
        pfn_notify: Option<unsafe extern "C" fn(*const c_char, *const c_void, size_t, *mut c_void)>,
        user_data: *mut c_void,
    ) -> Result<Context> {
        let properties_ptr = if !properties.is_empty() {
            properties.as_ptr()
        } else {
            ptr::null()
        };
        let context = context::create_context(devices, properties_ptr, pfn_notify, user_data)?;
        Ok(Context::new(context, devices))
    }

    /// Create a Context from a [Device].
    ///
    /// * `device` - a [Device].
    ///
    /// returns a Result containing the new OpenCL context
    /// or the error code from the OpenCL C API function.
    pub fn from_device(device: &Device) -> Result<Context> {
        let devices: Vec<cl_device_id> = vec![device.id()];
        let properties = Vec::<cl_context_properties>::default();
        Context::from_devices(&devices, &properties, None, ptr::null_mut())
    }

    /// Create a Context from a slice of SubDevices.  
    ///
    /// * `devices` - a slice of SubDevices for an OpenCL Platform.
    /// * `properties` - a null terminated list of cl_context_properties, see
    /// [Context Properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#context-properties-table).
    /// * `pfn_notify` - an optional callback function that can be registered by the application.
    /// * `user_data` - passed as the user_data argument when pfn_notify is called.
    ///
    /// returns a Result containing the new OpenCL context
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_1_2")]
    pub fn from_sub_devices(
        sub_devices: &[SubDevice],
        properties: &[cl_context_properties],
        pfn_notify: Option<unsafe extern "C" fn(*const c_char, *const c_void, size_t, *mut c_void)>,
        user_data: *mut c_void,
    ) -> Result<Context> {
        let devices = sub_devices
            .iter()
            .map(|dev| dev.id())
            .collect::<Vec<cl_device_id>>();
        Context::from_devices(&devices, properties, pfn_notify, user_data)
    }

    /// Create a Context from a cl_device_type.  
    ///
    /// * `device_type` - the cl_device_type to create a Context for.
    /// * `properties` - a null terminated list of cl_context_properties, see
    /// [Context Properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#context-properties-table).
    /// * `pfn_notify` - an optional callback function that can be registered by the application.
    /// * `user_data` - passed as the user_data argument when pfn_notify is called.
    ///
    /// returns a Result containing the new OpenCL context
    /// or the error code from the OpenCL C API function.
    pub fn from_device_type(
        device_type: cl_device_type,
        properties: &[cl_context_properties],
        pfn_notify: Option<unsafe extern "C" fn(*const c_char, *const c_void, size_t, *mut c_void)>,
        user_data: *mut c_void,
    ) -> Result<Context> {
        let properties_ptr = if !properties.is_empty() {
            properties.as_ptr()
        } else {
            ptr::null()
        };
        let context =
            context::create_context_from_type(device_type, properties_ptr, pfn_notify, user_data)?;
        let dev_ptrs: Vec<intptr_t> =
            context::get_context_info(context, context::CL_CONTEXT_DEVICES)?.into();
        let devices = dev_ptrs
            .iter()
            .map(|ptr| *ptr as cl_device_id)
            .collect::<Vec<cl_device_id>>();
        Ok(Context::new(context, &devices))
    }

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

    #[cfg(feature = "cl_arm_import_memory")]
    pub fn import_memory_arm(
        &self,
        flags: cl_mem_flags,
        properties: *const ext::cl_import_properties_arm,
        memory: *mut c_void,
        size: size_t,
    ) -> Result<cl_mem> {
        Ok(ext::import_memory_arm(
            self.context,
            flags,
            properties,
            memory,
            size,
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
        pfn_notify: Option<unsafe extern "C" fn(cl_context, *mut c_void)>,
        user_data: *mut c_void,
    ) -> Result<()> {
        context::set_context_destructor_callback(self.context, pfn_notify, user_data)
            .map_err(Into::into)
    }

    pub fn reference_count(&self) -> Result<cl_uint> {
        Ok(context::get_context_info(self.context, context::CL_CONTEXT_REFERENCE_COUNT)?.into())
    }

    pub fn properties(&self) -> Result<Vec<intptr_t>> {
        Ok(context::get_context_info(self.context, context::CL_CONTEXT_PROPERTIES)?.into())
    }

    /// Get data about an OpenCL context.
    /// Calls clGetContextInfo to get the desired data about the context.
    pub fn get_data(&self, param_name: cl_context_info) -> Result<Vec<u8>> {
        Ok(context::get_context_data(self.context, param_name)?)
    }

    #[cfg(feature = "cl_khr_terminate_context")]
    pub fn terminate(&self) -> Result<()> {
        Ok(ext::terminate_context_khr(self.context)?)
    }

    /// Create a cl_event linked to an OpenGL sync object.  
    /// Requires the cl_khr_gl_event extension
    ///
    /// * `sync` - the sync object in the GL share group associated with context.  
    ///
    /// returns a Result containing the new OpenCL event
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "cl_khr_gl_sharing")]
    pub fn create_event_from_gl_sync_khr(&self, sync: gl::gl_sync) -> Result<cl_event> {
        Ok(gl::create_event_from_gl_sync_khr(self.context, sync)?)
    }

    /// Create an event object linked to an EGL fence sync object.  
    /// Requires the cl_khr_egl_event extension
    ///
    /// * `sync` - the handle to an EGLSync object.  
    /// * `display` - the handle to an EGLDisplay.  
    ///
    /// returns a Result containing the new OpenCL event
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "cl_khr_egl_event")]
    pub fn create_event_from_egl_sync_khr(
        &self,
        sync: egl::CLeglSyncKHR,
        display: egl::CLeglDisplayKHR,
    ) -> Result<cl_event> {
        Ok(egl::create_event_from_egl_sync_khr(
            self.context,
            sync,
            display,
        )?)
    }

    #[cfg(feature = "cl_khr_semaphore")]
    pub fn create_semaphore_with_properties_khr(
        &self,
        sema_props: *const ext::cl_semaphore_properties_khr,
    ) -> Result<cl_mem> {
        Ok(ext::create_semaphore_with_properties_khr(
            self.context,
            sema_props,
        )?)
    }

    #[cfg(any(
        feature = "cl_khr_dx9_media_sharing",
        feature = "cl_intel_dx9_media_sharing"
    ))]
    pub fn get_supported_dx9_media_surface_formats_intel(
        &self,
        flags: cl_mem_flags,
        image_type: cl_mem_object_type,
        plane: cl_uint,
    ) -> Result<Vec<cl_uint>> {
        Ok(
            dx9_media_sharing::get_supported_dx9_media_surface_formats_intel(
                self.context,
                flags,
                image_type,
                plane,
            )?,
        )
    }

    #[cfg(feature = "cl_khr_d3d10_sharing")]
    pub fn get_supported_d3d10_texture_formats_intel(
        &self,
        flags: cl_mem_flags,
        image_type: cl_mem_object_type,
    ) -> Result<Vec<cl_uint>> {
        Ok(d3d10::get_supported_d3d10_texture_formats_intel(
            self.context,
            flags,
            image_type,
        )?)
    }

    #[cfg(feature = "cl_khr_d3d11_sharing")]
    pub fn get_supported_d3d11_texture_formats_intel(
        &self,
        flags: cl_mem_flags,
        image_type: cl_mem_object_type,
        plane: cl_uint,
    ) -> Result<Vec<cl_uint>> {
        Ok(d3d11::get_supported_d3d11_texture_formats_intel(
            self.context,
            flags,
            image_type,
            plane,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::platform::get_platforms;
    use cl3::device::CL_DEVICE_TYPE_GPU;
    use cl3::info_type::InfoType;
    use cl3::memory::{CL_MEM_OBJECT_IMAGE2D, CL_MEM_READ_WRITE};

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
        let context = Context::from_device(&device).unwrap();

        println!(
            "CL_DEVICE_SVM_CAPABILITIES: {:X}",
            context.get_svm_mem_capability()
        );

        println!(
            "clGetSupportedImageFormats:\norder: data_type {}",
            InfoType::VecImageFormat(
                context
                    .get_supported_image_formats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D)
                    .unwrap()
            )
        );

        println!(
            "CL_CONTEXT_REFERENCE_COUNT: {}",
            context.reference_count().unwrap()
        );

        println!("CL_CONTEXT_PROPERTIES: {:?}", context.properties().unwrap());
    }

    #[test]
    fn test_context_from_device_type() {
        let properties = Vec::<cl_context_properties>::default();
        let context =
            Context::from_device_type(CL_DEVICE_TYPE_GPU, &properties, None, ptr::null_mut());

        match context {
            Ok(value) => {
                println!("Context num devices: {}", value.num_devices())
            }
            Err(e) => println!("OpenCL error, Context::from_device_type: {}", e),
        }
    }
}
