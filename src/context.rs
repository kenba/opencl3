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

use super::command_queue::CommandQueue;
use super::device::{
    Device, SubDevice, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
};
use super::kernel::Kernel;
use super::memory::get_supported_image_formats;
use super::program::Program;
use super::svm::SvmVec;

use cl3::context;
use cl3::types::{
    cl_command_queue_properties, cl_context, cl_context_properties, cl_device_id,
    cl_device_partition_property, cl_device_svm_capabilities, cl_image_format, cl_int,
    cl_mem_flags, cl_mem_object_type, cl_uint,
};
use libc::{c_char, c_void, intptr_t, size_t};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ptr;

/// An OpenCL context.  
/// A Context manages OpenCL objects that are constructed from it, i.e.:
/// * [CommandQueue]s
/// * [SubDevice]s
/// * [Program]s
/// * [Kernel]s
/// * [Sampler]s
///
/// It implements the Drop trait so that the OpenCL objects are released when
/// the Context goes out of scope.
pub struct Context {
    context: cl_context,
    devices: Vec<cl_device_id>,
    sub_devices: Vec<SubDevice>,
    queues: Vec<CommandQueue>,
    programs: Vec<Program>,
    kernels: HashMap<CString, Kernel>,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.kernels.clear();
        self.programs.clear();
        self.queues.clear();
        self.sub_devices.clear();
        self.devices.clear();
        context::release_context(self.context).unwrap();
        self.context = ptr::null_mut();
    }
}

impl Context {
    fn new(context: cl_context, devices: Vec<cl_device_id>) -> Context {
        Context {
            context,
            devices,
            sub_devices: Vec::<SubDevice>::default(),
            queues: Vec::<CommandQueue>::default(),
            programs: Vec::<Program>::default(),
            kernels: HashMap::<CString, Kernel>::default(),
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
        devices: Vec<cl_device_id>,
        properties: *const cl_context_properties,
        pfn_notify: Option<extern "C" fn(*const c_char, *const c_void, size_t, *mut c_void)>,
        user_data: *mut c_void,
    ) -> Result<Context, cl_int> {
        let context = context::create_context(&devices, properties, pfn_notify, user_data)?;
        Ok(Context::new(context, devices))
    }

    /// Create a Context from a [Device].
    ///
    /// * `device` - a [Device].
    ///
    /// returns a Result containing the new OpenCL context
    /// or the error code from the OpenCL C API function.
    pub fn from_device(device: Device) -> Result<Context, cl_int> {
        let devices: Vec<cl_device_id> = vec![device.id()];
        Context::from_devices(devices, ptr::null(), None, ptr::null_mut())
    }

    /// Create sub-devices from the device.
    /// Stores the sub-devices in the sub_devices vector.
    ///
    /// * `device` - the cl_device_id.
    /// * `properties` - the slice of cl_device_partition_property, see
    /// [Subdevice Partition](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#subdevice-partition-table).
    ///
    /// returns a Result containing the number of sub-devices created
    /// or the error code from the OpenCL C API function.
    pub fn create_sub_devices(
        &mut self,
        device: cl_device_id,
        properties: &[cl_device_partition_property],
    ) -> Result<usize, cl_int> {
        let device = Device::new(device);
        let sub_devs = device.create_sub_devices(properties)?;
        let count = sub_devs.len();
        for device_id in sub_devs {
            self.sub_devices.push(SubDevice::new(device_id));
        }
        Ok(count)
    }

    /// Add a [CommandQueue] to the Context for it to manage.
    ///
    /// * `queue` - a command queue on one of the devices or sub-devices
    /// managed by the Context.
    pub fn add_command_queue(&mut self, queue: CommandQueue) {
        self.queues.push(queue);
    }

    /// Create a [CommandQueue] for every device and append them to the queues
    /// managed by this context.
    /// Deprecated in CL_VERSION_2_0 by create_command_queue_with_properties.
    ///
    /// * `properties` - a list of properties for the command-queue, see
    /// [cl_command_queue_properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#legacy-queue-properties-table).
    ///
    /// returns an empty Result
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_1_2")]
    pub fn create_command_queues(
        &mut self,
        properties: cl_command_queue_properties,
    ) -> Result<(), cl_int> {
        for index in 0..self.devices.len() {
            let device = self.devices[index];
            let queue = CommandQueue::create(self.context, device, properties)?;
            self.add_command_queue(queue);
        }
        Ok(())
    }

    /// Create a [CommandQueue] for every device and append them to the queues
    /// managed by this context.
    /// CL_VERSION_2_0 onwards.
    ///
    /// * `properties` - a null terminated list of properties for the command-queue, see
    /// [cl_queue_properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#queue-properties-table).
    ///
    /// returns an empty Result
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_2_0")]
    pub fn create_command_queues_with_properties(
        &mut self,
        properties: cl_command_queue_properties,
        queue_size: cl_uint,
    ) -> Result<(), cl_int> {
        for index in 0..self.devices.len() {
            let device = self.devices[index];
            let queue =
                CommandQueue::create_with_properties(self.context, device, properties, queue_size)?;
            self.add_command_queue(queue);
        }
        Ok(())
    }

    /// Add a built [Program] to the Context for it to manage.
    /// It also creates and manages the [Kernel]s in the program.
    ///
    /// * `program` - a [Program] and its [Kernel]s to be  managed by the Context.
    ///
    /// returns a Result containing the number of kernels in the Program.
    /// or the error code from the OpenCL C API function.
    pub fn add_program(&mut self, program: Program) -> Result<usize, cl_int> {
        let kernels = program.create_kernels_in_program()?;
        let count = kernels.len();
        for kernel in kernels {
            let kernel = Kernel::new(kernel)?;
            let name = kernel.function_name()?;
            self.kernels.insert(name, kernel);
        }
        self.programs.push(program);
        Ok(count)
    }

    /// Create and build a Program from source code.  
    ///
    /// * `src` - a CStr containing the source code character string.
    /// * `options` - the build options in a null-terminated string.
    ///
    /// returns a Result containing the number of kernels in the Program.
    /// or the error code from the OpenCL C API function.
    pub fn build_program_from_source(
        &mut self,
        src: &CStr,
        options: &CStr,
    ) -> Result<usize, cl_int> {
        let src_array = [src];
        let program = Program::create_from_source(self.context, &src_array)?;
        program.build(&self.devices, &options)?;
        self.add_program(program)
    }

    /// Create and build a Program from binaries.
    ///
    /// * `src` - a CStr containing the source code character string.
    /// * `binaries` - a slice of program binaries slices.
    /// * `options` - the build options in a null-terminated string.
    ///
    /// returns a Result containing the number of kernels in the Program.
    /// or the error code from the OpenCL C API function.
    pub fn build_program_from_binary(
        &mut self,
        binaries: &[&[u8]],
        options: &CStr,
    ) -> Result<usize, cl_int> {
        let program = Program::create_from_binary(self.context, &self.devices, binaries)?;
        program.build(&self.devices, &options)?;
        self.add_program(program)
    }

    /// Get the kernel with the given name.
    ///
    /// * `kernel_name` -  the name of the [Kernel]
    ///
    /// returns an Option containing a reference to the [Kernel] on None.
    pub fn get_kernel(&self, kernel_name: &CStr) -> Option<&Kernel> {
        self.kernels.get::<CStr>(&kernel_name)
    }

    pub fn get_svm_mem_capability(&self) -> cl_device_svm_capabilities {
        let device = Device::new(self.devices[0]);
        let mut svm_capability = device.svm_mem_capability();

        for index in 1..self.devices.len() {
            let device = Device::new(self.devices[index]);
            svm_capability &= device.svm_mem_capability();
        }

        svm_capability
    }

    pub fn create_svm_vec<T>(&self, svm_capability: cl_device_svm_capabilities) -> SvmVec<T> {
        assert!(
            0 < svm_capability
                & (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER),
            "create_svm_vec: devices not not support SVM buffers"
        );
        SvmVec::new(&self, svm_capability)
    }

    pub fn get_supported_image_formats(
        &self,
        flags: cl_mem_flags,
        image_type: cl_mem_object_type,
    ) -> Result<Vec<cl_image_format>, cl_int> {
        get_supported_image_formats(self.context, flags, image_type)
    }

    #[cfg(feature = "CL_VERSION_2_1")]
    #[inline]
    pub fn set_default_device_command_queue(
        &self,
        device: cl_device_id,
        queue: CommandQueue,
    ) -> Result<(), cl_int> {
        set_default_device_command_queue(self.context, device, queue.get())
    }

    // references devices
    pub fn devices(&self) -> &[cl_device_id] {
        &self.devices
    }

    pub fn sub_devices(&self) -> &[SubDevice] {
        &self.sub_devices
    }

    // references queues
    pub fn queues(&self) -> &[CommandQueue] {
        &self.queues
    }

    pub fn default_queue(&self) -> &CommandQueue {
        &self.queues[0]
    }

    pub fn programs(&self) -> &[Program] {
        &self.programs
    }

    pub fn kernels(&self) -> &HashMap<CString, Kernel> {
        &self.kernels
    }

    #[cfg(feature = "CL_VERSION_3_0")]
    #[inline]
    pub fn set_destructor_callback(
        &self,
        pfn_notify: extern "C" fn(cl_context, *const c_void),
        user_data: *mut c_void,
    ) -> Result<(), cl_int> {
        set_context_destructor_callback(self.context, pfn_notify, user_data)
    }

    pub fn reference_count(&self) -> Result<cl_uint, cl_int> {
        Ok(context::get_context_info(
            self.context,
            context::ContextInfo::CL_CONTEXT_REFERENCE_COUNT,
        )?
        .to_uint())
    }

    pub fn properties(&self) -> Result<Vec<intptr_t>, cl_int> {
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
            "CL_CONTEXT_REFERENCE_COUNT: {}",
            context.reference_count().unwrap()
        );
    }
}
