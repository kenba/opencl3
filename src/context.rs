// Copyright (c) 2020 Via Technology Ltd. All Rights Reserved.
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

use super::memory::{get_supported_image_formats, Buffer, Image, Pipe};
use super::command_queue::CommandQueue;
use super::device::{Device, SubDevice};
use super::kernel::Kernel;
use super::program::Program;
use super::sampler::Sampler;
use super::svm::SvmVec;

use cl3::context;
use cl3::types::{
    cl_command_queue_properties, cl_context, cl_context_properties, cl_device_id, cl_int, cl_mem,
    cl_mem_flags, cl_uint, cl_device_partition_property, cl_image_format, cl_image_desc, cl_bool,
    cl_sampler, cl_addressing_mode, cl_filter_mode, cl_sampler_properties, cl_mem_object_type,
};
use libc::{c_void, intptr_t, size_t};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::ptr;

/// An OpenCL context.  
/// The Context struct manages all the OpenCL objects in an application:
/// * [CommandQueue]s
/// * [SubDevice]s
/// * [Program]s
/// * [Kernel]s
/// * [Buffer]s
/// * [Image]s
/// * [Sampler]s
/// * [SubDevice]s
/// * [Pipe]s
pub struct Context {
    context: cl_context,
    devices: Vec<cl_device_id>,
    sub_devices: Vec<SubDevice>,
    queues: Vec<CommandQueue>,
    sub_queues: Vec<CommandQueue>,
    programs: Vec<Program>,
    kernels: HashMap<CString, Kernel>,
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    samplers: Vec<Sampler>,
    pipes: Vec<Pipe>,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.pipes.clear();
        self.samplers.clear();
        self.images.clear();
        self.buffers.clear();
        self.kernels.clear();
        self.programs.clear();
        self.sub_queues.clear();
        self.queues.clear();
        self.sub_devices.clear();
        self.devices.clear();
        context::release_context(self.context).unwrap();
        self.context = ptr::null_mut();
        // println!("Context::drop");
    }
}

impl Context {
    fn new(context: cl_context, devices: Vec<cl_device_id>) -> Context {
        Context {
            context,
            devices,
            sub_devices: Vec::<SubDevice>::default(),
            queues: Vec::<CommandQueue>::default(),
            sub_queues: Vec::<CommandQueue>::default(),
            programs: Vec::<Program>::default(),
            kernels: HashMap::<CString, Kernel>::default(),
            buffers: Vec::<Buffer>::default(),
            images: Vec::<Image>::default(),
            samplers: Vec::<Sampler>::default(),
            pipes: Vec::<Pipe>::default(),
        }
    }

    pub fn get(&self) -> cl_context {
        self.context
    }

    // Note: devices are moved into Context
    pub fn from_devices(
        devices: Vec<cl_device_id>,
        properties: *const cl_context_properties,
    ) -> Result<Context, cl_int> {
        let context = context::create_context(&devices, properties, None, ptr::null_mut())?;
        Ok(Context::new(context, devices))
    }

    pub fn from_device(device: Device) -> Result<Context, cl_int> {
        let devices: Vec<cl_device_id> = vec![device.id()];
        Context::from_devices(devices, ptr::null())
    }

    pub fn create_sub_devices(
        &mut self,
        index: usize,
        properties: &[cl_device_partition_property],
    ) -> Result<usize, cl_int> {
        assert!(index < self.devices.len());
        let device = Device::new(self.devices[index]);
        let sub_devs = device.create_sub_devices(properties)?;
        let count = sub_devs.len();
        for device_id in sub_devs {
            self.sub_devices.push(SubDevice::new(device_id));
        }
        Ok(count)
    }

    #[cfg(feature = "CL_VERSION_1_2")]
    pub fn create_command_queue(
        &mut self,
        properties: cl_command_queue_properties,
    ) -> Result<(), cl_int> {
        let index = self.queues.len();
        let device = self.devices[index];
        let queue = CommandQueue::create(self.context, device, properties)?;
        self.queues.push(queue);
        Ok(())
    }

    #[cfg(feature = "CL_VERSION_2_0")]
    pub fn create_command_queue_with_properties(
        &mut self,
        properties: cl_command_queue_properties,
        queue_size: cl_uint,
    ) -> Result<(), cl_int> {
        let index = self.queues.len();
        let device = self.devices[index];
        let queue =
            CommandQueue::create_with_properties(self.context, device, properties, queue_size)?;
        self.queues.push(queue);
        Ok(())
    }

    #[cfg(feature = "CL_VERSION_1_2")]
    pub fn create_sub_device_command_queue(
        &mut self,
        properties: cl_command_queue_properties,
    ) -> Result<(), cl_int> {
        let index = self.sub_queues.len();
        let sub_device = &self.sub_devices[index];
        let queue = CommandQueue::create(self.context, sub_device.id(), properties)?;
        self.sub_queues.push(queue);
        Ok(())
    }

    #[cfg(feature = "CL_VERSION_2_0")]
    pub fn create_sub_device_command_queue_with_properties(
        &mut self,
        properties: cl_command_queue_properties,
        queue_size: cl_uint,
    ) -> Result<(), cl_int> {
        let index = self.sub_queues.len();
        let sub_device = &self.sub_devices[index];
        let queue =
            CommandQueue::create_with_properties(self.context, sub_device.id(), properties, queue_size)?;
        self.sub_queues.push(queue);
        Ok(())
    }

    fn add_kernels(&mut self, program: &Program) -> Result<usize, cl_int> {
        let kernels = program.create_kernels_in_program()?;
        let count = kernels.len();
        for kernel in kernels {
            let kernel = Kernel::new(kernel)?;
            let name = kernel.function_name()?;
            self.kernels.insert(name, kernel);
        }
        Ok(count)
    }

    pub fn build_program_from_source(&mut self, src: &CStr, options: &CStr) -> Result<usize, cl_int> {
        let program = Program::create_from_source(self.context, src)?;
        program.build(&self.devices, &options)?;
        let count = self.add_kernels(&program)?;
        self.programs.push(program);
        Ok(count)
    }

    pub fn build_program_from_binary(
        &mut self,
        binaries: &[&[u8]],
        options: &CStr,
    ) -> Result<usize, cl_int> {
        let program = Program::create_from_binary(self.context, &self.devices, binaries)?;
        program.build(&self.devices, &options)?;
        let count = self.add_kernels(&program)?;
        self.programs.push(program);
        Ok(count)
    }

    pub fn get_kernel(&self, kernel_name: &CStr) -> Option<&Kernel> {
        self.kernels.get::<CStr>(&kernel_name)
    }

    pub fn get_kernel_string(&self, kernel_name: &str) -> Option<&Kernel> {
        let cname = CString::new(kernel_name).unwrap();
        self.kernels.get::<CStr>(&cname)
    }

    pub fn create_buffer<T>(
        &mut self,
        flags: cl_mem_flags,
        count: size_t,
        host_ptr: *mut c_void,
    ) -> Result<cl_mem, cl_int> {
        let buffer = Buffer::create::<T>(self.context, flags, count, host_ptr)?;
        let mem = buffer.get();
        self.buffers.push(buffer);
        Ok(mem)
    }

    pub fn create_svm_vec<T>(&self, capacity: usize) -> SvmVec<T> {
        let device = Device::new(self.devices[0]);
        let svm_capability = device.svm_capabilities().unwrap();
        let mut svm = SvmVec::new(&self, svm_capability);
        svm.reserve(capacity);
        svm
    }

    pub fn get_supported_image_formats(
        &self,
        flags: cl_mem_flags,
        image_type: cl_mem_object_type,
    ) -> Result<Vec<cl_image_format>, cl_int> {
        get_supported_image_formats(self.context, flags, image_type)
    }

    pub fn create_image<T>(
        &mut self,
        flags: cl_mem_flags,
        image_format: *const cl_image_format,
        image_desc: *const cl_image_desc,
        host_ptr: *mut c_void,
    ) -> Result<cl_mem, cl_int> {
        let image = Image::create::<T>(self.context, flags, image_format, image_desc, host_ptr)?;
        let mem = image.get();
        self.images.push(image);
        Ok(mem)
    }

    pub fn create_sampler<T>(
        &mut self,
        normalize_coords: cl_bool,
        addressing_mode: cl_addressing_mode,
        filter_mode: cl_filter_mode,
    ) -> Result<cl_sampler, cl_int> {
        let sampler = Sampler::create::<T>(self.context, normalize_coords, addressing_mode, filter_mode)?;
        let smplr = sampler.get();
        self.samplers.push(sampler);
        Ok(smplr)
    }

    pub fn create_sampler_with_properties<T>(
        &mut self,
        properties: *const cl_sampler_properties,
    ) -> Result<cl_sampler, cl_int> {
        let sampler = Sampler::create_with_properties::<T>(self.context, properties)?;
        let smplr = sampler.get();
        self.samplers.push(sampler);
        Ok(smplr)
    }

    #[cfg(feature = "CL_VERSION_2_0")]
    pub fn create_pipe<T>(
        &mut self,
        flags: cl_mem_flags,
        pipe_packet_size: cl_uint,
        pipe_max_packets: cl_uint,
    ) -> Result<cl_mem, cl_int> {
        let pipe = Pipe::create::<T>(self.context, flags, pipe_packet_size, pipe_max_packets)?;
        let mem = pipe.get();
        self.pipes.push(pipe);
        Ok(mem)
    }

    #[cfg(feature = "CL_VERSION_2_1")]
    #[inline]
    pub fn set_default_device_command_queue(
        &self,
        device: cl_device_id,
        queue: CommandQueue
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

    pub fn sub_queues(&self) -> &[CommandQueue] {
        &self.sub_queues
    }

    pub fn programs(&self) -> &[Program] {
        &self.programs
    }

    pub fn kernels(&self) -> &HashMap<CString, Kernel> {
        &self.kernels
    }

    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    pub fn images(&self) -> &[Image] {
        &self.images
    }

    pub fn samplers(&self) -> &[Sampler] {
        &self.samplers
    }

    pub fn pipes(&self) -> &[Pipe] {
        &self.pipes
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
