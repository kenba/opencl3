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

pub use cl3::kernel::*;

use super::command_queue::CommandQueue;
use super::event::Event;
use super::program::Program;
use super::Result;

#[allow(unused_imports)]
use cl3::ext;
use cl3::types::{
    cl_context, cl_device_id, cl_event, cl_kernel, cl_kernel_exec_info, cl_program, cl_uint,
    cl_ulong,
};
use libc::{c_void, size_t};
use std::ffi::CString;
use std::mem;
use std::ptr;

/// An OpenCL kernel object.  
/// Implements the Drop trait to call release_kernel when the object is dropped.
#[derive(Debug)]
pub struct Kernel {
    kernel: cl_kernel,
}

#[cfg(feature = "CL_VERSION_2_1")]
impl Clone for Kernel {
    /// Clone an OpenCL kernel object.  
    /// CL_VERSION_2_1 see: [Copying Kernel Objects](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_copying_kernel_objects)
    ///
    /// returns a Result containing the new Kernel
    /// or the error code from the OpenCL C API function.
    fn clone(&self) -> Self {
        let kernel = clone_kernel(self.kernel).expect("Error: clCloneKernel");
        Kernel { kernel }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        release_kernel(self.kernel).expect("Error: clReleaseKernel");
    }
}

unsafe impl Send for Kernel {}

impl Kernel {
    /// Create a Kernel from an OpenCL cl_kernel.
    ///
    /// * `kernel` - a valid OpenCL cl_kernel.
    ///
    /// returns a Result containing the new Kernel
    /// or the error code from the OpenCL C API function to get the number
    /// of kernel arguments.
    pub fn new(kernel: cl_kernel) -> Kernel {
        Kernel { kernel }
    }

    /// Get the underlying OpenCL cl_kernel.
    pub fn get(&self) -> cl_kernel {
        self.kernel
    }

    /// Create a Kernel from an OpenCL Program.
    ///
    /// * `program` - a built OpenCL Program.
    /// * `name` - the name of the OpenCL kernel.
    ///
    /// returns a Result containing the new Kernel
    /// or the error code from the OpenCL C API function to get the number
    /// of kernel arguments.
    pub fn create(program: &Program, name: &str) -> Result<Kernel> {
        // Ensure c_name string is null terminated
        let c_name = CString::new(name).expect("Kernel::create, invalid name");
        Ok(Self::new(create_kernel(program.get(), &c_name)?))
    }

    /// Set the argument value for a specific argument of a kernel.  
    ///
    /// * `arg_index` - the kernel argument index.
    /// * `arg` - a reference to the data for the argument at arg_index.
    ///
    /// returns an empty Result or the error code from the OpenCL C API function.
    pub fn set_arg<T>(&self, arg_index: cl_uint, arg: &T) -> Result<()> {
        Ok(set_kernel_arg(
            self.kernel,
            arg_index,
            mem::size_of::<T>(),
            arg as *const _ as *const c_void,
        )?)
    }

    /// Create a local memory buffer for a specific argument of a kernel.  
    ///
    /// * `arg_index` - the kernel argument index.
    /// * `size` - the size of the local memory buffer in bytes.
    ///
    /// returns an empty Result or the error code from the OpenCL C API function.
    pub fn set_arg_local_buffer(&self, arg_index: cl_uint, size: size_t) -> Result<()> {
        Ok(set_kernel_arg(self.kernel, arg_index, size, ptr::null())?)
    }

    /// Set set a SVM pointer as the argument value for a specific argument of a kernel.  
    ///
    /// * `arg_index` - the kernel argument index.
    /// * `arg_ptr` - the SVM pointer to the data for the argument at arg_index.
    ///
    /// returns an empty Result or the error code from the OpenCL C API function.
    pub fn set_arg_svm_pointer(&self, arg_index: cl_uint, arg_ptr: *const c_void) -> Result<()> {
        Ok(set_kernel_arg_svm_pointer(self.kernel, arg_index, arg_ptr)?)
    }

    /// Pass additional information other than argument values to a kernel.  
    ///
    /// * `param_name` - the information to be passed to kernel, see:
    /// [Kernel Execution Properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#kernel-exec-info-table).
    /// * `param_ptr` - pointer to the data for the param_name.
    ///
    /// returns an empty Result or the error code from the OpenCL C API function.
    pub fn set_exec_info<T>(
        &self,
        param_name: cl_kernel_exec_info,
        param_ptr: *const T,
    ) -> Result<()> {
        Ok(set_kernel_exec_info(
            self.kernel,
            param_name,
            mem::size_of::<T>(),
            param_ptr as *const c_void,
        )?)
    }

    pub fn function_name(&self) -> Result<String> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_FUNCTION_NAME)?.to_string())
    }

    pub fn num_args(&self) -> Result<cl_uint> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_NUM_ARGS)?.to_uint())
    }

    pub fn reference_count(&self) -> Result<cl_uint> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_REFERENCE_COUNT)?.to_uint())
    }

    pub fn context(&self) -> Result<cl_context> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_CONTEXT)?.to_ptr() as cl_context)
    }

    pub fn program(&self) -> Result<cl_program> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_PROGRAM)?.to_ptr() as cl_program)
    }

    pub fn attributes(&self) -> Result<String> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_ATTRIBUTES)?.to_string())
    }

    pub fn get_arg_address_qualifier(&self, arg_indx: cl_uint) -> Result<cl_uint> {
        Ok(get_kernel_arg_info(
            self.kernel,
            arg_indx,
            KernelArgInfo::CL_KERNEL_ARG_ADDRESS_QUALIFIER,
        )?
        .to_uint())
    }

    pub fn get_arg_access_qualifier(&self, arg_indx: cl_uint) -> Result<cl_uint> {
        Ok(get_kernel_arg_info(
            self.kernel,
            arg_indx,
            KernelArgInfo::CL_KERNEL_ARG_ACCESS_QUALIFIER,
        )?
        .to_uint())
    }

    pub fn get_arg_type_qualifier(&self, arg_indx: cl_uint) -> Result<cl_ulong> {
        Ok(get_kernel_arg_info(
            self.kernel,
            arg_indx,
            KernelArgInfo::CL_KERNEL_ARG_TYPE_QUALIFIER,
        )?
        .to_ulong())
    }

    pub fn get_arg_type_name(&self, arg_indx: cl_uint) -> Result<String> {
        Ok(get_kernel_arg_info(
            self.kernel,
            arg_indx,
            KernelArgInfo::CL_KERNEL_ARG_TYPE_NAME,
        )?
        .to_string())
    }

    pub fn get_arg_name(&self, arg_indx: cl_uint) -> Result<String> {
        Ok(
            get_kernel_arg_info(self.kernel, arg_indx, KernelArgInfo::CL_KERNEL_ARG_NAME)?
                .to_string(),
        )
    }

    pub fn get_work_group_size(&self, device: cl_device_id) -> Result<size_t> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_WORK_GROUP_SIZE,
        )?
        .to_size())
    }

    pub fn get_compile_work_group_size(&self, device: cl_device_id) -> Result<Vec<size_t>> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
        )?
        .to_vec_size())
    }

    pub fn get_local_mem_size(&self, device: cl_device_id) -> Result<cl_ulong> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_LOCAL_MEM_SIZE,
        )?
        .to_ulong())
    }

    pub fn get_work_group_size_multiple(&self, device: cl_device_id) -> Result<size_t> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        )?
        .to_size())
    }

    pub fn get_private_mem_size(&self, device: cl_device_id) -> Result<cl_ulong> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_PRIVATE_MEM_SIZE,
        )?
        .to_ulong())
    }

    #[cfg(feature = "cl_khr_subgroups")]
    pub fn get_kernel_sub_group_info_khr(
        &self,
        device: cl_device_id,
        device: cl_device_id,
        param_name: ext::KernelSubGroupInfoKhr,
        input_values: &[size_t],
    ) -> Result<size_t> {
        Ok(get_kernel_sub_group_info_khr(
            self.kernel,
            device,
            param_name,
            input_values.len(),
            input_values.as_ptr(),
        )?)
    }

    #[cfg(feature = "cl_khr_suggested_local_work_size")]
    pub fn get_kernel_suggested_local_work_size_khr(
        &self,
        command_queue: cl_command_queue,
        work_dim: cl_uint,
        global_work_offset: *const size_t,
        global_work_size: *const size_t,
    ) -> Result<size_t> {
        Ok(get_kernel_suggested_local_work_size_khr(
            command_queue,
            self.kernel,
            work_dim,
            global_work_offset,
            global_work_size,
        )?)
    }
}

/// Create OpenCL Kernel objects for all the kernel functions in a program.
///
/// * `program` - a valid OpenCL program.
///
/// returns a Result containing the new Kernels in a Vec
/// or the error code from the OpenCL C API function.
pub fn create_program_kernels(program: &Program) -> Result<Vec<Kernel>> {
    let kernels = create_kernels_in_program(program.get())?;
    Ok(kernels
        .iter()
        .map(|kernel| Kernel::new(*kernel))
        .collect::<Vec<Kernel>>())
}

/// A struct that implements the [builder pattern](https://doc.rust-lang.org/1.0.0/style/ownership/builders.html)
/// to simplify setting up [Kernel] arguments and the [NDRange](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_mapping_work_items_onto_an_ndrange)
/// when enqueueing a [Kernel] on a [CommandQueue].
#[derive(Debug)]
pub struct ExecuteKernel<'a> {
    pub kernel: &'a Kernel,
    pub num_args: cl_uint,
    pub global_work_offsets: Vec<size_t>,
    pub global_work_sizes: Vec<size_t>,
    pub local_work_sizes: Vec<size_t>,
    pub event_wait_list: Vec<cl_event>,

    arg_index: cl_uint,
}

unsafe impl Send for ExecuteKernel<'_> {}

impl<'a> ExecuteKernel<'a> {
    pub fn new(kernel: &'a Kernel) -> ExecuteKernel {
        ExecuteKernel {
            kernel,
            num_args: kernel
                .num_args()
                .expect("ExecuteKernel: error reading kernel.num_args"),
            global_work_offsets: Vec::new(),
            global_work_sizes: Vec::new(),
            local_work_sizes: Vec::new(),
            event_wait_list: Vec::new(),

            arg_index: 0,
        }
    }

    /// Set the next argument of the kernel.  
    /// Calls `self.kernel.set_arg` to set the next unset kernel argument.
    ///
    /// # Panics
    ///
    /// Panics if too many arguments have been set.
    ///
    /// * `arg` - a reference to the data for the kernel argument.
    ///
    /// returns a reference to self.
    pub fn set_arg<'b, T>(&'b mut self, arg: &T) -> &'b mut Self {
        assert!(
            self.arg_index < self.num_args,
            "ExecuteKernel::set_arg too many args"
        );

        self.kernel.set_arg(self.arg_index, arg).unwrap();
        self.arg_index += 1;
        self
    }

    /// Set the next argument of the kernel as a local buffer
    /// Calls `self.kernel.set_arg_local_buffer` to set the next unset kernel argument.
    ///
    /// # Panics
    ///
    /// Panics if too many arguments have been set.
    ///
    /// * `size` - the size of the local memory buffer in bytes.
    ///
    /// returns a reference to self.
    pub fn set_arg_local_buffer<'b, T>(&'b mut self, size: size_t) -> &'b mut Self {
        assert!(
            self.arg_index < self.num_args,
            "ExecuteKernel::set_arg_local_buffer too many args"
        );

        self.kernel
            .set_arg_local_buffer(self.arg_index, size)
            .unwrap();
        self.arg_index += 1;
        self
    }

    /// Set the next argument of the kernel.  
    /// Calls `self.kernel.set_arg` to set the next unset kernel argument.
    ///
    /// # Panics
    ///
    /// Panics if too many arguments have been set.
    ///
    /// * `arg` - a reference to the data for the kernel argument.
    ///
    /// returns a reference to self.
    pub fn set_arg_svm<'b, T>(&'b mut self, arg_ptr: *const T) -> &'b mut Self {
        assert!(
            self.arg_index < self.num_args,
            "ExecuteKernel::set_arg_svm too many args"
        );

        self.kernel
            .set_arg_svm_pointer(self.arg_index, arg_ptr as *const c_void)
            .unwrap();
        self.arg_index += 1;
        self
    }

    /// Pass additional information other than argument values to a kernel.  
    ///
    /// * `param_name` - the information to be passed to kernel, see:
    /// [Kernel Execution Properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#kernel-exec-info-table).
    /// * `param_ptr` - pointer to the data for the param_name.
    ///
    /// returns a reference to self.
    pub fn set_exec_info<'b, T>(
        &'b mut self,
        param_name: cl_kernel_exec_info,
        param_ptr: *const T,
    ) -> &'b mut Self {
        self.kernel.set_exec_info(param_name, param_ptr).unwrap();
        self
    }

    /// Set a global work offset for a call to clEnqueueNDRangeKernel.  
    ///
    /// * `size` - the size of the global work offset.
    ///
    /// returns a reference to self.
    pub fn set_global_work_offset<'b>(&'b mut self, size: size_t) -> &'b mut Self {
        self.global_work_offsets.push(size);
        self
    }

    /// Set the global work offsets for a call to clEnqueueNDRangeKernel.  
    ///
    /// # Panics
    ///
    /// Panics if global_work_offsets is already set.
    ///
    /// * `sizes` - the sizes of the global work offset.
    ///
    /// returns a reference to self.
    pub fn set_global_work_offsets<'b>(&'b mut self, sizes: &[size_t]) -> &'b mut Self {
        assert!(
            self.global_work_offsets.is_empty(),
            "ExecuteKernel::set_global_work_offsets already set"
        );
        self.global_work_offsets.resize(sizes.len(), 0);
        self.global_work_offsets.copy_from_slice(sizes);
        self
    }

    /// Set a global work size for a call to clEnqueueNDRangeKernel.  
    ///
    /// * `size` - the size of the global work size.
    ///
    /// returns a reference to self.
    pub fn set_global_work_size<'b>(&'b mut self, size: size_t) -> &'b mut Self {
        self.global_work_sizes.push(size);
        self
    }

    /// Set the global work sizes for a call to clEnqueueNDRangeKernel.  
    ///
    /// # Panics
    ///
    /// Panics if global_work_sizes is already set.
    ///
    /// * `sizes` - the sizes of the global work sizes.
    ///
    /// returns a reference to self.
    pub fn set_global_work_sizes<'b>(&'b mut self, sizes: &[size_t]) -> &'b mut Self {
        assert!(
            self.global_work_sizes.is_empty(),
            "ExecuteKernel::global_work_sizes already set"
        );
        self.global_work_sizes.resize(sizes.len(), 0);
        self.global_work_sizes.copy_from_slice(sizes);
        self
    }

    /// Set a local work size for a call to clEnqueueNDRangeKernel.  
    ///
    /// * `size` - the size of the local work size.
    ///
    /// returns a reference to self.
    pub fn set_local_work_size<'b>(&'b mut self, size: size_t) -> &'b mut Self {
        self.local_work_sizes.push(size);
        self
    }

    /// Set the local work sizes for a call to clEnqueueNDRangeKernel.  
    ///
    /// # Panics
    ///
    /// Panics if local_work_sizes is already set.
    ///
    /// * `sizes` - the sizes of the local work sizes.
    ///
    /// returns a reference to self.
    pub fn set_local_work_sizes<'b>(&'b mut self, sizes: &[size_t]) -> &'b mut Self {
        assert!(
            self.local_work_sizes.is_empty(),
            "ExecuteKernel::local_work_sizes already set"
        );
        self.local_work_sizes.resize(sizes.len(), 0);
        self.local_work_sizes.copy_from_slice(sizes);
        self
    }

    /// Set an event for the event_wait_list in a call to clEnqueueNDRangeKernel.  
    ///
    /// * `event` - the Event to add to the event_wait_list.
    ///
    /// returns a reference to self.
    pub fn set_wait_event<'b>(&'b mut self, event: &Event) -> &'b mut Self {
        self.event_wait_list.push(event.get());
        self
    }

    /// Set the event_wait_list in a call to clEnqueueNDRangeKernel.  
    ///
    /// # Panics
    ///
    /// Panics if event_wait_list is already set.
    ///
    /// * `events` - the cl_events in the call to clEnqueueNDRangeKernel.
    ///
    /// returns a reference to self.
    pub fn set_event_wait_list<'b>(&'b mut self, events: &[cl_event]) -> &'b mut Self {
        assert!(
            self.event_wait_list.is_empty(),
            "ExecuteKernel::event_wait_list already set"
        );
        self.event_wait_list.resize(events.len(), ptr::null_mut());
        self.event_wait_list.copy_from_slice(events);
        self
    }

    fn validate(&self, max_work_item_dimensions: usize) {
        assert!(
            self.num_args == self.arg_index,
            "ExecuteKernel too few args"
        );

        let work_dim = self.global_work_sizes.len();
        assert!(0 < work_dim, "ExecuteKernel not enough global_work_sizes");

        assert!(
            work_dim <= max_work_item_dimensions,
            "ExecuteKernel too many global_work_sizes"
        );

        let offsets_dim = self.global_work_offsets.len();
        assert!(
            (0 == offsets_dim) || (offsets_dim == work_dim),
            "ExecuteKernel global_work_offsets dimensions != global_work_sizes"
        );

        let locals_dim = self.local_work_sizes.len();
        assert!(
            (0 == locals_dim) || (locals_dim == work_dim),
            "ExecuteKernel local_work_sizes dimensions != global_work_sizes"
        );
    }

    fn clear(&mut self) {
        self.global_work_offsets.clear();
        self.global_work_sizes.clear();
        self.local_work_sizes.clear();
        self.event_wait_list.clear();

        self.arg_index = 0;
    }

    /// Calls clEnqueueNDRangeKernel on the given with [CommandQueue] with the
    /// global and local work sizes and the global work offsets together with
    /// an events wait list.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * too few kernel arguments have been set
    /// * no global_work_sizes have been set
    /// * too many global_work_sizes have been set
    /// * global_work_offsets have been set and their dimensions do not match
    /// global_work_sizes
    /// * local_work_sizes have been set and their dimensions do not match
    /// global_work_sizes
    ///
    /// * `queue` - the [CommandQueue] to enqueue the [Kernel] on.
    ///
    /// return the [Event] for this command
    /// or the error code from the OpenCL C API function.
    pub fn enqueue_nd_range(&mut self, queue: &CommandQueue) -> Result<Event> {
        // Get max_work_item_dimensions for the device CommandQueue
        let max_work_item_dimensions = queue.max_work_item_dimensions() as usize;
        self.validate(max_work_item_dimensions);

        let event = queue.enqueue_nd_range_kernel(
            self.kernel.get(),
            self.global_work_sizes.len() as cl_uint,
            if self.global_work_offsets.is_empty() {
                ptr::null()
            } else {
                self.global_work_offsets.as_ptr()
            },
            self.global_work_sizes.as_ptr(),
            if self.local_work_sizes.is_empty() {
                ptr::null()
            } else {
                self.local_work_sizes.as_ptr()
            },
            &self.event_wait_list,
        )?;

        self.clear();
        Ok(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::device::Device;
    use crate::platform::get_platforms;
    use crate::program::{Program, CL_KERNEL_ARG_INFO};
    use cl3::device::CL_DEVICE_TYPE_GPU;
    use std::collections::HashSet;

    const PROGRAM_SOURCE: &str = r#"
        kernel void add(global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }

        kernel void subtract(global float* buffer, float scalar) {
            buffer[get_global_id(0)] -= scalar;
        }
    "#;

    #[test]
    fn test_create_program_kernels() {
        let platforms = get_platforms().unwrap();
        assert!(0 < platforms.len());

        // Get the first platform
        let platform = &platforms[0];

        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
        assert!(0 < devices.len());

        // Get the first device
        let device = Device::new(devices[0]);
        let context = Context::from_device(&device).unwrap();

        let program =
            Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_KERNEL_ARG_INFO)
                .expect("Program::create_and_build_from_source failed");

        // Create the kernels from the OpenCL program source.
        let kernels = create_program_kernels(&program).unwrap();
        assert!(2 == kernels.len());

        let kernel_0_name = kernels[0].function_name().unwrap();
        println!("OpenCL kernel_0_name: {}", kernel_0_name);

        let kernel_1_name = kernels[1].function_name().unwrap();
        println!("OpenCL kernel_1_name: {}", kernel_1_name);

        let kernel_names: HashSet<&str> = program.kernel_names().split(';').collect();

        assert!(kernel_names.contains(&kernel_0_name as &str));
        assert!(kernel_names.contains(&kernel_1_name as &str));

        let num_args_0 = kernels[0].num_args().expect("OpenCL kernel_0.num_args");
        println!("OpenCL kernel_0 num args: {}", num_args_0);

        let value = kernels[0].num_args().unwrap();
        println!("kernel.num_args(): {}", value);
        assert_eq!(2, value);

        let value = kernels[0].reference_count().unwrap();
        println!("kernel.reference_count(): {}", value);
        assert_eq!(1, value);

        let value = kernels[0].context().unwrap();
        assert!(context.get() == value);

        let value = kernels[0].program().unwrap();
        assert!(program.get() == value);

        let value = kernels[0].attributes().unwrap();
        println!("kernel.attributes(): {}", value);
        // assert!(value.is_empty());

        let arg0_address = kernels[0]
            .get_arg_address_qualifier(0)
            .expect("OpenCL kernel_0.get_arg_address_qualifier");
        println!(
            "OpenCL kernel_0.get_arg_address_qualifier: {:X}",
            arg0_address
        );

        let arg0_access = kernels[0]
            .get_arg_access_qualifier(0)
            .expect("OpenCL kernel_0.get_arg_access_qualifier");
        println!(
            "OpenCL kernel_0.get_arg_access_qualifier: {:X}",
            arg0_access
        );

        let arg0_type_name = kernels[0]
            .get_arg_type_name(0)
            .expect("OpenCL kernel_0.get_arg_type_name");
        println!("OpenCL kernel_0.get_arg_type_name: {}", arg0_type_name);

        let arg0_type = kernels[0]
            .get_arg_type_qualifier(0)
            .expect("OpenCL kernel_0.get_arg_type_qualifier");
        println!("OpenCL kernel_0.get_arg_type_qualifier: {}", arg0_type);

        let arg0_name = kernels[0]
            .get_arg_name(0)
            .expect("OpenCL kernel_0.get_arg_name");
        println!("OpenCL kernel_0.get_arg_name: {}", arg0_name);

        let value = kernels[0].get_work_group_size(device.id()).unwrap();
        println!("kernel.get_work_group_size(): {}", value);
        // assert_eq!(256, value);

        let value = kernels[0].get_compile_work_group_size(device.id()).unwrap();
        println!("kernel.get_work_group_size(): {:?}", value);
        assert_eq!(3, value.len());

        let value = kernels[0].get_local_mem_size(device.id()).unwrap();
        println!("kernel.get_local_mem_size(): {}", value);
        // assert_eq!(1, value);

        let value = kernels[0]
            .get_work_group_size_multiple(device.id())
            .unwrap();
        println!("kernel.get_work_group_size_multiple(): {}", value);
        // assert_eq!(32, value);

        let value = kernels[0].get_private_mem_size(device.id()).unwrap();
        println!("kernel.get_private_mem_size(): {}", value);
        // assert_eq!(0, value);
    }
}
