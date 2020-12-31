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

pub use cl3::kernel::*;

use super::command_queue::CommandQueue;
use super::event::Event;

use cl3::types::{
    cl_device_id, cl_event, cl_int, cl_kernel, cl_kernel_exec_info, cl_uint, cl_ulong,
};

use libc::{c_void, intptr_t, size_t};
use std::ffi::CString;
use std::mem;
use std::ptr;

pub struct Kernel {
    kernel: cl_kernel,
    num_args: cl_uint,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        release_kernel(self.kernel).unwrap();
        // println!("Kernel::drop");
    }
}

impl Kernel {
    pub fn new(kernel: cl_kernel) -> Result<Kernel, cl_int> {
        let num_args = get_kernel_info(kernel, KernelInfo::CL_KERNEL_NUM_ARGS)?.to_uint();
        Ok(Kernel { kernel, num_args })
    }

    pub fn get(&self) -> cl_kernel {
        self.kernel
    }

    #[cfg(feature = "CL_VERSION_2_1")]
    pub fn clone(&self) -> Result<Kernel, cl_int> {
        let kernel = clone_kernel(self.kernel)?;
        Ok(Kernel {
            kernel,
            num_args: self.num_args,
        })
    }

    pub fn set_arg<T>(&self, arg_index: cl_uint, arg: &T) -> Result<(), cl_int> {
        set_kernel_arg(
            self.kernel,
            arg_index,
            mem::size_of::<T>(),
            arg as *const _ as *const c_void,
        )
    }

    pub fn set_arg_local_buffer(&self, arg_index: cl_uint, size: size_t) -> Result<(), cl_int> {
        set_kernel_arg(self.kernel, arg_index, size, ptr::null())
    }

    pub fn set_arg_svm_pointer(
        &self,
        arg_index: cl_uint,
        arg_ptr: *const c_void,
    ) -> Result<(), cl_int> {
        set_kernel_arg_svm_pointer(self.kernel, arg_index, arg_ptr)
    }

    pub fn set_exec_info<T>(
        &self,
        param_name: cl_kernel_exec_info,
        param_ptr: *const T,
    ) -> Result<(), cl_int> {
        set_kernel_exec_info(
            self.kernel,
            param_name,
            mem::size_of::<T>(),
            param_ptr as *const c_void,
        )
    }

    pub fn function_name(&self) -> Result<CString, cl_int> {
        Ok(
            get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_FUNCTION_NAME)?
                .to_str()
                .unwrap(),
        )
    }

    pub fn attributes(&self) -> Result<CString, cl_int> {
        Ok(
            get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_ATTRIBUTES)?
                .to_str()
                .unwrap(),
        )
    }

    pub fn num_args(&self) -> cl_uint {
        self.num_args
    }

    pub fn reference_count(&self) -> Result<cl_uint, cl_int> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_REFERENCE_COUNT)?.to_uint())
    }

    pub fn context(&self) -> Result<intptr_t, cl_int> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_CONTEXT)?.to_ptr())
    }

    pub fn program(&self) -> Result<intptr_t, cl_int> {
        Ok(get_kernel_info(self.kernel, KernelInfo::CL_KERNEL_PROGRAM)?.to_ptr())
    }

    pub fn get_arg_address_qualifier(&self, arg_indx: cl_uint) -> Result<cl_uint, cl_int> {
        Ok(get_kernel_arg_info(
            self.kernel,
            arg_indx,
            KernelArgInfo::CL_KERNEL_ARG_ADDRESS_QUALIFIER,
        )?
        .to_uint())
    }

    pub fn get_arg_access_qualifier(&self, arg_indx: cl_uint) -> Result<cl_uint, cl_int> {
        Ok(get_kernel_arg_info(
            self.kernel,
            arg_indx,
            KernelArgInfo::CL_KERNEL_ARG_ACCESS_QUALIFIER,
        )?
        .to_uint())
    }

    pub fn get_arg_type_qualifier(&self, arg_indx: cl_uint) -> Result<cl_uint, cl_int> {
        Ok(get_kernel_arg_info(
            self.kernel,
            arg_indx,
            KernelArgInfo::CL_KERNEL_ARG_TYPE_QUALIFIER,
        )?
        .to_uint())
    }

    pub fn get_arg_type_name(&self, arg_indx: cl_uint) -> Result<CString, cl_int> {
        Ok(get_kernel_arg_info(
            self.kernel,
            arg_indx,
            KernelArgInfo::CL_KERNEL_ARG_TYPE_NAME,
        )?
        .to_str()
        .unwrap())
    }

    pub fn get_arg_name(&self, arg_indx: cl_uint) -> Result<CString, cl_int> {
        Ok(
            get_kernel_arg_info(self.kernel, arg_indx, KernelArgInfo::CL_KERNEL_ARG_NAME)?
                .to_str()
                .unwrap(),
        )
    }

    pub fn get_work_group_size(&self, device: cl_device_id) -> Result<size_t, cl_int> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_WORK_GROUP_SIZE,
        )?
        .to_size())
    }

    pub fn get_work_group_size_multiple(&self, device: cl_device_id) -> Result<size_t, cl_int> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        )?
        .to_size())
    }

    pub fn get_compile_work_group_size(&self, device: cl_device_id) -> Result<Vec<size_t>, cl_int> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
        )?
        .to_vec_size())
    }

    pub fn get_local_mem_size(&self, device: cl_device_id) -> Result<cl_ulong, cl_int> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_LOCAL_MEM_SIZE,
        )?
        .to_ulong())
    }

    pub fn get_private_mem_size(&self, device: cl_device_id) -> Result<cl_ulong, cl_int> {
        Ok(get_kernel_work_group_info(
            self.kernel,
            device,
            KernelWorkGroupInfo::CL_KERNEL_PRIVATE_MEM_SIZE,
        )?
        .to_ulong())
    }
}

pub struct ExecuteKernel<'a> {
    pub kernel: &'a Kernel,
    pub global_work_offsets: Vec<size_t>,
    pub global_work_sizes: Vec<size_t>,
    pub local_work_sizes: Vec<size_t>,

    arg_index: cl_uint,
}

impl<'a> ExecuteKernel<'a> {
    pub fn new(kernel: &'a Kernel) -> ExecuteKernel {
        ExecuteKernel {
            kernel,
            global_work_offsets: Vec::new(),
            global_work_sizes: Vec::new(),
            local_work_sizes: Vec::new(),

            arg_index: 0,
        }
    }

    pub fn set_arg<'b, T>(&'b mut self, arg: &T) -> &'b mut Self {
        if self.kernel.num_args() <= self.arg_index {
            panic!("ExecuteKernel::set_arg too many args");
        }
        self.kernel.set_arg(self.arg_index, arg).unwrap();
        self.arg_index += 1;
        self
    }

    pub fn set_arg_local_buffer(&mut self, size: size_t) -> Result<(), cl_int> {
        if self.kernel.num_args() <= self.arg_index {
            panic!("ExecuteKernel::set_arg_local_buffer too many args");
        }

        self.kernel
            .set_arg_local_buffer(self.arg_index, size)
            .unwrap();
        self.arg_index += 1;
        Ok(())
    }

    pub fn set_arg_svm<'b, T>(&'b mut self, arg_ptr: *const T) -> &'b mut Self {
        if self.kernel.num_args() <= self.arg_index {
            panic!("ExecuteKernel::set_arg_local_buffer too many args");
        }

        self.kernel
            .set_arg_svm_pointer(self.arg_index, arg_ptr as *const c_void)
            .unwrap();
        self.arg_index += 1;
        self
    }

    pub fn set_exec_info<'b, T>(
        &'b mut self,
        param_name: cl_kernel_exec_info,
        param_ptr: *const T,
    ) -> &'b mut Self {
        self.kernel.set_exec_info(param_name, param_ptr).unwrap();
        self
    }

    pub fn set_global_work_offset<'b>(&'b mut self, size: size_t) -> &'b mut Self {
        self.global_work_offsets.push(size);
        self
    }

    pub fn set_global_work_offsets<'b>(&'b mut self, sizes: &[size_t]) -> &'b mut Self {
        if !self.global_work_offsets.is_empty() {
            panic!("ExecuteKernel::set_global_work_offsets already set");
        }
        self.global_work_offsets.resize(sizes.len(), 0);
        self.global_work_offsets.copy_from_slice(sizes);
        self
    }

    pub fn set_global_work_size<'b>(&'b mut self, size: size_t) -> &'b mut Self {
        self.global_work_sizes.push(size);
        self
    }
    pub fn set_global_work_sizes<'b>(&'b mut self, sizes: &[size_t]) -> &'b mut Self {
        if !self.global_work_sizes.is_empty() {
            panic!("ExecuteKernel::set_global_work_sizes already set");
        }
        self.global_work_sizes.resize(sizes.len(), 0);
        self.global_work_sizes.copy_from_slice(sizes);
        self
    }

    pub fn set_local_work_size<'b>(&'b mut self, size: size_t) -> &'b mut Self {
        self.local_work_sizes.push(size);
        self
    }

    pub fn set_local_work_sizes<'b>(&'b mut self, sizes: &[size_t]) -> &'b mut Self {
        if !self.local_work_sizes.is_empty() {
            panic!("ExecuteKernel::set_local_work_sizes already set");
        }
        self.local_work_sizes.resize(sizes.len(), 0);
        self.local_work_sizes.copy_from_slice(sizes);
        self
    }

    pub fn validate(&self) {
        if self.kernel.num_args() != self.arg_index {
            panic!("ExecuteKernel too few args");
        }

        let work_dim = self.global_work_sizes.len();
        if 0 == work_dim {
            panic!("ExecuteKernel not enough global_work_sizes");
        }

        if 3 < work_dim {
            panic!("ExecuteKernel too many global_work_sizes");
        }

        let offsets_dim = self.global_work_offsets.len();
        if (0 != offsets_dim) && (offsets_dim != work_dim) {
            panic!("ExecuteKernel global_work_offsets != global_work_sizes");
        }

        let locals_dim = self.local_work_sizes.len();
        if (0 != locals_dim) && (locals_dim != work_dim) {
            panic!("ExecuteKernel local_work_sizes != global_work_sizes");
        }
    }

    pub fn enqueue_nd_range(
        &self,
        queue: &CommandQueue,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        self.validate();
        queue.enqueue_nd_range_kernel(
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
            event_wait_list,
        )
    }
}
