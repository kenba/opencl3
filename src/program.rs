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

pub use cl3::program::*;

use cl3::kernel;
use cl3::types::{cl_context, cl_device_id, cl_int, cl_kernel, cl_program, cl_uchar, cl_uint};
use libc::{intptr_t, size_t};
use std::ffi::{CStr, CString};
use std::ptr;
pub struct Program {
    program: cl_program,
}

impl Drop for Program {
    fn drop(&mut self) {
        release_program(self.program).unwrap();
        self.program = ptr::null_mut();
        // println!("Program::drop");
    }
}

impl Program {
    fn new(program: cl_program) -> Program {
        Program { program }
    }

    pub fn create_from_source(context: cl_context, src: &CStr) -> Result<Program, cl_int> {
        let char_ptrs: [*const _; 1] = [src.as_ptr()];
        let program = create_program_with_source(context, 1, char_ptrs.as_ptr(), ptr::null())?;
        Ok(Program::new(program))
    }

    pub fn create_from_binary(
        context: cl_context,
        devices: &[cl_device_id],
        binaries: &[&[u8]],
    ) -> Result<Program, cl_int> {
        let program = create_program_with_binary(context, devices, binaries)?;
        Ok(Program::new(program))
    }

    pub fn create_from_builtin_kernels(
        context: cl_context,
        devices: &[cl_device_id],
        kernel_names: &CStr,
    ) -> Result<Program, cl_int> {
        let program = create_program_with_builtin_kernels(context, devices, kernel_names)?;
        Ok(Program::new(program))
    }

    // #[cfg(feature = "CL_VERSION_2_1")]
    // pub fn create_from_il(context: cl_context, il: &[u8]) -> Result<Program, cl_int> {
    //     let program = create_program_with_il(context, &il)?;
    //     Ok(Program::new(program))
    // }

    pub fn build(&self, devices: &[cl_device_id], options: &CStr) -> Result<(), cl_int> {
        build_program(self.program, &devices, &options, None, ptr::null_mut())
    }

    pub fn create_kernel(&self, kernel_name: &CStr) -> Result<cl_kernel, cl_int> {
        kernel::create_kernel(self.program, kernel_name)
    }

    pub fn create_kernels_in_program(&self) -> Result<Vec<cl_kernel>, cl_int> {
        kernel::create_kernels_in_program(self.program)
    }

    pub fn get_reference_count(&self) -> Result<cl_uint, cl_int> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_REFERENCE_COUNT)?.to_uint())
    }

    pub fn get_context(&self) -> Result<intptr_t, cl_int> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_CONTEXT)?.to_ptr())
    }

    pub fn get_num_devices(&self) -> Result<cl_uint, cl_int> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_NUM_DEVICES)?.to_uint())
    }

    pub fn get_devices(&self) -> Result<Vec<intptr_t>, cl_int> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_DEVICES)?.to_vec_intptr())
    }

    pub fn get_source(&self) -> Result<CString, cl_int> {
        Ok(
            get_program_info(self.program, ProgramInfo::CL_PROGRAM_SOURCE)?
                .to_str()
                .unwrap(),
        )
    }

    pub fn get_binary_sizes(&self) -> Result<Vec<size_t>, cl_int> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_BINARY_SIZES)?.to_vec_size())
    }

    pub fn get_binaries(&self) -> Result<Vec<Vec<cl_uchar>>, cl_int> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_BINARIES)?.to_vec_vec_uchar())
    }

    pub fn get_num_kernels(&self) -> Result<cl_uint, cl_int> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_NUM_KERNELS)?.to_uint())
    }

    pub fn get_kernel_names(&self) -> Result<CString, cl_int> {
        Ok(
            get_program_info(self.program, ProgramInfo::CL_PROGRAM_KERNEL_NAMES)?
                .to_str()
                .unwrap(),
        )
    }

    pub fn get_program_il(&self) -> Result<CString, cl_int> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_IL)?
            .to_str()
            .unwrap())
    }

    pub fn get_build_status(&self, device: cl_device_id) -> Result<cl_int, cl_int> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BUILD_STATUS,
        )?
        .to_int())
    }

    pub fn get_build_options(&self, device: cl_device_id) -> Result<CString, cl_int> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BUILD_OPTIONS,
        )?
        .to_str()
        .unwrap())
    }

    pub fn get_build_log(&self, device: cl_device_id) -> Result<CString, cl_int> {
        Ok(
            get_program_build_info(self.program, device, ProgramBuildInfo::CL_PROGRAM_BUILD_LOG)?
                .to_str()
                .unwrap(),
        )
    }

    pub fn get_build_binary_type(&self, device: cl_device_id) -> Result<cl_uint, cl_int> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BINARY_TYPE,
        )?
        .to_uint())
    }

    pub fn get_build_global_variable_total_size(
        &self,
        device: cl_device_id,
    ) -> Result<size_t, cl_int> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE,
        )?
        .to_size())
    }
}
