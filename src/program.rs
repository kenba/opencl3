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

pub use cl3::program::*;

use super::Result;
use cl3::types::{cl_context, cl_device_id, cl_int, cl_program, cl_uchar, cl_uint};
#[allow(unused_imports)]
use libc::{c_char, c_void, intptr_t, size_t};
use std::ffi::{CStr, CString};
use std::ptr;

/// An OpenCL program object.  
/// Implements the Drop trait to call release_program when the object is dropped.
pub struct Program {
    program: cl_program,
}

impl Drop for Program {
    fn drop(&mut self) {
        release_program(self.program).expect("Error: clReleaseProgram");
    }
}

impl Program {
    fn new(program: cl_program) -> Program {
        Program { program }
    }

    /// Get the underlying OpenCL cl_program.
    pub fn get(&self) -> cl_program {
        self.program
    }

    /// Create a Program for a context and load source code into that object.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `sources` - an array of strs containing the source code strings.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    pub fn create_from_sources(context: cl_context, sources: &[&str]) -> Result<Program> {
        Ok(Program::new(create_program_with_source(context, sources)?))
    }

    /// Create a Program for a context and load a source code string into that object.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `source` - a str containing a source code string.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    pub fn create_from_source(context: cl_context, src: &str) -> Result<Program> {
        let sources = [src];
        Ok(Program::new(create_program_with_source(context, &sources)?))
    }

    /// Create a Program for a context and load binary bits into that object.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `devices` - a slice of devices that are in context.
    /// * `binaries` - a slice of program binaries slices.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    pub fn create_from_binary(
        context: cl_context,
        devices: &[cl_device_id],
        binaries: &[&[u8]],
    ) -> Result<Program> {
        Ok(Program::new(create_program_with_binary(
            context, devices, binaries,
        )?))
    }

    /// Create a Program for a context and  loads the information related to
    /// the built-in kernels into that object.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `devices` - a slice of devices that are in context.
    /// * `kernel_names` - a semi-colon separated list of built-in kernel names.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    pub fn create_from_builtin_kernels(
        context: cl_context,
        devices: &[cl_device_id],
        kernel_names: &str,
    ) -> Result<Program> {
        // Ensure options string is null terminated
        let c_names = CString::new(kernel_names)
            .expect("Program::create_from_builtin_kernels, invalid kernel_names");
        Ok(Program::new(create_program_with_builtin_kernels(
            context, devices, &c_names,
        )?))
    }

    /// Create a Program for a context and load code in an intermediate language
    /// into that object.  
    /// CL_VERSION_2_1
    ///
    /// * `context` - a valid OpenCL context.
    /// * `il` - a slice of program intermediate language code.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_2_1")]
    pub fn create_from_il(context: cl_context, il: &[u8]) -> Result<Program> {
        Ok(Program::new(create_program_with_il(context, &il)?))
    }

    /// Build (compile & link) a Program.
    ///
    /// * `devices` - a slice of devices that are in context.
    /// * `options` - the build options in a null-terminated string.
    /// * `pfn_notify` - an optional function pointer to a notification routine.
    /// * `user_data` - passed as an argument when pfn_notify is called, or ptr::null_mut().
    ///
    /// returns a null Result
    /// or the error code from the OpenCL C API function.
    pub fn build(&self, devices: &[cl_device_id], options: &str) -> Result<()> {
        // Ensure options string is null terminated
        let c_options = CString::new(options).expect("Program::build, invalid options");
        Ok(build_program(
            self.program,
            &devices,
            &c_options,
            None,
            ptr::null_mut(),
        )?)
    }

    /// Compile a program’s source for the devices the OpenCL context associated
    /// with the program.
    /// * `devices` - a slice of devices that are in context.
    /// * `options` - the compilation options in a null-terminated string.
    /// * `input_headers` - a slice of programs that describe headers in the input_headers.
    /// * `header_include_names` - an array that has a one to one correspondence with
    /// input_headers.
    ///
    /// returns a null Result
    /// or the error code from the OpenCL C API function.
    pub fn compile(
        &self,
        devices: &[cl_device_id],
        options: &str,
        input_headers: &[cl_program],
        header_include_names: &[&CStr],
    ) -> Result<()> {
        // Ensure options string is null terminated
        let c_options = CString::new(options).expect("Program::compile, invalid options");
        Ok(compile_program(
            self.program,
            &devices,
            &c_options,
            &input_headers,
            &header_include_names,
            None,
            ptr::null_mut(),
        )?)
    }

    /// Link a set of compiled program objects and libraries for the devices in the
    /// OpenCL context associated with the program.
    ///
    /// * `devices` - a slice of devices that are in context.
    /// * `options` - the link options in a null-terminated string.
    /// * `input_programs` - a slice of programs that describe headers in the input_headers.
    ///
    /// returns a null Result
    /// or the error code from the OpenCL C API function.
    pub fn link(
        &mut self,
        devices: &[cl_device_id],
        options: &str,
        input_programs: &[cl_program],
    ) -> Result<()> {
        // Ensure options string is null terminated
        let c_options = CString::new(options).expect("Program::link, invalid options");
        self.program = link_program(
            self.program,
            &devices,
            &c_options,
            &input_programs,
            None,
            ptr::null_mut(),
        )?;
        Ok(())
    }

    /// Register a callback function with a program object that is called when the
    /// program object is destroyed.
    /// CL_VERSION_2_2
    ///
    /// * `pfn_notify` - function pointer to the notification routine.
    /// * `user_data` - passed as an argument when pfn_notify is called, or ptr::null_mut().
    ///
    /// returns an empty Result or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_2_2")]
    pub fn set_release_callback(
        &self,
        pfn_notify: Option<extern "C" fn(program: cl_program, user_data: *mut c_void)>,
        user_data: *mut c_void,
    ) -> Result<()> {
        Ok(set_program_release_callback(
            self.program,
            pfn_notify,
            user_data,
        )?)
    }

    /// Set the value of a specialization constant.
    /// CL_VERSION_2_2
    ///
    /// * `spec_id` - the specialization constant whose value will be set.
    /// * `spec_size` - size in bytes of the data pointed to by spec_value.
    /// * `spec_value` - pointer to the memory location that contains the value
    /// of the specialization constant.
    ///
    /// returns an empty Result or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_2_2")]
    pub fn set_specialization_constant(
        &self,
        spec_id: cl_uint,
        spec_size: size_t,
        spec_value: *const c_void,
    ) -> Result<()> {
        Ok(set_program_specialization_constant(
            self.program,
            spec_id,
            spec_size,
            spec_value,
        )?)
    }

    pub fn get_reference_count(&self) -> Result<cl_uint> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_REFERENCE_COUNT)?.to_uint())
    }

    pub fn get_context(&self) -> Result<cl_context> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_CONTEXT)?.to_ptr() as cl_context)
    }

    pub fn get_num_devices(&self) -> Result<cl_uint> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_NUM_DEVICES)?.to_uint())
    }

    pub fn get_devices(&self) -> Result<Vec<intptr_t>> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_DEVICES)?.to_vec_intptr())
    }

    pub fn get_source(&self) -> Result<String> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_SOURCE)?.to_string())
    }

    pub fn get_binary_sizes(&self) -> Result<Vec<size_t>> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_BINARY_SIZES)?.to_vec_size())
    }

    pub fn get_binaries(&self) -> Result<Vec<Vec<cl_uchar>>> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_BINARIES)?.to_vec_vec_uchar())
    }

    pub fn get_num_kernels(&self) -> Result<cl_uint> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_NUM_KERNELS)?.to_uint())
    }

    pub fn get_kernel_names(&self) -> Result<String> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_KERNEL_NAMES)?.to_string())
    }

    pub fn get_program_il(&self) -> Result<String> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_IL)?.to_string())
    }

    pub fn get_build_status(&self, device: cl_device_id) -> Result<cl_int> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BUILD_STATUS,
        )?
        .to_int())
    }

    pub fn get_build_options(&self, device: cl_device_id) -> Result<String> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BUILD_OPTIONS,
        )?
        .to_string())
    }

    pub fn get_build_log(&self, device: cl_device_id) -> Result<String> {
        Ok(
            get_program_build_info(self.program, device, ProgramBuildInfo::CL_PROGRAM_BUILD_LOG)?
                .to_string(),
        )
    }

    pub fn get_build_binary_type(&self, device: cl_device_id) -> Result<cl_uint> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BINARY_TYPE,
        )?
        .to_uint())
    }

    pub fn get_build_global_variable_total_size(&self, device: cl_device_id) -> Result<size_t> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE,
        )?
        .to_size())
    }
}
