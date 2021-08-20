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

use super::context::Context;

use super::Result;
#[allow(unused_imports)]
use cl3::error_codes::CL_BUILD_PROGRAM_FAILURE;
#[allow(unused_imports)]
use cl3::ext;
#[allow(unused_imports)]
use cl3::types::{cl_context, cl_device_id, cl_int, cl_program, cl_uchar, cl_uint, CL_FALSE};
#[allow(unused_imports)]
use libc::{c_void, intptr_t, size_t};
#[allow(unused_imports)]
use std::ffi::{CStr, CString};
use std::ptr;
use std::result;

// Compile, link and build options.
// These options can be passed to Program::compile, Program::link or Program::build, see:
// [Compiler Options](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#compiler-options)
// [Linker Options](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#linker-options)
// [Build Options](https://man.opencl.org/clBuildProgram.html)

// Note: the options have a trailing space so that they can be concatenated.

// Math Intrinsics Options
pub const CL_SINGLE_RECISION_CONSTANT: &str = "-cl-single-precision-constant ";
pub const CL_DENORMS_ARE_ZERO: &str = "-cl-denorms-are-zero ";
pub const CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: &str = "-cl-fp32-correctly-rounded-divide-sqrt ";

// Optimization Options
pub const CL_OPT_DISABLE: &str = "-cl-opt-disable ";
pub const CL_STRICT_ALIASING: &str = "-cl-strict-aliasing ";
pub const CL_UNIFORM_WORK_GROUP_SIZE: &str = "-cl-uniform-work-group-size ";
pub const CL_NO_SUBGROUP_INFO: &str = "-cl-no-subgroup-ifp ";
pub const CL_MAD_ENABLE: &str = "-cl-mad-enable ";
pub const CL_NO_SIGNED_ZEROS: &str = "-cl-no-signed-zeros ";
pub const CL_UNSAFE_MATH_OPTIMIZATIONS: &str = "-cl-unsafe-math-optimizations ";
pub const CL_FINITE_MATH_ONLY: &str = "-cl-finite-math-only ";
pub const CL_FAST_RELAXED_MATH: &str = "-cl-fast-relaxed-math ";

// OpenCL C version Options

/// Applications are required to specify the -cl-std=CL2.0 build option to
/// compile or build programs with OpenCL C 2.0.
pub const CL_STD_2_0: &str = "-cl-std=CL2.0 ";

/// Applications are required to specify the -cl-std=CL3.0 build option to
/// compile or build programs with OpenCL C 3.0.
pub const CL_STD_3_0: &str = "-cl-std=CL3.0 ";

/// This option allows the compiler to store information about the
/// arguments of kernels in the program executable.
pub const CL_KERNEL_ARG_INFO: &str = "-cl-kernel-arg-info ";

pub const DEBUG_OPTION: &str = "-g ";

// Options enabled by the cl_khr_spir extension
pub const BUILD_OPTION_X_SPIR: &str = "-x spir ";
pub const BUILD_OPTION_SPIR_STD_1_2: &str = "-spir-std=1.2 ";

// Link and build options.
pub const CREATE_LIBRARY: &str = "-create-library ";
pub const ENABLE_LINK_OPTIONS: &str = "-enable-link-options ";

/// An OpenCL program object.  
/// Stores the names of the OpenCL kernels in the program.
/// Implements the Drop trait to call release_program when the object is dropped.
#[derive(Debug)]
pub struct Program {
    program: cl_program,
    kernel_names: String,
}

impl Drop for Program {
    fn drop(&mut self) {
        release_program(self.program).expect("Error: clReleaseProgram");
    }
}

impl Program {
    fn new(program: cl_program, kernel_names: &str) -> Program {
        Program {
            program,
            kernel_names: kernel_names.to_owned(),
        }
    }

    /// Get the underlying OpenCL cl_program.
    pub fn get(&self) -> cl_program {
        self.program
    }

    /// Get the names of the OpenCL kernels in the Program, in a string
    /// separated by semicolons.
    pub fn kernel_names(&self) -> &str {
        &self.kernel_names
    }

    /// Create a Program for a context and load source code into that object.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `sources` - an array of strs containing the source code strings.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    pub fn create_from_sources(context: &Context, sources: &[&str]) -> Result<Program> {
        Ok(Program::new(
            create_program_with_source(context.get(), sources)?,
            "",
        ))
    }

    /// Create a Program for a context and load a source code string into that object.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `src` - a str containing a source code string.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    pub fn create_from_source(context: &Context, src: &str) -> Result<Program> {
        let sources = [src];
        Ok(Program::new(
            create_program_with_source(context.get(), &sources)?,
            "",
        ))
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
        context: &Context,
        devices: &[cl_device_id],
        binaries: &[&[u8]],
    ) -> Result<Program> {
        Ok(Program::new(
            create_program_with_binary(context.get(), devices, binaries)?,
            "",
        ))
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
    #[cfg(feature = "CL_VERSION_1_2")]
    pub fn create_from_builtin_kernels(
        context: &Context,
        devices: &[cl_device_id],
        kernel_names: &str,
    ) -> Result<Program> {
        // Ensure options string is null terminated
        let c_names = CString::new(kernel_names)
            .expect("Program::create_from_builtin_kernels, invalid kernel_names");
        Ok(Program::new(
            create_program_with_builtin_kernels(context.get(), devices, &c_names)?,
            kernel_names,
        ))
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
    pub fn create_from_il(context: &Context, il: &[u8]) -> Result<Program> {
        Ok(Program::new(
            create_program_with_il(context.get(), &il)?,
            "",
        ))
    }

    #[cfg(feature = "cl_khr_il_program")]
    pub fn create_from_il_khr(context: &Context, il: &[u8]) -> Result<Program> {
        Ok(Program::new(
            ext::create_program_with_il_khr(context.get(), &il)?,
            "",
        ))
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
    pub fn build(&mut self, devices: &[cl_device_id], options: &str) -> Result<()> {
        // Ensure options string is null terminated
        let c_options = CString::new(options).expect("Program::build, invalid options");
        build_program(self.program, devices, &c_options, None, ptr::null_mut())?;
        self.kernel_names = self.get_kernel_names()?;
        Ok(())
    }

    /// Create and build an OpenCL Program from an array of source code strings
    /// with the given options.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `sources` - an array of strs containing the source code strings.
    /// * `options` - the build options in a null-terminated string.
    ///
    /// returns a Result containing the new Program, the name of the error code
    /// from the OpenCL C API function or the build log, if the build failed.
    pub fn create_and_build_from_sources(
        context: &Context,
        sources: &[&str],
        options: &str,
    ) -> result::Result<Program, String> {
        let mut program =
            Program::create_from_sources(context, sources).map_err(|e| e.to_string())?;
        match program.build(context.devices(), options) {
            Ok(_) => Ok(program),
            Err(e) => {
                if CL_BUILD_PROGRAM_FAILURE == e.0 {
                    let log = program
                        .get_build_log(context.devices()[0])
                        .map_err(|e| e.to_string())?;
                    Err(e.to_string() + ", build log: " + &log)
                } else {
                    Err(e.to_string())
                }
            }
        }
    }

    /// Create and build an OpenCL Program from source code with the given options.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `src` - a str containing a source code string.
    /// * `options` - the build options in a null-terminated string.
    ///
    /// returns a Result containing the new Program, the name of the error code
    /// from the OpenCL C API function or the build log, if the build failed.
    pub fn create_and_build_from_source(
        context: &Context,
        src: &str,
        options: &str,
    ) -> result::Result<Program, String> {
        let sources = [src];
        Program::create_and_build_from_sources(
            context, &sources, options,
        )
    }

    /// Create and build an OpenCL Program from binaries with the given options.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `binaries` - a slice of program binaries slices.
    /// * `options` - the build options in a null-terminated string.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    pub fn create_and_build_from_binary(
        context: &Context,
        binaries: &[&[u8]],
        options: &str,
    ) -> Result<Program> {
        let mut program = Program::create_from_binary(context, context.devices(), binaries)?;
        program.build(context.devices(), options)?;
        Ok(program)
    }

    /// Create and build an OpenCL Program from intermediate language with the
    /// given options.  
    /// CL_VERSION_2_1
    ///
    /// * `context` - a valid OpenCL context.
    /// * `il` - a slice of program intermediate language code.
    /// * `options` - the build options in a null-terminated string.
    ///
    /// returns a Result containing the new Program
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_2_1")]
    pub fn create_and_build_from_il(
        context: &Context,
        il: &[u8],
        options: &str,
    ) -> Result<Program> {
        let mut program = Program::create_from_il(&context, il)?;
        program.build(context.devices(), options)?;
        Ok(program)
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
    #[cfg(feature = "CL_VERSION_1_2")]
    pub fn compile(
        &mut self,
        devices: &[cl_device_id],
        options: &str,
        input_headers: &[cl_program],
        header_include_names: &[&CStr],
    ) -> Result<()> {
        // Ensure options string is null terminated
        let c_options = CString::new(options).expect("Program::compile, invalid options");
        Ok(compile_program(
            self.program,
            devices,
            &c_options,
            input_headers,
            header_include_names,
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
    #[cfg(feature = "CL_VERSION_1_2")]
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
            devices,
            &c_options,
            input_programs,
            None,
            ptr::null_mut(),
        )?;
        self.kernel_names = self.get_kernel_names()?;
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

    pub fn get_num_kernels(&self) -> Result<size_t> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_NUM_KERNELS)?.to_size())
    }

    pub fn get_kernel_names(&self) -> Result<String> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_KERNEL_NAMES)?.to_string())
    }

    /// CL_VERSION_2_1
    pub fn get_program_il(&self) -> Result<String> {
        Ok(get_program_info(self.program, ProgramInfo::CL_PROGRAM_IL)?.to_string())
    }

    /// CL_VERSION_2_2
    pub fn get_program_scope_global_ctors_present(&self) -> Result<bool> {
        Ok(get_program_info(
            self.program,
            ProgramInfo::CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT,
        )?
        .to_uint()
            != CL_FALSE)
    }

    /// CL_VERSION_2_2
    pub fn get_program_scope_global_dtors_present(&self) -> Result<bool> {
        Ok(get_program_info(
            self.program,
            ProgramInfo::CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT,
        )?
        .to_uint()
            != CL_FALSE)
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

    /// CL_VERSION_2_0
    pub fn get_build_global_variable_total_size(&self, device: cl_device_id) -> Result<size_t> {
        Ok(get_program_build_info(
            self.program,
            device,
            ProgramBuildInfo::CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE,
        )?
        .to_size())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::device::Device;
    use crate::platform::get_platforms;
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
    fn test_create_and_build_from_source() {
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
            Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_DENORMS_ARE_ZERO)
                .expect("Program::create_and_build_from_source failed");

        let names: HashSet<&str> = program.kernel_names().split(';').collect();
        println!("OpenCL Program kernel_names len: {}", names.len());
        println!("OpenCL Program kernel_names: {:?}", names);

        let value = program.get_reference_count().unwrap();
        println!("program.get_reference_count(): {}", value);
        assert_eq!(1, value);

        let value = program.get_context().unwrap();
        assert!(context.get() == value);

        let value = program.get_num_devices().unwrap();
        println!("program.get_num_devices(): {}", value);
        assert_eq!(1, value);

        let value = program.get_devices().unwrap();
        assert!(device.id() == value[0] as cl_device_id);

        let value = program.get_source().unwrap();
        println!("program.get_source(): {}", value);
        assert!(!value.is_empty());

        let value = program.get_binary_sizes().unwrap();
        println!("program.get_binary_sizes(): {:?}", value);
        assert!(0 < value[0]);

        let value = program.get_binaries().unwrap();
        // println!("program.get_binaries(): {:?}", value);
        assert!(!value[0].is_empty());

        let value = program.get_num_kernels().unwrap();
        println!("program.get_num_kernels(): {}", value);
        assert_eq!(2, value);

        // let value = program.get_program_il().unwrap();
        // println!("program.get_program_il(): {:?}", value);
        // assert!(!value.is_empty());

        let value = program.get_build_status(device.id()).unwrap();
        println!("program.get_build_status(): {}", value);
        assert!(CL_BUILD_SUCCESS == value);

        let value = program.get_build_options(device.id()).unwrap();
        println!("program.get_build_options(): {}", value);
        assert!(!value.is_empty());

        let value = program.get_build_log(device.id()).unwrap();
        println!("program.get_build_log(): {}", value);
        // assert!(!value.is_empty());

        let value = program.get_build_binary_type(device.id()).unwrap();
        println!("program.get_build_binary_type(): {}", value);
        assert_eq!(CL_PROGRAM_BINARY_TYPE_EXECUTABLE as u32, value);

        // CL_VERSION_2_0 value
        match program.get_build_global_variable_total_size(device.id()) {
            Ok(value) => println!("program.get_build_global_variable_total_size(): {}", value),
            Err(e) => println!(
                "OpenCL error, program.get_build_global_variable_total_size(): {}",
                e
            ),
        };
    }
}
