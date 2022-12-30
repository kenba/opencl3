# Releases

## Version 0.9.2 (2022-12-30)

### Bug fixes

* PR [#58](https://github.com/kenba/opencl3/issues/58) Compile failure due to unmarked unsafe function call in program.rs.

## Version 0.9.1 (2022-09-20)

### Bug fixes

* PR [#54](https://github.com/kenba/opencl3/issues/54) Unshadow device re-exports.

### New Features

* Issue [#55](https://github.com/kenba/opencl3/issues/55) Add Khronos `cl_khr_command_buffer_mutable_dispatch` extension.

## Version 0.9.0 (2022-09-10)

### Breaking Changes

* Issue [#51](https://github.com/kenba/opencl3/issues/51) Undefined behaviour when using underlying OpenCL pointers without any unsafe.
* Issue [#52](https://github.com/kenba/opencl3/issues/52) `CL_MEM_USE_HOST_PTR` can result in undefined behaviour.

### New Features

* PR [#53](https://github.com/kenba/opencl3/issues/53) added `sync` for all threadsafe OpenCL objects.

## Version 0.8.1 (2022-07-23)

### Bug fixes

* Issue [#49](https://github.com/kenba/opencl3/issues/49) Better Error messages for
`ExecuteKernel::set_arg`.

### New Features

* Update for OpenCL extension `cl_ext_image_requirements_info`.

* PR [#50](https://github.com/kenba/opencl3/issues/50) Introduce `track_caller` for panicing methods of `ExecuteKernel`.

## Version 0.8.0 (2022-05-02)

### Breaking Changes

* Issue [#47](https://github.com/kenba/opencl3/issues/47) Base `opencl3` on [opencl-sys](https://crates.io/crates/opencl-sys) crate.
* Issue [#48](https://github.com/kenba/opencl3/issues/48) Put deprecation notices behind conditional attributes.

## Version 0.7.0 (2022-04-10)

### Breaking Changes

* Transition to Rust 2021 Edition.

## Version 0.6.3 (2021-12-30)

### New Features

* Issue [#44](https://github.com/kenba/opencl3/issues/44) Add provisional command-buffer extension. See: [cl_khr_command_buffer](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_command_buffer).

## Version 0.6.2 (2021-12-19)

### New Features

* Issue [#37](https://github.com/kenba/opencl3/issues/37) Implement DeserializeSeed for ExtendSvmVec to enable `serde` to deserialize directly into an `SvmVec`.

### Bug fixes

* Issue [#45](https://github.com/kenba/opencl3/issues/45) Examples and tests have `CL_MEM_READ_ONLY` and `CL_MEM_WRITE_ONLY` swapped over.

## Version 0.6.1 (2021-11-12)

### New Features

* Issue [#43](https://github.com/kenba/opencl3/issues/43) Update for new OpenCL extensions: `cl_khr_external_memory`, `cl_khr_external_semaphore` and `cl_khr_semaphore`.

## Version 0.6.0 (2021-10-16)

### Breaking Changes

* Issue [#41](https://github.com/kenba/opencl3/issues/41) Remove cl3 Info enums to support new OpenCL versions and extensions.
* Add UUID and LUID types. See cl3 Issue [#13](https://github.com/kenba/cl3/issues/13) Remove Info enums to support new OpenCL versions and extensions.
* Remove example from README.md.

## Version 0.5.3 (2021-10-10)

### New Features

* Issue [#38](https://github.com/kenba/opencl3/issues/38) Add SVM fine grain system support.
* Issue [#40](https://github.com/kenba/opencl3/issues/40) Replace all calls to `to_string` with `from` or `into`.
* Issue [#42](https://github.com/kenba/opencl3/issues/42) add `From` traits.
* Add `get_all_devices` function.

## Version 0.5.2 (2021-09-19)

```toml
[dependencies]
libc = "0.2"
cl3 = { version = "0.4", default-features = false }
serde = { version = "1.0", optional = true }
```

### New Features

* Issue [#39](https://github.com/kenba/opencl3/issues/39) Update for latest OpenCL-Headers.
* Add CONTRIBUTING and CODE_OF_CONDUCT documents.

## Version 0.5.1 (2021-09-17)

```toml
[dependencies]
libc = "0.2"
cl3 = { version = "0.4.2", default-features = false }
serde = { version = "1.0", optional = true }
```

### New Features

* Issue [#37](https://github.com/kenba/opencl3/issues/37) Implement Serde's Serialize, Deserialize for SvmVec.

### Bug fixes

* Issue [#32](https://github.com/kenba/opencl3/issues/32) Example from readme has zero output on GTX 1060 Max-Q.
* Issue [#35](https://github.com/kenba/opencl3/issues/35) Superfluous/Misleading generic parameter in `ExecuteKernel::set_arg_local_buffer`.

## Version 0.5.0 (2021-09-12)

### Breaking Changes

* Improve `SVM` interface and documentation.
* Remove svm_capabilities parameter from `SvmVec` methods.

### Bug fixes

* Issue [#33](https://github.com/kenba/opencl3/issues/33) Coarse-grained SVM has to be mapped before usage!

## Version 0.4.1 (2021-08-21)

Depends on:  
`cl3 = { version = "0.4.2", default-features = false }`

### New Features

* Issue [#30](https://github.com/kenba/opencl3/issues/30) opencl3 cannot be compiled with OpenCl 1.2 features only.

## Version 0.4.0 (2021-08-20)

Depends on `cl3` = "0.4.2".

### Breaking Changes

* Issue [#26](https://github.com/kenba/opencl3/issues/26) Should `CommandQueue.html::enqueue_write_buffer` take a mutable buffer reference.
* PR [#27](https://github.com/kenba/opencl3/pull/27) Make mutability explicit.

### New Features

* Issue [#25](https://github.com/kenba/opencl3/issues/25) Using `set_event_callback`.

## Version 0.3.1 (2021-08-06)

Depends on `cl3` = "0.4.1".

### New Features

* Add Device method for `cl_khr_integer_dot_product` extension.

## Version 0.3.0 (2021-07-10)

### Breaking Changes

* Issue [#21](https://github.com/kenba/opencl3/issues/21) `Device::available()` should return a boolean.
* PR [#22](https://github.com/kenba/opencl3/pull/22) Return booleans for device information where applicable.
* Issue [#24](https://github.com/kenba/opencl3/issues/24) Use `bool` instead of `cl_bool`.
* Use CL_BLOCKING and CL_NON_BLOCKING in enqueue calls.

## Version 0.2.4 (2021-07-03)

### New Features

* Issue [#18](https://github.com/kenba/opencl3/issues/18) Return UUID as array.
* PR [#19](https://github.com/kenba/opencl3/pull/19) Export sizes of UUID and LUID.

### Bug fixes

* Issue [#20](https://github.com/kenba/opencl3/issues/20) Restore `c_void` to program.rs.

## Version 0.2.3 (2021-05-30)

Depends on `cl3` = "0.4.0".

### New Features

* Issue [#15](https://github.com/kenba/opencl3/issues/15) It's safe to implement `Send` for most of the types.
* PR [#16](https://github.com/kenba/opencl3/pull/16) Implement Send for most of the types.
* PR [#17](https://github.com/kenba/opencl3/pull/17) Implement Send for some of the types.

## Version 0.2.2 (2021-05-22)

Depends on `cl3` = "0.3.1".

### New Features

* Issue [#13](https://github.com/kenba/opencl3/issues/13) Higher level create_sub_buffer call.
* Issue [#14](https://github.com/kenba/opencl3/issues/14) Adding Debug derives.
* Add OpenCL `cl_ext.h` functions.
* Add `Direct3D` extension methods.
* Add feature `cl_apple_setmemobjectdestructor` for `cl3`.

## Version 0.2.1 (2021-05-16)

Depends on `cl3` = "0.3".

### New Features

* Add extension `device_info` values.
* Add `OpenGL` extension functions.
* Add `OpenGL ES` extension functions.

## Version 0.2.0 (2021-04-18)

Depends on `cl3` = "0.2".

### Breaking Changes

* Issue [#10](https://github.com/kenba/opencl3/issues/10) Change the API to use String instead of ffi::CString.
* Change `set_wait_event` to take `Event` reference.

### New Features

* Issue [#9](https://github.com/kenba/opencl3/issues/9) Support running multiple instances of the same kernel simultaneously.
* Issue [#12](https://github.com/kenba/opencl3/issues/12) Improve OpenCL error handling.
* Add `from_device_type` method for `Context`.
* Add `ClMem` trait object.
* Add `CommandExecutionStatus` and `EventCommandType`.

## Version 0.1.4 (2021-03-26)

### Changes

* PR [#4](https://github.com/kenba/opencl3/pull/4) Implement Clone for CommandQueue
* Issue [#5](https://github.com/kenba/opencl3/issues/5) Consider replacing unwrap with expect for error handling.
* PR [#6](https://github.com/kenba/opencl3/pull/6) Make types Send and Sync where applicable.
* PR [#7](https://github.com/kenba/opencl3/pull/7) Implement Clone for most of the types.
* Issue [#8](https://github.com/kenba/opencl3/issues/8) Retrieving a program build log might be impossible.
* PR [#10](https://github.com/kenba/opencl3/pull/10) Replace calls to to_str with to_string for issue [#10](https://github.com/kenba/opencl3/issues/10).

## Version 0.1.3 (2021-01-16)

### Changes

* PR [#1](https://github.com/kenba/opencl3/pull/1) Add Buffer type field as PhantomData.
* Issue [#2](https://github.com/kenba/opencl3/issues/2) Consider adding PhantomData to Image and Pipe memory objects.
* PR [#3](https://github.com/kenba/opencl3/pull/3) Remove Buffer cast method.
* Remove unnecessary templates from methods.

## Version 0.1.2 (2021-01-12)

### Changes

* Remove `event_wait_list` from the `enqueue_nd_range` method.
* Add `wait` method to `event`.
* Add `opencl2_kernel_test.rs`.
* Add example to README.
* Don't raise error in `integration_test` if device is not SVM capable

## Version 0.1.1 (2021-01-04)

### Bug fixes

* Fix build on OpenCL 2.0 ICD.
* Fix integration tests on Intel Skylake.
* Get the max_work_item_dimensions from the device CommandQueue.

## Version 0.1.0 (2020-12-31)

Depends on `cl3` = "0.1".

### Features

* OpenCL objects implemented by Rust structs that manage their resources by implementing the `Drop` trait to perform [RAII](https://doc.rust-lang.org/rust-by-example/scope/raii.html), e.g. Context, Program, CommandQueue, etc.
* `safe` Rust functions that call OpenCL C API functions and return Rust Result types.
* A `Vec` implemented using OpenCL Shared Virtual Memory (SVM), see [svm](src/svm.rs).
