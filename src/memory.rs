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

use core::marker::PhantomData;

pub use cl3::memory::*;

use super::context::Context;

use super::Result;
use cl3::memory;
use cl3::sampler;
#[allow(unused_imports)]
use cl3::types::{
    cl_addressing_mode, cl_bool, cl_buffer_create_type, cl_context, cl_filter_mode, cl_image_desc,
    cl_image_format, cl_int, cl_mem, cl_mem_flags, cl_mem_object_type, cl_mem_properties,
    cl_sampler, cl_sampler_properties, cl_uint, cl_ulong,
};
use libc::{c_void, intptr_t, size_t};
use std::mem;

pub trait ClMem {
    fn get(&self) -> cl_mem;

    fn mem_type(&self) -> Result<cl_mem_object_type> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_TYPE)?.to_uint())
    }

    fn flags(&self) -> Result<cl_mem_flags> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_FLAGS)?.to_ulong())
    }

    fn size(&self) -> Result<size_t> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_SIZE)?.to_size())
    }

    fn host_ptr(&self) -> Result<intptr_t> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_HOST_PTR)?.to_ptr())
    }

    fn map_count(&self) -> Result<cl_uint> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_MAP_COUNT)?.to_uint())
    }

    fn reference_count(&self) -> Result<cl_uint> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_REFERENCE_COUNT)?.to_uint())
    }

    fn context(&self) -> Result<cl_context> {
        Ok(
            memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_CONTEXT)?.to_ptr()
                as cl_context,
        )
    }

    fn associated_memobject(&self) -> Result<cl_mem> {
        Ok(
            memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_ASSOCIATED_MEMOBJECT)?.to_ptr()
                as cl_mem,
        )
    }

    fn offset(&self) -> Result<size_t> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_OFFSET)?.to_size())
    }

    fn uses_svm_pointer(&self) -> Result<cl_uint> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_USES_SVM_POINTER)?.to_uint())
    }

    // CL_VERSION_3_0
    fn properties(&self) -> Result<Vec<cl_ulong>> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_PROPERTIES)?.to_vec_ulong())
    }
}

/// An OpenCL buffer.  
/// Implements the Drop trait to call release_mem_object when the object is dropped.
pub struct Buffer<T> {
    buffer: cl_mem,
    #[doc(hidden)]
    _type: PhantomData<T>,
}

impl<T> ClMem for Buffer<T> {
    fn get(&self) -> cl_mem {
        self.buffer
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject");
    }
}

impl<T> Buffer<T> {
    pub fn new(buffer: cl_mem) -> Buffer<T> {
        Buffer {
            buffer,
            _type: PhantomData,
        }
    }

    /// Create a Buffer for a context.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `count` - the number of T objects to be allocated.
    /// * `host_ptr` - a pointer to the buffer data that may already be allocated
    /// by the application.
    ///
    /// returns a Result containing the new OpenCL buffer object
    /// or the error code from the OpenCL C API function.
    pub fn create(
        context: &Context,
        flags: cl_mem_flags,
        count: size_t,
        host_ptr: *mut c_void,
    ) -> Result<Buffer<T>> {
        let buffer =
            memory::create_buffer(context.get(), flags, count * mem::size_of::<T>(), host_ptr)?;
        Ok(Buffer::new(buffer))
    }

    /// Create an OpenCL buffer object for a context.  
    /// CL_VERSION_3_0
    ///
    /// * `context` - a valid OpenCL context.
    /// * `properties` - an optional null terminated list of properties.
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `count` - the number of T objects to be allocated.
    /// * `host_ptr` - a pointer to the buffer data that may already be allocated
    /// by the application.
    ///
    /// returns a Result containing the new OpenCL buffer object
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_3_0")]
    pub fn create_with_properties(
        context: &Context,
        properties: *const cl_mem_properties,
        flags: cl_mem_flags,
        count: size_t,
        host_ptr: *mut c_void,
    ) -> Result<Buffer<T>> {
        let buffer = memory::create_buffer_with_properties(
            context.get(),
            properties,
            flags,
            count * mem::size_of::<T>(),
            host_ptr,
        )?;
        Ok(Buffer::new(buffer))
    }

    /// Create an new OpenCL buffer object from an existing buffer object.  
    ///
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the sub-buffer memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `buffer_create_type`,`buffer_create_info` - describe the type of
    /// buffer object to be created, see:
    /// [SubBuffer Attributes](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#subbuffer-create-info-table).
    ///
    /// returns a Result containing the new OpenCL buffer object
    /// or the error code from the OpenCL C API function.
    pub fn create_sub_buffer(
        &self,
        flags: cl_mem_flags,
        buffer_create_type: cl_buffer_create_type,
        buffer_create_info: *const c_void,
    ) -> Result<Buffer<T>> {
        let buffer =
            memory::create_sub_buffer(self.buffer, flags, buffer_create_type, buffer_create_info)?;
        Ok(Buffer::new(buffer))
    }
}

/// An OpenCL image.  
/// Has methods to return information from calls to clGetImageInfo with the
/// appropriate parameters.  
/// Implements the Drop trait to call release_mem_object when the object is dropped.
pub struct Image {
    image: cl_mem,
}

impl ClMem for Image {
    fn get(&self) -> cl_mem {
        self.image
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject");
    }
}

impl Image {
    pub fn new(image: cl_mem) -> Image {
        Image { image }
    }

    /// Create an OpenCL image object for a context.  
    ///
    /// * `context` - a valid OpenCL context.
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `image_format` - a pointer to a structure that describes format properties
    /// of the image to be allocated.
    /// * `image_desc` - a pointer to a structure that describes type and dimensions
    /// of the image to be allocated.
    /// * `host_ptr` - a pointer to the image data that may already be allocated
    /// by the application.
    ///
    /// returns a Result containing the new OpenCL image object
    /// or the error code from the OpenCL C API function.
    pub fn create(
        context: &Context,
        flags: cl_mem_flags,
        image_format: *const cl_image_format,
        image_desc: *const cl_image_desc,
        host_ptr: *mut c_void,
    ) -> Result<Image> {
        let image = memory::create_image(context.get(), flags, image_format, image_desc, host_ptr)?;
        Ok(Image::new(image))
    }

    /// Create an OpenCL image object for a context.  
    /// CL_VERSION_3_0
    ///
    /// * `context` - a valid OpenCL context.
    /// * `properties` - an optional null terminated list of properties.
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `image_format` - a pointer to a structure that describes format properties
    /// of the image to be allocated.
    /// * `image_desc` - a pointer to a structure that describes type and dimensions
    /// of the image to be allocated.
    /// * `host_ptr` - a pointer to the image data that may already be allocated
    /// by the application.
    ///
    /// returns a Result containing the new OpenCL image object
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "CL_VERSION_3_0")]
    pub fn create_with_properties(
        context: &Context,
        properties: *const cl_mem_properties,
        flags: cl_mem_flags,
        image_format: *const cl_image_format,
        image_desc: *const cl_image_desc,
        host_ptr: *mut c_void,
    ) -> Result<Image> {
        let image = memory::create_image_with_properties(
            context.get(),
            properties,
            flags,
            image_format,
            image_desc,
            host_ptr,
        )?;
        Ok(Image::new(image))
    }

    pub fn format(&self) -> Result<Vec<cl_image_format>> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_FORMAT)?.to_vec_image_format())
    }

    pub fn element_size(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_ELEMENT_SIZE)?.to_size())
    }

    pub fn row_pitch(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_ROW_PITCH)?.to_size())
    }

    pub fn slice_pitch(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_SLICE_PITCH)?.to_size())
    }

    pub fn width(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_WIDTH)?.to_size())
    }

    pub fn height(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_HEIGHT)?.to_size())
    }

    pub fn depth(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_DEPTH)?.to_size())
    }
    pub fn array_size(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_ARRAY_SIZE)?.to_size())
    }

    pub fn buffer(&self) -> Result<cl_mem> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_BUFFER)?.to_ptr() as cl_mem)
    }

    pub fn num_mip_levels(&self) -> Result<cl_uint> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_NUM_MIP_LEVELS)?.to_uint())
    }

    pub fn num_samples(&self) -> Result<cl_uint> {
        Ok(memory::get_image_info(self.image, ImageInfo::CL_IMAGE_NUM_SAMPLES)?.to_uint())
    }
}

/// An OpenCL sampler.  
/// Has methods to return information from calls to clGetSamplerInfo with the
/// appropriate parameters.  
/// Implements the Drop trait to call release_sampler when the object is dropped.
pub struct Sampler {
    sampler: cl_sampler,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        sampler::release_sampler(self.sampler).expect("Error: clReleaseSampler");
    }
}

impl Sampler {
    pub fn new(sampler: cl_sampler) -> Sampler {
        Sampler { sampler }
    }

    pub fn create(
        context: &Context,
        normalize_coords: cl_bool,
        addressing_mode: cl_addressing_mode,
        filter_mode: cl_filter_mode,
    ) -> Result<Sampler> {
        let sampler = sampler::create_sampler(
            context.get(),
            normalize_coords,
            addressing_mode,
            filter_mode,
        )?;
        Ok(Sampler::new(sampler))
    }

    pub fn create_with_properties(
        context: &Context,
        properties: *const cl_sampler_properties,
    ) -> Result<Sampler> {
        let sampler = sampler::create_sampler_with_properties(context.get(), properties)?;
        Ok(Sampler::new(sampler))
    }

    pub fn get(&self) -> cl_sampler {
        self.sampler
    }
}

/// An OpenCL pipe.  
/// Has methods to return information from calls to clGetPipeInfo with the
/// appropriate parameters.  
/// Implements the Drop trait to call release_mem_object when the object is dropped.
pub struct Pipe {
    pipe: cl_mem,
}

impl ClMem for Pipe {
    fn get(&self) -> cl_mem {
        self.pipe
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject");
    }
}

impl Pipe {
    pub fn new(pipe: cl_mem) -> Pipe {
        Pipe { pipe }
    }

    pub fn create(
        context: &Context,
        flags: cl_mem_flags,
        pipe_packet_size: cl_uint,
        pipe_max_packets: cl_uint,
    ) -> Result<Pipe> {
        let pipe = memory::create_pipe(context.get(), flags, pipe_packet_size, pipe_max_packets)?;
        Ok(Pipe::new(pipe))
    }

    pub fn pipe_packet_size(&self) -> Result<cl_uint> {
        Ok(memory::get_pipe_info(self.get(), PipeInfo::CL_PIPE_PACKET_SIZE)?.to_uint())
    }

    pub fn pipe_max_packets(&self) -> Result<cl_uint> {
        Ok(memory::get_pipe_info(self.get(), PipeInfo::CL_PIPE_MAX_PACKETS)?.to_uint())
    }

    pub fn pipe_properties(&self) -> Result<Vec<intptr_t>> {
        Ok(memory::get_pipe_info(self.get(), PipeInfo::CL_PIPE_PROPERTIES)?.to_vec_intptr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::device::{Device, CL_DEVICE_TYPE_GPU};
    use crate::platform::get_platforms;
    use crate::types::cl_float;
    use std::ptr;

    #[test]
    fn test_memory_buffer() {
        let platforms = get_platforms().unwrap();
        assert!(0 < platforms.len());

        // Get the first platform
        let platform = &platforms[0];

        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
        assert!(0 < devices.len());

        // Get the first device
        let device = Device::new(devices[0]);
        let context = Context::from_device(&device).unwrap();

        const ARRAY_SIZE: usize = 1024;

        let buffer =
            Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())
                .unwrap();

        let value = buffer.mem_type().unwrap();
        println!("buffer.mem_type(): {}", value);
        assert_eq!(CL_MEM_OBJECT_BUFFER, value);

        let value = buffer.flags().unwrap();
        println!("buffer.flags(): {}", value);
        assert_eq!(CL_MEM_WRITE_ONLY, value);

        let value = buffer.size().unwrap();
        println!("buffer.size(): {}", value);
        assert_eq!(ARRAY_SIZE * mem::size_of::<cl_float>(), value);

        let value = buffer.host_ptr().unwrap();
        println!("buffer.host_ptr(): {:?}", value);
        assert_eq!(0, value);

        let value = buffer.map_count().unwrap();
        println!("buffer.map_count(): {}", value);
        assert_eq!(0, value);

        let value = buffer.reference_count().unwrap();
        println!("buffer.reference_count(): {}", value);
        assert_eq!(1, value);

        let value = buffer.context().unwrap();
        assert!(context.get() == value);

        let value = buffer.associated_memobject().unwrap() as intptr_t;
        println!("buffer.associated_memobject(): {:?}", value);
        assert_eq!(0, value);

        let value = buffer.offset().unwrap();
        println!("buffer.offset(): {}", value);
        assert_eq!(0, value);

        let value = buffer.uses_svm_pointer().unwrap();
        println!("buffer.uses_svm_pointer(): {}", value);
        assert_eq!(0, value);

        // CL_VERSION_3_0
        match buffer.properties() {
            Ok(value) => {
                println!("buffer.properties: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_MEM_PROPERTIES: {:?}, {}", e, e),
        }
    }
}
