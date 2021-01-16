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

use cl3::memory;
use cl3::sampler;
#[allow(unused_imports)]
use cl3::types::{
    cl_addressing_mode, cl_bool, cl_buffer_create_type, cl_filter_mode, cl_image_desc,
    cl_image_format, cl_int, cl_mem, cl_mem_flags, cl_mem_properties, cl_sampler,
    cl_sampler_properties, cl_uint, cl_ulong,
};
use libc::{c_void, intptr_t, size_t};
use std::mem;

pub fn get_mem_type(memobj: cl_mem) -> Result<cl_uint, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_TYPE)?;
    Ok(value.to_uint())
}

pub fn get_mem_flags(memobj: cl_mem) -> Result<cl_uint, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_FLAGS)?;
    Ok(value.to_uint())
}

pub fn get_mem_size(memobj: cl_mem) -> Result<size_t, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_SIZE)?;
    Ok(value.to_size())
}

pub fn get_mem_host_ptr(memobj: cl_mem) -> Result<intptr_t, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_HOST_PTR)?;
    Ok(value.to_ptr())
}

pub fn get_mem_map_count(memobj: cl_mem) -> Result<cl_uint, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_MAP_COUNT)?;
    Ok(value.to_uint())
}

pub fn get_mem_reference_count(memobj: cl_mem) -> Result<cl_uint, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_REFERENCE_COUNT)?;
    Ok(value.to_uint())
}

pub fn get_mem_context(memobj: cl_mem) -> Result<intptr_t, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_CONTEXT)?;
    Ok(value.to_ptr())
}

pub fn get_mem_associated_memobject(memobj: cl_mem) -> Result<intptr_t, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_ASSOCIATED_MEMOBJECT)?;
    Ok(value.to_ptr())
}

pub fn get_mem_offset(memobj: cl_mem) -> Result<size_t, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_OFFSET)?;
    Ok(value.to_size())
}

pub fn get_mem_uses_svm_pointer(memobj: cl_mem) -> Result<cl_uint, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_USES_SVM_POINTER)?;
    Ok(value.to_uint())
}

// CL_VERSION_3_0
pub fn get_mem_properties(memobj: cl_mem) -> Result<Vec<cl_ulong>, cl_int> {
    let value = memory::get_mem_object_info(memobj, MemInfo::CL_MEM_PROPERTIES)?;
    Ok(value.to_vec_ulong())
}

/// An OpenCL buffer.  
/// Implements the Drop trait to call release_mem_object when the object is dropped.
pub struct Buffer<T> {
    buffer: cl_mem,
    #[doc(hidden)]
    _type: PhantomData<T>,
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        memory::release_mem_object(self.buffer).unwrap();
    }
}

impl<T> Buffer<T> {
    pub fn new(buffer: cl_mem) -> Buffer<T> {
        Buffer { buffer, _type: PhantomData }
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
    ) -> Result<Buffer<T>, cl_int> {
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
    ) -> Result<Buffer<T>, cl_int> {
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
    ) -> Result<Buffer<T>, cl_int> {
        let buffer =
            memory::create_sub_buffer(self.buffer, flags, buffer_create_type, buffer_create_info)?;
        Ok(Buffer::new(buffer))
    }

    pub fn get(&self) -> cl_mem {
        self.buffer
    }
}

/// An OpenCL image.  
/// Has methods to return information from calls to clGetImageInfo with the
/// appropriate parameters.  
/// Implements the Drop trait to call release_mem_object when the object is dropped.
pub struct Image {
    image: cl_mem,
}

impl Drop for Image {
    fn drop(&mut self) {
        memory::release_mem_object(self.image).unwrap();
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
    ) -> Result<Image, cl_int> {
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
    ) -> Result<Image, cl_int> {
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

    pub fn get(&self) -> cl_mem {
        self.image
    }

    pub fn get_format(&self) -> Result<Vec<cl_image_format>, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_FORMAT)?;
        Ok(value.to_vec_image_format())
    }

    pub fn get_element_size(&self) -> Result<size_t, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_ELEMENT_SIZE)?;
        Ok(value.to_size())
    }

    pub fn get_row_pitch(&self) -> Result<size_t, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_ROW_PITCH)?;
        Ok(value.to_size())
    }

    pub fn get_slice_pitch(&self) -> Result<size_t, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_SLICE_PITCH)?;
        Ok(value.to_size())
    }

    pub fn get_width(&self) -> Result<size_t, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_WIDTH)?;
        Ok(value.to_size())
    }

    pub fn get_height(&self) -> Result<size_t, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_HEIGHT)?;
        Ok(value.to_size())
    }

    pub fn get_depth(&self) -> Result<size_t, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_DEPTH)?;
        Ok(value.to_size())
    }
    pub fn get_array_size(&self) -> Result<size_t, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_ARRAY_SIZE)?;
        Ok(value.to_size())
    }

    pub fn get_buffer(&self) -> Result<intptr_t, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_BUFFER)?;
        Ok(value.to_ptr())
    }

    pub fn get_num_mip_levels(&self) -> Result<cl_uint, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_NUM_MIP_LEVELS)?;
        Ok(value.to_uint())
    }

    pub fn get_num_samples(&self) -> Result<cl_uint, cl_int> {
        let value = memory::get_image_info(self.image, ImageInfo::CL_IMAGE_NUM_SAMPLES)?;
        Ok(value.to_uint())
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
        sampler::release_sampler(self.sampler).unwrap();
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
    ) -> Result<Sampler, cl_int> {
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
    ) -> Result<Sampler, cl_int> {
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

impl Drop for Pipe {
    fn drop(&mut self) {
        memory::release_mem_object(self.pipe).unwrap();
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
    ) -> Result<Pipe, cl_int> {
        let pipe = memory::create_pipe(context.get(), flags, pipe_packet_size, pipe_max_packets)?;
        Ok(Pipe::new(pipe))
    }

    pub fn get(&self) -> cl_mem {
        self.pipe
    }

    pub fn get_packet_size(&self) -> Result<cl_uint, cl_int> {
        let value = memory::get_pipe_info(self.pipe, PipeInfo::CL_PIPE_PACKET_SIZE)?;
        Ok(value.to_uint())
    }

    pub fn get_max_packets(&self) -> Result<cl_uint, cl_int> {
        let value = memory::get_pipe_info(self.pipe, PipeInfo::CL_PIPE_MAX_PACKETS)?;
        Ok(value.to_uint())
    }

    pub fn get_properties(&self) -> Result<Vec<intptr_t>, cl_int> {
        let value = memory::get_pipe_info(self.pipe, PipeInfo::CL_PIPE_PROPERTIES)?;
        Ok(value.to_vec_intptr())
    }
}
