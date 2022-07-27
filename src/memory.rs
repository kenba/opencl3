// Copyright (c) 2020-2022 Via Technology Ltd.
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

#![allow(deprecated)]
#![allow(clippy::missing_safety_doc)]

use core::marker::PhantomData;

pub use cl3::memory::*;

use super::context::Context;

use super::Result;
#[cfg(feature = "cl_intel_dx9_media_sharing")]
use cl3::dx9_media_sharing;
#[allow(unused_imports)]
use cl3::egl;
#[allow(unused_imports)]
use cl3::ext;
use cl3::gl;
use cl3::memory;
use cl3::sampler;

#[allow(unused_imports)]
use cl3::types::{
    cl_addressing_mode, cl_bool, cl_buffer_create_type, cl_buffer_region, cl_context,
    cl_filter_mode, cl_image_desc, cl_image_format, cl_image_info, cl_int, cl_mem, cl_mem_flags,
    cl_mem_info, cl_mem_object_type, cl_mem_properties, cl_pipe_info, cl_sampler, cl_sampler_info,
    cl_sampler_properties, cl_uint, cl_ulong, CL_FALSE,
};

use libc::{c_void, intptr_t, size_t};
use std::mem;

pub trait ClMem {
    fn get(&self) -> cl_mem;

    fn get_mut(&mut self) -> cl_mem;

    fn mem_type(&self) -> Result<cl_mem_object_type> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_TYPE)?.into())
    }

    fn flags(&self) -> Result<cl_mem_flags> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_FLAGS)?.into())
    }

    fn size(&self) -> Result<size_t> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_SIZE)?.into())
    }

    fn host_ptr(&self) -> Result<intptr_t> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_HOST_PTR)?.into())
    }

    fn map_count(&self) -> Result<cl_uint> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_MAP_COUNT)?.into())
    }

    fn reference_count(&self) -> Result<cl_uint> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_REFERENCE_COUNT)?.into())
    }

    fn context(&self) -> Result<cl_context> {
        Ok(intptr_t::from(memory::get_mem_object_info(self.get(), CL_MEM_CONTEXT)?) as cl_context)
    }

    fn associated_memobject(&self) -> Result<cl_mem> {
        Ok(intptr_t::from(memory::get_mem_object_info(
            self.get(),
            CL_MEM_ASSOCIATED_MEMOBJECT,
        )?) as cl_mem)
    }

    fn offset(&self) -> Result<size_t> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_OFFSET)?.into())
    }

    fn uses_svm_pointer(&self) -> Result<cl_uint> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_USES_SVM_POINTER)?.into())
    }

    /// CL_VERSION_3_0
    fn properties(&self) -> Result<Vec<cl_ulong>> {
        Ok(memory::get_mem_object_info(self.get(), CL_MEM_PROPERTIES)?.into())
    }

    /// Get memory data about an OpenCL memory object.
    /// Calls clGetMemObjectInfo to get the desired data about the memory object.
    fn get_mem_data(&self, param_name: cl_mem_info) -> Result<Vec<u8>> {
        Ok(get_mem_object_data(self.get(), param_name)?)
    }

    /// Query an OpenGL object used to create an OpenCL memory object.  
    ///
    /// returns a Result containing the OpenGL object type and name
    /// or the error code from the OpenCL C API function.
    fn gl_object_info(&self) -> Result<(gl::cl_GLuint, gl::cl_GLuint)> {
        Ok(gl::get_gl_object_info(self.get())?)
    }
}

/// An OpenCL buffer.  
/// Implements the Drop trait to call release_mem_object when the object is dropped.
#[derive(Debug)]
pub struct Buffer<T> {
    buffer: cl_mem,
    #[doc(hidden)]
    _type: PhantomData<T>,
}

impl<T> From<Buffer<T>> for cl_mem {
    fn from(value: Buffer<T>) -> Self {
        value.buffer
    }
}

impl<T> ClMem for Buffer<T> {
    fn get(&self) -> cl_mem {
        self.buffer
    }

    fn get_mut(&mut self) -> cl_mem {
        self.buffer
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe { memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject") };
    }
}

unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}

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
    pub unsafe fn create(
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
    pub unsafe fn create_with_properties(
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

    /// Create an OpenCL buffer object for a context from an OpenGL buffer.  
    ///
    /// * `context` - a valid OpenCL context created from an OpenGL context.
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `bufobj` - the OpenGL buffer.  
    ///
    /// returns a Result containing the new OpenCL buffer object
    /// or the error code from the OpenCL C API function.
    pub unsafe fn create_from_gl_buffer(
        context: &Context,
        flags: cl_mem_flags,
        bufobj: gl::cl_GLuint,
    ) -> Result<Buffer<T>> {
        let buffer = gl::create_from_gl_buffer(context.get(), flags, bufobj)?;
        Ok(Buffer::new(buffer))
    }

    #[cfg(feature = "cl_intel_create_buffer_with_properties")]
    pub unsafe fn create_with_properties_intel(
        context: &Context,
        properties: *const ext::cl_mem_properties_intel,
        flags: cl_mem_flags,
        count: size_t,
        host_ptr: *mut c_void,
    ) -> Result<Buffer<T>> {
        let buffer = ext::create_buffer_with_properties_intel(
            context.get(),
            properties,
            flags,
            count * mem::size_of::<T>(),
            host_ptr,
        )?;
        Ok(Buffer::new(buffer))
    }

    /// Create an new OpenCL buffer object from an existing buffer object.  
    /// See: [SubBuffer Attributes](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#subbuffer-create-info-table).  
    ///
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the sub-buffer memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `origin` - the offset in number of objects of type <T>.
    /// * `count` - the size of the sub-buffer in number of objects of type <T>.
    ///
    /// returns a Result containing the new OpenCL buffer object
    /// or the error code from the OpenCL C API function.
    pub unsafe fn create_sub_buffer(
        &self,
        flags: cl_mem_flags,
        origin: usize,
        count: usize,
    ) -> Result<Buffer<T>> {
        let buffer_create_info = cl_buffer_region {
            origin: origin * std::mem::size_of::<T>(),
            size: count * std::mem::size_of::<T>(),
        };
        let buffer = memory::create_sub_buffer(
            self.buffer,
            flags,
            CL_BUFFER_CREATE_TYPE_REGION,
            &buffer_create_info as *const _ as *const c_void,
        )?;
        Ok(Buffer::new(buffer))
    }
}

/// An OpenCL image.  
/// Has methods to return information from calls to clGetImageInfo with the
/// appropriate parameters.  
/// Implements the Drop trait to call release_mem_object when the object is dropped.
#[derive(Debug)]
pub struct Image {
    image: cl_mem,
}

impl From<Image> for cl_mem {
    fn from(value: Image) -> Self {
        value.image
    }
}

impl ClMem for Image {
    fn get(&self) -> cl_mem {
        self.image
    }

    fn get_mut(&mut self) -> cl_mem {
        self.image
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject") };
    }
}

unsafe impl Send for Image {}

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
    #[cfg(feature = "CL_VERSION_1_2")]
    pub unsafe fn create(
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
    pub unsafe fn create_with_properties(
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

    /// Create an OpenCL image object, image array object, or image buffer object
    /// for a context from an OpenGL texture object, texture array object,
    /// texture buffer object, or a single face of an OpenGL cubemap texture object.  
    ///
    /// * `context` - a valid OpenCL context created from an OpenGL context.
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `texture_target` - used to define the image type of texture.  
    /// * `miplevel ` - used to define the mipmap level.  
    /// * `texture  ` - the name of a GL buffer texture object.
    ///
    /// returns a Result containing the new OpenCL image object
    /// or the error code from the OpenCL C API function.
    pub unsafe fn create_from_gl_texture(
        context: &Context,
        flags: cl_mem_flags,
        texture_target: gl::cl_GLenum,
        miplevel: gl::cl_GLint,
        texture: gl::cl_GLuint,
    ) -> Result<Image> {
        let image =
            gl::create_from_gl_texture(context.get(), flags, texture_target, miplevel, texture)?;
        Ok(Image::new(image))
    }

    /// Create an OpenCL 2D image object from an OpenGL renderbuffer object.  
    ///
    /// * `context` - a valid OpenCL context created from an OpenGL context.
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `renderbuffer`  - a GL renderbuffer object.  
    ///
    /// returns a Result containing the new OpenCL image object
    /// or the error code from the OpenCL C API function.
    pub unsafe fn create_from_gl_render_buffer(
        context: &Context,
        flags: cl_mem_flags,
        renderbuffer: gl::cl_GLuint,
    ) -> Result<Image> {
        let image = gl::create_from_gl_render_buffer(context.get(), flags, renderbuffer)?;
        Ok(Image::new(image))
    }

    /// Create an OpenCL image object, from the EGLImage source provided as image.  
    /// Requires the cl_khr_egl_image extension.  
    ///
    /// * `context` - a valid OpenCL context created from an OpenGL context.
    /// * `display` - should be of type EGLDisplay, cast into the type CLeglDisplayKHR
    /// * `image` - should be of type EGLImageKHR, cast into the type CLeglImageKHR.  
    /// * `flags` -  usage information about the memory object being created.  
    /// * `properties` - a null terminated list of property names and their
    /// corresponding values.  
    ///
    /// returns a Result containing the new OpenCL image object
    /// or the error code from the OpenCL C API function.
    #[cfg(feature = "cl_khr_egl_image")]
    #[inline]
    pub unsafe fn create_from_egl_image(
        context: &Context,
        display: egl::CLeglDisplayKHR,
        image: egl::CLeglImageKHR,
        flags: cl_mem_flags,
        properties: &[egl::cl_egl_image_properties_khr],
    ) -> Result<Image> {
        let image =
            egl::create_from_egl_image(context.get(), display, image, flags, properties.as_ptr())?;
        Ok(Image::new(image))
    }

    #[cfg(feature = "cl_intel_dx9_media_sharing")]
    #[inline]
    pub unsafe fn create_from_dx9_media_surface_intel(
        context: &Context,
        flags: cl_mem_flags,
        resource: dx9_media_sharing::IDirect3DSurface9_ptr,
        shared_handle: dx9_media_sharing::HANDLE,
        plane: cl_uint,
    ) -> Result<Image> {
        let image = dx9_media_sharing::create_from_dx9_media_surface_intel(
            context.get(),
            flags,
            resource,
            shared_handle,
            plane,
        )?;
        Ok(Image::new(image))
    }

    pub fn format(&self) -> Result<Vec<cl_image_format>> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_FORMAT)?.into())
    }

    pub fn element_size(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_ELEMENT_SIZE)?.into())
    }

    pub fn row_pitch(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_ROW_PITCH)?.into())
    }

    pub fn slice_pitch(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_SLICE_PITCH)?.into())
    }

    pub fn width(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_WIDTH)?.into())
    }

    pub fn height(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_HEIGHT)?.into())
    }

    pub fn depth(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_DEPTH)?.into())
    }
    pub fn array_size(&self) -> Result<size_t> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_ARRAY_SIZE)?.into())
    }

    pub fn buffer(&self) -> Result<cl_mem> {
        Ok(intptr_t::from(memory::get_image_info(self.image, CL_IMAGE_BUFFER)?) as cl_mem)
    }

    pub fn num_mip_levels(&self) -> Result<cl_uint> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_NUM_MIP_LEVELS)?.into())
    }

    pub fn num_samples(&self) -> Result<cl_uint> {
        Ok(memory::get_image_info(self.image, CL_IMAGE_NUM_SAMPLES)?.into())
    }

    /// Get data about an OpenCL image object.
    /// Calls clGetImageInfo to get the desired data about the image object.
    pub fn get_data(&self, param_name: cl_image_info) -> Result<Vec<u8>> {
        Ok(get_image_data(self.image, param_name)?)
    }

    ///  Get information about the GL texture target associated with a memory object.
    pub fn gl_texture_target(&self) -> Result<cl_uint> {
        Ok(gl::get_gl_texture_info(self.image, gl::CL_GL_TEXTURE_TARGET)?.into())
    }

    /// Get information about the GL mipmap level associated with a memory object.
    pub fn gl_mipmap_level(&self) -> Result<cl_int> {
        Ok(gl::get_gl_texture_info(self.image, gl::CL_GL_MIPMAP_LEVEL)?.into())
    }

    ///  Get information about the GL number of samples associated with a memory object.
    pub fn gl_num_samples(&self) -> Result<cl_int> {
        Ok(gl::get_gl_texture_info(self.image, gl::CL_GL_NUM_SAMPLES)?.into())
    }

    ///  Get GL texture information associated with a memory object.
    pub fn get_gl_texture_data(&self, param_name: gl::cl_gl_texture_info) -> Result<Vec<u8>> {
        Ok(gl::get_gl_texture_data(self.image, param_name)?)
    }
}

/// An OpenCL sampler.  
/// Has methods to return information from calls to clGetSamplerInfo with the
/// appropriate parameters.  
/// Implements the Drop trait to call release_sampler when the object is dropped.
#[derive(Debug)]
pub struct Sampler {
    sampler: cl_sampler,
}

impl From<Sampler> for cl_sampler {
    fn from(value: Sampler) -> Self {
        value.sampler
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { sampler::release_sampler(self.sampler).expect("Error: clReleaseSampler") };
    }
}

unsafe impl Send for Sampler {}

impl Sampler {
    pub fn new(sampler: cl_sampler) -> Sampler {
        Sampler { sampler }
    }

    #[cfg_attr(
        any(
            feature = "CL_VERSION_2_0",
            feature = "CL_VERSION_2_1",
            feature = "CL_VERSION_2_2",
            feature = "CL_VERSION_3_0"
        ),
        deprecated(
            since = "0.1.0",
            note = "From CL_VERSION_2_0 use create_sampler_with_properties"
        )
    )]
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

    #[cfg(feature = "CL_VERSION_2_0")]
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

    pub fn reference_count(&self) -> Result<cl_uint> {
        Ok(sampler::get_sampler_info(self.get(), sampler::CL_SAMPLER_REFERENCE_COUNT)?.into())
    }

    pub fn context(&self) -> Result<cl_context> {
        Ok(intptr_t::from(sampler::get_sampler_info(
            self.get(),
            sampler::CL_SAMPLER_CONTEXT,
        )?) as cl_context)
    }

    pub fn normalized_coords(&self) -> Result<bool> {
        Ok(cl_uint::from(sampler::get_sampler_info(
            self.get(),
            sampler::CL_SAMPLER_NORMALIZED_COORDS,
        )?) != CL_FALSE)
    }

    pub fn addressing_mode(&self) -> Result<cl_addressing_mode> {
        Ok(sampler::get_sampler_info(self.get(), sampler::CL_SAMPLER_ADDRESSING_MODE)?.into())
    }

    pub fn filter_mode(&self) -> Result<cl_filter_mode> {
        Ok(sampler::get_sampler_info(self.get(), sampler::CL_SAMPLER_FILTER_MODE)?.into())
    }

    pub fn sampler_properties(&self) -> Result<Vec<intptr_t>> {
        Ok(sampler::get_sampler_info(self.get(), sampler::CL_SAMPLER_PROPERTIES)?.into())
    }

    /// Get data about an OpenCL sampler object.
    /// Calls clGetDeviceInfo to get the desired data about the sampler object.
    pub fn get_data(&self, param_name: cl_sampler_info) -> Result<Vec<u8>> {
        Ok(sampler::get_sampler_data(self.get(), param_name)?)
    }
}

/// An OpenCL pipe.  
/// Has methods to return information from calls to clGetPipeInfo with the
/// appropriate parameters.  
/// Implements the Drop trait to call release_mem_object when the object is dropped.
#[cfg(feature = "CL_VERSION_2_0")]
#[derive(Debug)]
pub struct Pipe {
    pipe: cl_mem,
}

#[cfg(feature = "CL_VERSION_2_0")]
impl From<cl_mem> for Pipe {
    fn from(pipe: cl_mem) -> Self {
        Pipe { pipe }
    }
}

#[cfg(feature = "CL_VERSION_2_0")]
impl From<Pipe> for cl_mem {
    fn from(value: Pipe) -> Self {
        value.pipe as cl_mem
    }
}

#[cfg(feature = "CL_VERSION_2_0")]
impl ClMem for Pipe {
    fn get(&self) -> cl_mem {
        self.pipe
    }

    fn get_mut(&mut self) -> cl_mem {
        self.pipe
    }
}

#[cfg(feature = "CL_VERSION_2_0")]
impl Drop for Pipe {
    fn drop(&mut self) {
        unsafe { memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject") };
    }
}

#[cfg(feature = "CL_VERSION_2_0")]
impl Pipe {
    pub fn new(pipe: cl_mem) -> Pipe {
        Pipe { pipe }
    }

    pub unsafe fn create(
        context: &Context,
        flags: cl_mem_flags,
        pipe_packet_size: cl_uint,
        pipe_max_packets: cl_uint,
    ) -> Result<Pipe> {
        let pipe = memory::create_pipe(context.get(), flags, pipe_packet_size, pipe_max_packets)?;
        Ok(Pipe::new(pipe))
    }

    pub fn pipe_packet_size(&self) -> Result<cl_uint> {
        Ok(memory::get_pipe_info(self.get(), CL_PIPE_PACKET_SIZE)?.into())
    }

    pub fn pipe_max_packets(&self) -> Result<cl_uint> {
        Ok(memory::get_pipe_info(self.get(), CL_PIPE_MAX_PACKETS)?.into())
    }

    pub fn pipe_properties(&self) -> Result<Vec<intptr_t>> {
        Ok(memory::get_pipe_info(self.get(), CL_PIPE_PROPERTIES)?.into())
    }

    /// Get data about an OpenCL pipe object.  
    /// Calls clGetPipeInfo to get the desired information about the pipe object.
    pub fn get_data(&self, param_name: cl_pipe_info) -> Result<Vec<u8>> {
        Ok(memory::get_pipe_data(self.get(), param_name)?)
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

        let buffer = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())
                .unwrap()
        };

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
