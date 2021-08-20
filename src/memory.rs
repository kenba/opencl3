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

#[allow(unused_imports)]
pub use cl3::ffi::cl_ext::cl_mem_properties_intel;
pub use cl3::memory::*;

use super::context::Context;

use super::Result;
#[allow(unused_imports)]
use cl3::d3d10;
#[allow(unused_imports)]
use cl3::d3d11;
#[allow(unused_imports)]
use cl3::dx9_media_sharing;
#[allow(unused_imports)]
use cl3::egl;
#[allow(unused_imports)]
use cl3::ext;
#[allow(unused_imports)]
use cl3::ffi::cl_d3d10::{ID3D10Buffer_ptr, ID3D10Texture2D_ptr, ID3D10Texture3D_ptr};
#[allow(unused_imports)]
use cl3::ffi::cl_d3d11::{ID3D11Buffer_ptr, ID3D11Texture2D_ptr, ID3D11Texture3D_ptr};
#[allow(unused_imports)]
use cl3::ffi::cl_dx9_media_sharing::{
    cl_dx9_media_adapter_type_khr, IDirect3DSurface9_ptr, HANDLE,
};
use cl3::gl;
use cl3::memory;
use cl3::sampler;
#[allow(unused_imports)]
use cl3::types::{
    cl_addressing_mode, cl_bool, cl_buffer_create_type, cl_buffer_region, cl_context,
    cl_filter_mode, cl_image_desc, cl_image_format, cl_int, cl_mem, cl_mem_flags,
    cl_mem_object_type, cl_mem_properties, cl_sampler, cl_sampler_properties, cl_uint, cl_ulong,
    CL_FALSE,
};
use libc::{c_void, intptr_t, size_t};
use std::mem;

pub trait ClMem {
    fn get(&self) -> cl_mem;

    fn get_mut(&mut self) -> cl_mem;

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

    /// CL_VERSION_3_0
    fn properties(&self) -> Result<Vec<cl_ulong>> {
        Ok(memory::get_mem_object_info(self.get(), MemInfo::CL_MEM_PROPERTIES)?.to_vec_ulong())
    }

    /// Query an OpenGL object used to create an OpenCL memory object.  
    ///
    /// returns a Result containing the OpenGL object type and name
    /// or the error code from the OpenCL C API function.
    fn gl_object_info(&self) -> Result<(gl::gl_uint, gl::gl_uint)> {
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
        memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject");
    }
}

unsafe impl<T: Send> Send for Buffer<T> {}

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
    pub fn create_from_gl_buffer(
        context: &Context,
        flags: cl_mem_flags,
        bufobj: gl::gl_uint,
    ) -> Result<Buffer<T>> {
        let buffer = gl::create_from_gl_buffer(context.get(), flags, bufobj)?;
        Ok(Buffer::new(buffer))
    }

    #[cfg(feature = "cl_intel_create_buffer_with_properties")]
    pub fn create_with_properties_intel(
        context: &Context,
        properties: *const cl_mem_properties_intel,
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
    pub fn create_sub_buffer(
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
        memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject");
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
    pub fn create_from_gl_texture(
        context: &Context,
        flags: cl_mem_flags,
        texture_target: gl::gl_enum,
        miplevel: gl::gl_int,
        texture: gl::gl_uint,
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
    pub fn create_from_gl_render_buffer(
        context: &Context,
        flags: cl_mem_flags,
        renderbuffer: gl::gl_uint,
    ) -> Result<Image> {
        let image = gl::create_from_gl_render_buffer(context.get(), flags, renderbuffer)?;
        Ok(Image::new(image))
    }

    /// Create an OpenCL 2D image object from an OpenGL 2D texture object,
    /// or a single face of an OpenGL cubemap texture object.  
    /// Deprecated in CL_VERSION_1_2, use create_from_gl_texture.  
    ///
    /// * `context` - a valid OpenCL context created from an OpenGL context.
    /// * `flags` - a bit-field used to specify allocation and usage information
    /// about the image memory object being created, see:
    /// [Memory Flags](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-flags-table).
    /// * `texture_target` - used to define the image type of texture.  
    /// * `miplevel ` - used to define the mipmap level.  
    /// * `texture  ` - the name of a GL 2D, cubemap or rectangle texture object.  
    ///
    /// returns a Result containing the new OpenCL image object
    /// or the error code from the OpenCL C API function.
    pub fn create_from_gl_texture_2d(
        context: &Context,
        flags: cl_mem_flags,
        texture_target: gl::gl_enum,
        miplevel: gl::gl_int,
        texture: gl::gl_uint,
    ) -> Result<Image> {
        let image =
            gl::create_from_gl_texture_2d(context.get(), flags, texture_target, miplevel, texture)?;
        Ok(Image::new(image))
    }

    /// Create an OpenCL 3D image object from an OpenGL 3D texture object.   
    /// Deprecated in CL_VERSION_1_2, use create_from_gl_texture.  
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
    pub fn create_from_gl_texture_3d(
        context: &Context,
        flags: cl_mem_flags,
        texture_target: gl::gl_enum,
        miplevel: gl::gl_int,
        texture: gl::gl_uint,
    ) -> Result<Image> {
        let image =
            gl::create_from_gl_texture_3d(context.get(), flags, texture_target, miplevel, texture)?;
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
    pub fn create_from_egl_image(
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

    #[cfg(feature = "cl_khr_dx9_media_sharing")]
    #[inline]
    pub fn create_from_dx9_media_surface_khr(
        context: &Context,
        flags: cl_mem_flags,
        adapter_type: cl_dx9_media_adapter_type_khr,
        surface_info: *mut c_void,
        plane: cl_uint,
    ) -> Result<Image> {
        let image = dx9_media_sharing::create_from_dx9_media_surface_khr(
            context.get(),
            flags,
            adapter_type,
            surface_info,
            plane,
        )?;
        Ok(Image::new(image))
    }

    #[cfg(feature = "cl_intel_dx9_media_sharing")]
    #[inline]
    pub fn create_from_dx9_media_surface_intel(
        context: &Context,
        flags: cl_mem_flags,
        resource: IDirect3DSurface9_ptr,
        shared_handle: HANDLE,
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

    #[cfg(feature = "cl_khr_d3d10_sharing")]
    #[inline]
    pub fn create_from_d3d10_buffer_khr(
        context: &Context,
        flags: cl_mem_flags,
        resource: ID3D10Buffer_ptr,
    ) -> Result<Image> {
        let image = d3d10::create_from_d3d10_buffer_khr(context.get(), flags, resource)?;
        Ok(Image::new(image))
    }

    #[cfg(feature = "cl_khr_d3d10_sharing")]
    #[inline]
    pub fn create_from_d3d10_texture2d_khr(
        context: &Context,
        flags: cl_mem_flags,
        resource: ID3D10Texture2D_ptr,
        subresource: cl_uint,
    ) -> Result<Image> {
        let image =
            d3d10::create_from_d3d10_texture2d_khr(context.get(), flags, resource, subresource)?;
        Ok(Image::new(image))
    }

    #[cfg(feature = "cl_khr_d3d10_sharing")]
    #[inline]
    pub fn create_from_d3d10_texture3d_khr(
        context: &Context,
        flags: cl_mem_flags,
        resource: ID3D10Texture3D_ptr,
        subresource: cl_uint,
    ) -> Result<Image> {
        let image =
            d3d10::create_from_d3d10_texture3d_khr(context.get(), flags, resource, subresource)?;
        Ok(Image::new(image))
    }

    #[cfg(feature = "cl_khr_d3d11_sharing")]
    #[inline]
    pub fn create_from_d3d11_buffer_khr(
        context: &Context,
        flags: cl_mem_flags,
        resource: ID3D11Buffer_ptr,
    ) -> Result<Image> {
        let image = d3d11::create_from_d3d11_buffer_khr(context.get(), flags, resource)?;
        Ok(Image::new(image))
    }

    #[cfg(feature = "cl_khr_d3d11_sharing")]
    #[inline]
    pub fn create_from_d3d11_texture2d_khr(
        context: &Context,
        flags: cl_mem_flags,
        resource: ID3D11Texture2D_ptr,
        subresource: cl_uint,
    ) -> Result<Image> {
        let image =
            d3d11::create_from_d3d11_texture2d_khr(context.get(), flags, resource, subresource)?;
        Ok(Image::new(image))
    }

    #[cfg(feature = "cl_khr_d3d11_sharing")]
    #[inline]
    pub fn create_from_d3d11_texture3d_khr(
        context: &Context,
        flags: cl_mem_flags,
        resource: ID3D11Texture3D_ptr,
        subresource: cl_uint,
    ) -> Result<Image> {
        let image =
            d3d11::create_from_d3d11_texture3d_khr(context.get(), flags, resource, subresource)?;
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

    ///  Get information about the GL texture target associated with a memory object.
    pub fn gl_texture_target(&self) -> Result<cl_uint> {
        Ok(gl::get_gl_texture_info(self.image, gl::TextureInfo::CL_GL_TEXTURE_TARGET)?.to_uint())
    }

    /// Get information about the GL mipmap level associated with a memory object.
    pub fn gl_mipmap_level(&self) -> Result<cl_int> {
        Ok(gl::get_gl_texture_info(self.image, gl::TextureInfo::CL_GL_MIPMAP_LEVEL)?.to_int())
    }

    ///  Get information about the GL number of samples associated with a memory object.
    pub fn gl_num_samples(&self) -> Result<cl_int> {
        Ok(gl::get_gl_texture_info(self.image, gl::TextureInfo::CL_GL_NUM_SAMPLES)?.to_int())
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

impl Drop for Sampler {
    fn drop(&mut self) {
        sampler::release_sampler(self.sampler).expect("Error: clReleaseSampler");
    }
}

unsafe impl Send for Sampler {}

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
        Ok(
            sampler::get_sampler_info(
                self.get(),
                sampler::SamplerInfo::CL_SAMPLER_REFERENCE_COUNT,
            )?
            .to_uint(),
        )
    }

    pub fn context(&self) -> Result<cl_context> {
        Ok(
            sampler::get_sampler_info(self.get(), sampler::SamplerInfo::CL_SAMPLER_CONTEXT)?
                .to_ptr() as cl_context,
        )
    }

    pub fn normalized_coords(&self) -> Result<bool> {
        Ok(sampler::get_sampler_info(
            self.get(),
            sampler::SamplerInfo::CL_SAMPLER_NORMALIZED_COORDS,
        )?
        .to_uint()
            != CL_FALSE)
    }

    pub fn addressing_mode(&self) -> Result<cl_addressing_mode> {
        Ok(
            sampler::get_sampler_info(
                self.get(),
                sampler::SamplerInfo::CL_SAMPLER_ADDRESSING_MODE,
            )?
            .to_uint(),
        )
    }

    pub fn filter_mode(&self) -> Result<cl_filter_mode> {
        Ok(
            sampler::get_sampler_info(self.get(), sampler::SamplerInfo::CL_SAMPLER_FILTER_MODE)?
                .to_uint(),
        )
    }

    pub fn sampler_properties(&self) -> Result<Vec<intptr_t>> {
        Ok(
            sampler::get_sampler_info(self.get(), sampler::SamplerInfo::CL_SAMPLER_PROPERTIES)?
                .to_vec_intptr(),
        )
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
        memory::release_mem_object(self.get()).expect("Error: clReleaseMemObject");
    }
}

#[cfg(feature = "CL_VERSION_2_0")]
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
