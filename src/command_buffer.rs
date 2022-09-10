// Copyright (c) 2021-2022 Via Technology Ltd.
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

//! OpenCL Command Buffers extension. Enable with feature: cl_khr_command_buffer.

#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use super::event::Event;
use super::memory::*;
use super::Result;

#[allow(unused_imports)]
use cl3::ext::{
    cl_command_buffer_info_khr, cl_command_buffer_khr, cl_command_buffer_properties_khr,
    cl_ndrange_kernel_command_properties_khr, cl_sync_point_khr,
    command_barrier_with_wait_list_khr, command_copy_buffer_khr, command_copy_buffer_rect_khr,
    command_copy_buffer_to_image_khr, command_copy_image_khr, command_copy_image_to_buffer_khr,
    command_fill_buffer_khr, command_fill_image_khr, command_nd_range_kernel_khr,
    create_command_buffer_khr, enqueue_command_buffer_khr, finalize_command_buffer_khr,
    get_command_buffer_data_khr, get_command_buffer_info_khr, release_command_buffer_khr,
    CL_COMMAND_BUFFER_NUM_QUEUES_KHR, CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR,
    CL_COMMAND_BUFFER_QUEUES_KHR, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
    CL_COMMAND_BUFFER_STATE_KHR,
};
#[allow(unused_imports)]
use cl3::types::{cl_command_queue, cl_event, cl_kernel, cl_mem, cl_uint};
use libc::{c_void, size_t};
use std::mem;
use std::ptr;

/// An OpenCL command-buffer.  
/// This extension adds the ability to record and replay buffers of OpenCL commands.  
/// See [cl_khr_command_buffer](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_command_buffer)
#[derive(Debug)]
pub struct CommandBuffer {
    buffer: cl_command_buffer_khr,
}

impl From<CommandBuffer> for cl_command_buffer_khr {
    fn from(value: CommandBuffer) -> Self {
        value.buffer
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            release_command_buffer_khr(self.buffer).expect("Error: clReleaseCommandBufferKHR")
        };
    }
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    fn new(buffer: cl_command_buffer_khr) -> CommandBuffer {
        CommandBuffer { buffer }
    }

    /// Get the underlying OpenCL cl_command_buffer_khr.
    pub fn get(&self) -> cl_command_buffer_khr {
        self.buffer
    }

    /// Create a command-buffer that can record commands to the specified queues.
    pub fn create(
        queues: &[cl_command_queue],
        properties: &[cl_command_buffer_properties_khr],
    ) -> Result<CommandBuffer> {
        let buffer = create_command_buffer_khr(queues, properties.as_ptr())?;
        Ok(CommandBuffer::new(buffer))
    }

    /// Finalizes command recording ready for enqueuing the command-buffer on a command-queue.
    pub fn finalize(&self) -> Result<()> {
        Ok(finalize_command_buffer_khr(self.buffer)?)
    }

    /// Enqueues a command-buffer to execute on command-queues specified by queues,
    /// or on default command-queues used during recording if queues is empty.
    pub unsafe fn enqueue(
        &self,
        queues: &mut [cl_command_queue],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_command_buffer_khr(
            queues.len() as cl_uint,
            queues.as_mut_ptr(),
            self.buffer,
            event_wait_list.len() as cl_uint,
            if !event_wait_list.is_empty() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    /// Records a barrier operation used as a synchronization point.
    pub fn command_barrier_with_wait_list(
        &self,
        queue: cl_command_queue,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        unsafe {
            command_barrier_with_wait_list_khr(
                self.buffer,
                queue,
                sync_point_wait_list,
                &mut sync_point,
                ptr::null_mut(),
            )?
        };
        Ok(sync_point)
    }

    /// Records a command to copy from one buffer object to another.
    pub unsafe fn copy_buffer<T>(
        &self,
        queue: cl_command_queue,
        src_buffer: &Buffer<T>,
        dst_buffer: &mut Buffer<T>,
        src_offset: size_t,
        dst_offset: size_t,
        size: size_t,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        command_copy_buffer_khr(
            self.buffer,
            queue,
            src_buffer.get(),
            dst_buffer.get_mut(),
            src_offset,
            dst_offset,
            size,
            sync_point_wait_list,
            &mut sync_point,
            ptr::null_mut(),
        )?;
        Ok(sync_point)
    }

    /// Records a command to copy a rectangular region from a buffer object to another buffer object.
    pub unsafe fn copy_buffer_rect<T>(
        &self,
        queue: cl_command_queue,
        src_buffer: &Buffer<T>,
        dst_buffer: &mut Buffer<T>,
        src_origin: *const size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        src_row_pitch: size_t,
        src_slice_pitch: size_t,
        dst_row_pitch: size_t,
        dst_slice_pitch: size_t,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        command_copy_buffer_rect_khr(
            self.buffer,
            queue,
            src_buffer.get(),
            dst_buffer.get_mut(),
            src_origin,
            dst_origin,
            region,
            src_row_pitch,
            src_slice_pitch,
            dst_row_pitch,
            dst_slice_pitch,
            sync_point_wait_list,
            &mut sync_point,
            ptr::null_mut(),
        )?;
        Ok(sync_point)
    }

    /// Records a command to copy a buffer object to an image object.
    pub unsafe fn copy_buffer_to_image<T>(
        &self,
        queue: cl_command_queue,
        src_buffer: &Buffer<T>,
        dst_image: &mut Image,
        src_offset: size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        command_copy_buffer_to_image_khr(
            self.buffer,
            queue,
            src_buffer.get(),
            dst_image.get_mut(),
            src_offset,
            dst_origin,
            region,
            sync_point_wait_list,
            &mut sync_point,
            ptr::null_mut(),
        )?;
        Ok(sync_point)
    }

    /// Records a command to copy image objects.
    pub unsafe fn copy_image<T>(
        &self,
        queue: cl_command_queue,
        src_image: Image,
        dst_image: &mut Image,
        src_origin: *const size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        command_copy_image_khr(
            self.buffer,
            queue,
            src_image.get(),
            dst_image.get_mut(),
            src_origin,
            dst_origin,
            region,
            sync_point_wait_list,
            &mut sync_point,
            ptr::null_mut(),
        )?;
        Ok(sync_point)
    }

    /// Records a command to copy an image object to a buffer object.
    pub unsafe fn copy_image_to_buffer<T>(
        &self,
        queue: cl_command_queue,
        src_image: &Image,
        dst_buffer: &mut Buffer<T>,
        src_origin: *const size_t,
        region: *const size_t,
        dst_offset: size_t,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        command_copy_image_to_buffer_khr(
            self.buffer,
            queue,
            src_image.get(),
            dst_buffer.get_mut(),
            src_origin,
            region,
            dst_offset,
            sync_point_wait_list,
            &mut sync_point,
            ptr::null_mut(),
        )?;
        Ok(sync_point)
    }

    /// Records a command to fill a buffer object with a pattern of a given pattern size.
    pub unsafe fn fill_buffer<T>(
        &self,
        queue: cl_command_queue,
        buffer: &mut Buffer<T>,
        pattern: &[T],
        offset: size_t,
        size: size_t,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        command_fill_buffer_khr(
            self.buffer,
            queue,
            buffer.get_mut(),
            pattern.as_ptr() as cl_mem,
            pattern.len() * mem::size_of::<T>(),
            offset,
            size,
            sync_point_wait_list,
            &mut sync_point,
            ptr::null_mut(),
        )?;
        Ok(sync_point)
    }

    /// Records a command to fill an image object with a specified color.
    pub unsafe fn fill_image<T>(
        &self,
        queue: cl_command_queue,
        image: &mut Image,
        fill_color: *const c_void,
        origin: *const size_t,
        region: *const size_t,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        command_fill_image_khr(
            self.buffer,
            queue,
            image.get_mut(),
            fill_color,
            origin,
            region,
            sync_point_wait_list,
            &mut sync_point,
            ptr::null_mut(),
        )?;
        Ok(sync_point)
    }

    /// Records a command to execute a kernel on a device.
    pub unsafe fn nd_range_kernel(
        &self,
        queue: cl_command_queue,
        properties: *const cl_ndrange_kernel_command_properties_khr,
        kernel: cl_kernel,
        work_dim: cl_uint,
        global_work_offsets: *const size_t,
        global_work_sizes: *const size_t,
        local_work_sizes: *const size_t,
        sync_point_wait_list: &[cl_sync_point_khr],
    ) -> Result<cl_sync_point_khr> {
        let mut sync_point = 0;
        command_nd_range_kernel_khr(
            self.buffer,
            queue,
            properties,
            kernel,
            work_dim,
            global_work_offsets,
            global_work_sizes,
            local_work_sizes,
            sync_point_wait_list,
            &mut sync_point,
            ptr::null_mut(),
        )?;
        Ok(sync_point)
    }

    pub fn num_queues(&self) -> Result<cl_uint> {
        Ok(get_command_buffer_info_khr(self.buffer, CL_COMMAND_BUFFER_NUM_QUEUES_KHR)?.into())
    }

    pub fn queues(&self) -> Result<Vec<isize>> {
        // cl_command_queue
        Ok(get_command_buffer_info_khr(self.buffer, CL_COMMAND_BUFFER_QUEUES_KHR)?.into())
    }

    pub fn reference_count(&self) -> Result<cl_uint> {
        Ok(get_command_buffer_info_khr(self.buffer, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR)?.into())
    }

    pub fn buffer_state(&self) -> Result<cl_uint> {
        Ok(get_command_buffer_info_khr(self.buffer, CL_COMMAND_BUFFER_STATE_KHR)?.into())
    }

    pub fn properties_array(&self) -> Result<Vec<cl_command_buffer_properties_khr>> {
        Ok(
            get_command_buffer_info_khr(self.buffer, CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR)?
                .into(),
        )
    }

    pub fn get_data(&self, param_name: cl_command_buffer_info_khr) -> Result<Vec<u8>> {
        Ok(get_command_buffer_data_khr(self.buffer, param_name)?)
    }
}
