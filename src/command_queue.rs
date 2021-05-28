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

pub use cl3::command_queue::*;

use super::context::Context;

use super::device::Device;
use super::event::Event;
use super::memory::*;
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
use cl3::ffi::cl_ext::cl_queue_properties_khr;
use cl3::gl;
use cl3::types::{
    cl_bool, cl_command_queue, cl_command_queue_properties, cl_context, cl_device_id, cl_event,
    cl_kernel, cl_map_flags, cl_mem, cl_mem_migration_flags, cl_queue_properties, cl_uint,
    cl_ulong,
};
use libc::{c_void, size_t};
use std::mem;
use std::ptr;

/// An OpenCL command-queue.  
/// Operations on OpenCL memory and kernel objects are performed using a
/// command-queue.
#[derive(Debug)]
pub struct CommandQueue {
    queue: cl_command_queue,
    max_work_item_dimensions: cl_uint,
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        release_command_queue(self.queue).expect("Error: clReleaseCommandQueue")
    }
}

unsafe impl Send for CommandQueue {}

impl CommandQueue {
    fn new(queue: cl_command_queue, max_work_item_dimensions: cl_uint) -> CommandQueue {
        CommandQueue {
            queue,
            max_work_item_dimensions,
        }
    }

    /// Get the underlying OpenCL cl_command_queue.
    pub fn get(&self) -> cl_command_queue {
        self.queue
    }

    /// Get the max_work_item_dimensions for the device that the underlying OpenCL
    /// device.
    pub fn max_work_item_dimensions(&self) -> cl_uint {
        self.max_work_item_dimensions
    }

    /// Create an OpenCL command-queue on a specific device.  
    /// Queries the device the max_work_item_dimensions.  
    /// Deprecated in CL_VERSION_2_0 by create_command_queue_with_properties.
    ///
    /// * `context` - a valid OpenCL context.
    /// * `device_id` - a device or sub-device associated with context.
    /// * `properties` - a list of properties for the command-queue, see
    /// [cl_command_queue_properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#legacy-queue-properties-table).
    ///
    /// returns a Result containing the new CommandQueue
    /// or the error code from the OpenCL C API function.
    pub fn create(
        context: &Context,
        device_id: cl_device_id,
        properties: cl_command_queue_properties,
    ) -> Result<CommandQueue> {
        let queue = create_command_queue(context.get(), device_id, properties)?;
        let device = Device::new(device_id);
        let max_work_item_dimensions = device.max_work_item_dimensions()?;
        Ok(CommandQueue::new(queue, max_work_item_dimensions))
    }

    /// Create an OpenCL command-queue on a specific device.  
    /// Queries the device the max_work_item_dimensions.  
    /// CL_VERSION_2_0 onwards.
    ///
    /// * `context` - a valid OpenCL context.
    /// * `device_id` - a device or sub-device associated with context.
    /// * `properties` - a null terminated list of properties for the command-queue, see
    /// [cl_queue_properties](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#queue-properties-table).
    ///
    /// returns a Result containing the new CommandQueue
    /// or the error code from the OpenCL C API function.
    pub fn create_with_properties(
        context: &Context,
        device_id: cl_device_id,
        properties: cl_command_queue_properties,
        queue_size: cl_uint,
    ) -> Result<CommandQueue> {
        let queue = if (0 < properties) || (0 < queue_size) {
            let mut props: [cl_queue_properties; 5] = [0; 5];

            let mut index = 0;
            if 0 < properties {
                props[index] = CommandQueueInfo::CL_QUEUE_PROPERTIES as cl_queue_properties;
                props[index + 1] = properties as cl_queue_properties;
                index += 2;
            }

            if 0 < queue_size {
                props[index] = CommandQueueInfo::CL_QUEUE_SIZE as cl_queue_properties;
                props[index + 1] = queue_size as cl_queue_properties;
            }
            create_command_queue_with_properties(context.get(), device_id, props.as_ptr())?
        } else {
            create_command_queue_with_properties(context.get(), device_id, ptr::null())?
        };

        let device = Device::new(device_id);
        let max_work_item_dimensions = device.max_work_item_dimensions()?;
        Ok(CommandQueue::new(queue, max_work_item_dimensions))
    }

    #[cfg(feature = "cl_khr_create_command_queue")]
    pub fn create_with_properties_khr(
        context: &Context,
        device_id: cl_device_id,
        properties: &[cl_queue_properties_khr],
    ) -> Result<CommandQueue> {
        let queue = ext::create_command_queue_with_properties_khr(
            context.get(),
            device_id,
            properties.as_ptr(),
        )?;

        let device = Device::new(device_id);
        let max_work_item_dimensions = device.max_work_item_dimensions()?;
        Ok(CommandQueue::new(queue, max_work_item_dimensions))
    }

    /// Flush commands to a device.  
    /// returns an empty Result or the error code from the OpenCL C API function.
    pub fn flush(&self) -> Result<()> {
        Ok(flush(self.queue)?)
    }

    /// Wait for completion of commands on a device.  
    /// returns an empty Result or the error code from the OpenCL C API function.
    pub fn finish(&self) -> Result<()> {
        Ok(finish(self.queue)?)
    }

    pub fn enqueue_read_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        blocking_read: cl_bool,
        offset: size_t,
        data: &mut [T],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_read_buffer(
            self.queue,
            buffer.get(),
            blocking_read,
            offset,
            (data.len() * mem::size_of::<T>()) as size_t,
            data.as_mut_ptr() as cl_mem,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_read_buffer_rect<T>(
        &self,
        buffer: &Buffer<T>,
        blocking_read: cl_bool,
        buffer_origin: *const size_t,
        host_origin: *const size_t,
        region: *const size_t,
        buffer_row_pitch: size_t,
        buffer_slice_pitch: size_t,
        host_row_pitch: size_t,
        host_slice_pitch: size_t,
        ptr: *mut c_void,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_read_buffer_rect(
            self.queue,
            buffer.get(),
            blocking_read,
            buffer_origin,
            host_origin,
            region,
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            ptr,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_write_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        blocking_write: cl_bool,
        offset: size_t,
        data: &[T],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_write_buffer(
            self.queue,
            buffer.get(),
            blocking_write,
            offset,
            (data.len() * mem::size_of::<T>()) as size_t,
            data.as_ptr() as cl_mem,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_write_buffer_rect<T>(
        &self,
        buffer: &Buffer<T>,
        blocking_write: cl_bool,
        buffer_origin: *const size_t,
        host_origin: *const size_t,
        region: *const size_t,
        buffer_row_pitch: size_t,
        buffer_slice_pitch: size_t,
        host_row_pitch: size_t,
        host_slice_pitch: size_t,
        ptr: *mut c_void,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_write_buffer_rect(
            self.queue,
            buffer.get(),
            blocking_write,
            buffer_origin,
            host_origin,
            region,
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            ptr,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_fill_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        pattern: &[T],
        offset: size_t,
        size: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_fill_buffer(
            self.queue,
            buffer.get(),
            pattern.as_ptr() as cl_mem,
            pattern.len() * mem::size_of::<T>(),
            offset,
            size,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_copy_buffer<T>(
        &self,
        src_buffer: &Buffer<T>,
        dst_buffer: &Buffer<T>,
        src_offset: size_t,
        dst_offset: size_t,
        size: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_copy_buffer(
            self.queue,
            src_buffer.get(),
            dst_buffer.get(),
            src_offset,
            dst_offset,
            size,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }
    pub fn enqueue_copy_buffer_rect<T>(
        &self,
        src_buffer: &Buffer<T>,
        dst_buffer: &Buffer<T>,
        src_origin: *const size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        src_row_pitch: size_t,
        src_slice_pitch: size_t,
        dst_row_pitch: size_t,
        dst_slice_pitch: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_copy_buffer_rect(
            self.queue,
            src_buffer.get(),
            dst_buffer.get(),
            src_origin,
            dst_origin,
            region,
            src_row_pitch,
            src_slice_pitch,
            dst_row_pitch,
            dst_slice_pitch,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_read_image(
        &self,
        image: &Image,
        blocking_read: cl_bool,
        origin: *const size_t,
        region: *const size_t,
        row_pitch: size_t,
        slice_pitch: size_t,
        ptr: *mut c_void,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_read_image(
            self.queue,
            image.get(),
            blocking_read,
            origin,
            region,
            row_pitch,
            slice_pitch,
            ptr,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_write_image(
        &self,
        image: &Image,
        blocking_write: cl_bool,
        origin: *const size_t,
        region: *const size_t,
        row_pitch: size_t,
        slice_pitch: size_t,
        ptr: *mut c_void,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_write_image(
            self.queue,
            image.get(),
            blocking_write,
            origin,
            region,
            row_pitch,
            slice_pitch,
            ptr,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_fill_image(
        &self,
        image: &Image,
        fill_color: *const c_void,
        origin: *const size_t,
        region: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_fill_image(
            self.queue,
            image.get(),
            fill_color,
            origin,
            region,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_copy_image(
        &self,
        src_image: &Image,
        dst_image: &Image,
        src_origin: *const size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_copy_image(
            self.queue,
            src_image.get(),
            dst_image.get(),
            src_origin,
            dst_origin,
            region,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_copy_image_to_buffer<T>(
        &self,
        src_image: &Image,
        dst_buffer: &Buffer<T>,
        src_origin: *const size_t,
        region: *const size_t,
        dst_offset: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_copy_image_to_buffer(
            self.queue,
            src_image.get(),
            dst_buffer.get(),
            src_origin,
            region,
            dst_offset,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_copy_buffer_to_image<T>(
        &self,
        src_buffer: &Buffer<T>,
        dst_image: &Image,
        src_offset: size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_copy_buffer_to_image(
            self.queue,
            src_buffer.get(),
            dst_image.get(),
            src_offset,
            dst_origin,
            region,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_map_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        blocking_map: cl_bool,
        map_flags: cl_map_flags,
        offset: size_t,
        size: size_t,
        buffer_ptr: &mut cl_mem,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_map_buffer(
            self.queue,
            buffer.get(),
            blocking_map,
            map_flags,
            offset,
            size,
            buffer_ptr,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_map_image(
        &self,
        image: &Image,
        blocking_map: cl_bool,
        map_flags: cl_map_flags,
        origin: *const size_t,
        region: *const size_t,
        image_row_pitch: *mut size_t,
        image_slice_pitch: *mut size_t,
        image_ptr: &mut cl_mem,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_map_image(
            self.queue,
            image.get(),
            blocking_map,
            map_flags,
            origin,
            region,
            image_row_pitch,
            image_slice_pitch,
            image_ptr,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_unmap_mem_object(
        &self,
        memobj: cl_mem,
        mapped_ptr: *mut c_void,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_unmap_mem_object(
            self.queue,
            memobj,
            mapped_ptr,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_migrate_mem_object(
        &self,
        num_mem_objects: cl_uint,
        mem_objects: *const cl_mem,
        flags: cl_mem_migration_flags,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_migrate_mem_object(
            self.queue,
            num_mem_objects,
            mem_objects,
            flags,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_ext_migrate_memobject")]
    pub fn enqueue_migrate_mem_object_ext(
        &self,
        num_mem_objects: cl_uint,
        mem_objects: *const cl_mem,
        flags: cl_mem_migration_flags_ext,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_migrate_mem_object_ext(
            self.queue,
            num_mem_objects,
            mem_objects,
            flags,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_nd_range_kernel(
        &self,
        kernel: cl_kernel,
        work_dim: cl_uint,
        global_work_offsets: *const size_t,
        global_work_sizes: *const size_t,
        local_work_sizes: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_nd_range_kernel(
            self.queue,
            kernel,
            work_dim,
            global_work_offsets,
            global_work_sizes,
            local_work_sizes,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_task(&self, kernel: cl_kernel, event_wait_list: &[cl_event]) -> Result<Event> {
        let event = enqueue_task(
            self.queue,
            kernel,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_native_kernel(
        &self,
        user_func: Option<extern "C" fn(*mut c_void)>,
        args: &[*mut c_void],
        mem_list: &[cl_mem],
        args_mem_loc: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_native_kernel(
            self.queue,
            user_func,
            args.as_ptr() as *mut c_void,
            args.len() as size_t,
            mem_list.len() as cl_uint,
            if 0 < mem_list.len() {
                mem_list.as_ptr()
            } else {
                ptr::null()
            },
            args_mem_loc.as_ptr(),
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_marker_with_wait_list(&self, event_wait_list: &[cl_event]) -> Result<Event> {
        let event = enqueue_marker_with_wait_list(
            self.queue,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_barrier_with_wait_list(&self, event_wait_list: &[cl_event]) -> Result<Event> {
        let event = enqueue_barrier_with_wait_list(
            self.queue,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_svm_free(
        &self,
        svm_pointers: &[*const c_void],
        pfn_free_func: Option<
            extern "C" fn(
                queue: cl_command_queue,
                num_svm_pointers: cl_uint,
                svm_pointers: *const *const c_void,
                user_data: *mut c_void,
            ),
        >,
        user_data: *mut c_void,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_svm_free(
            self.queue,
            svm_pointers.len() as cl_uint,
            svm_pointers.as_ptr(),
            pfn_free_func,
            user_data,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_svm_mem_cpy(
        &self,
        blocking_copy: cl_bool,
        dst_ptr: *mut c_void,
        src_ptr: *const c_void,
        size: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_svm_mem_cpy(
            self.queue,
            blocking_copy,
            dst_ptr,
            src_ptr,
            size,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_svm_mem_fill<T>(
        &self,
        svm_ptr: *mut c_void,
        pattern: &[T],
        size: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_svm_mem_fill(
            self.queue,
            svm_ptr,
            pattern.as_ptr() as *const c_void,
            pattern.len() * mem::size_of::<T>(),
            size,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_svm_map<T>(
        &self,
        blocking_map: cl_bool,
        flags: cl_map_flags,
        svm: &mut [T],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_svm_map(
            self.queue,
            blocking_map,
            flags,
            svm.as_mut_ptr() as *mut c_void,
            svm.len() * mem::size_of::<T>(),
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_svm_unmap<T>(&self, svm: &[T], event_wait_list: &[cl_event]) -> Result<Event> {
        let event = enqueue_svm_unmap(
            self.queue,
            svm.as_ptr() as *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "CL_VERSION_2_1")]
    pub fn enqueue_svm_migrate_mem(
        &self,
        svm_pointers: &[*const c_void],
        sizes: *const size_t,
        flags: cl_mem_migration_flags,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = enqueue_svm_migrate_mem(
            self.queue,
            svm_pointers.len() as cl_uint,
            svm_pointers.as_ptr(),
            sizes,
            flags,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_acquire_gl_objects(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = gl::enqueue_acquire_gl_objects(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn enqueue_release_gl_objects(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = gl::enqueue_release_gl_objects(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_khr_egl_image")]
    #[inline]
    pub fn enqueue_acquire_egl_objects(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = egl::enqueue_acquire_egl_objects(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_khr_egl_image")]
    #[inline]
    pub fn enqueue_release_egl_objects(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = egl::enqueue_release_egl_objects(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_img_use_gralloc_ptr")]
    #[inline]
    pub fn enqueue_acquire_gralloc_objects_img(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = ext::enqueue_acquire_gralloc_objects_img(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_img_use_gralloc_ptr")]
    #[inline]
    pub fn enqueue_release_gralloc_objects_img(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = ext::enqueue_release_gralloc_objects_img(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_khr_dx9_media_sharing")]
    #[inline]
    pub fn enqueue_acquire_dx9_media_surfaces_khr(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = dx9_media_sharing::enqueue_acquire_dx9_media_surfaces_khr(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_khr_dx9_media_sharing")]
    #[inline]
    pub fn enqueue_release_dx9_media_surfaces_khr(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = dx9_media_sharing::enqueue_release_dx9_media_surfaces_khr(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_intel_dx9_media_sharing")]
    #[inline]
    pub fn enqueue_acquire_dx9_objects_intel(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = dx9_media_sharing::enqueue_acquire_dx9_objects_intel(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_intel_dx9_media_sharing")]
    #[inline]
    pub fn enqueue_release_dx9_objects_intel(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = dx9_media_sharing::enqueue_release_dx9_objects_intel(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_khr_d3d10_sharing")]
    #[inline]
    pub fn enqueue_acquire_dx10_objects_khr(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = d3d10::enqueue_acquire_dx10_objects_khr(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_khr_d3d10_sharing")]
    #[inline]
    pub fn enqueue_release_dx10_objects_khr(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = d3d10::enqueue_release_dx10_objects_khr(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_khr_d3d11_sharing")]
    #[inline]
    pub fn enqueue_acquire_dx11_objects_khr(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = d3d11::enqueue_acquire_dx11_objects_khr(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_khr_d3d11_sharing")]
    #[inline]
    pub fn enqueue_release_dx11_objects_khr(
        &self,
        mem_objects: &[*const c_void],
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = d3d11::enqueue_release_dx11_objects_khr(
            self.queue,
            mem_objects.len() as cl_uint,
            mem_objects.as_ptr() as *const *mut c_void,
            event_wait_list.len() as cl_uint,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    #[cfg(feature = "cl_img_generate_mipmap")]
    #[inline]
    pub fn enqueue_generate_mipmap_img(
        &self,
        src_image: cl_mem,
        dst_image: cl_mem,
        mipmap_filter_mode: cl_mipmap_filter_mode_img,
        array_region: *const size_t,
        mip_region: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event> {
        let event = ext::enqueue_generate_mipmap_img(
            self.queue,
            src_image,
            dst_image,
            mipmap_filter_mode,
            array_region,
            mip_region,
            if 0 < event_wait_list.len() {
                event_wait_list.as_ptr()
            } else {
                ptr::null()
            },
        )?;
        Ok(Event::new(event))
    }

    pub fn context(&self) -> Result<cl_context> {
        Ok(
            get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_CONTEXT)?.to_ptr()
                as cl_context,
        )
    }

    pub fn device(&self) -> Result<cl_device_id> {
        Ok(
            get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_DEVICE)?.to_ptr()
                as cl_device_id,
        )
    }

    pub fn reference_count(&self) -> Result<cl_uint> {
        Ok(
            get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_REFERENCE_COUNT)?
                .to_uint(),
        )
    }

    pub fn properties(&self) -> Result<cl_ulong> {
        Ok(get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_PROPERTIES)?.to_ulong())
    }

    pub fn size(&self) -> Result<cl_uint> {
        Ok(get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_SIZE)?.to_uint())
    }

    // CL_VERSION_2_1
    pub fn device_default(&self) -> Result<cl_device_id> {
        Ok(
            get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_DEVICE_DEFAULT)?.to_ptr()
                as cl_device_id,
        )
    }

    // CL_VERSION_3_0
    pub fn properties_array(&self) -> Result<Vec<cl_ulong>> {
        Ok(
            get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_PROPERTIES_ARRAY)?
                .to_vec_ulong(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::device::Device;
    use crate::platform::get_platforms;
    use cl3::device::CL_DEVICE_TYPE_GPU;
    use libc::intptr_t;

    #[test]
    fn test_command_queue() {
        let platforms = get_platforms().unwrap();
        assert!(0 < platforms.len());

        // Get the first platform
        let platform = &platforms[0];

        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
        assert!(0 < devices.len());

        // Get the first device
        let device = Device::new(devices[0]);
        let context = Context::from_device(&device).unwrap();

        // Create a command_queue on the Context's device
        let queue = CommandQueue::create(
            &context,
            context.default_device(),
            CL_QUEUE_PROFILING_ENABLE,
        )
        .expect("CommandQueue::create failed");

        let value = queue.context().unwrap();
        assert!(context.get() == value);

        let value = queue.device().unwrap();
        assert!(device.id() == value);

        let value = queue.reference_count().unwrap();
        println!("queue.reference_count(): {}", value);
        assert_eq!(1, value);

        let value = queue.properties().unwrap();
        println!("queue.properties(): {:X}", value);
        // assert_eq!(2, value);

        // CL_VERSION_2_0 value
        match queue.size() {
            Ok(value) => println!("queue.size(): {:?}", value),
            Err(e) => println!("OpenCL error, queue.size(): {}", e),
        }

        // CL_VERSION_2_1 value
        match queue.device_default() {
            Ok(value) => println!("queue.device_default(): {:X}", value as intptr_t),
            Err(e) => println!("OpenCL error, queue.device_default(): {}", e),
        }

        // CL_VERSION_3_0 value
        match queue.properties_array() {
            Ok(value) => println!("queue.properties_array(): {:?}", value),
            Err(e) => println!("OpenCL error, queue.properties_array(): {}", e),
        }
    }
}
