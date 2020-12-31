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

pub use cl3::command_queue::*;

use super::event::Event;

use cl3::types::{
    cl_bool, cl_command_queue, cl_command_queue_properties, cl_context, cl_device_id, cl_event,
    cl_int, cl_kernel, cl_map_flags, cl_mem, cl_queue_properties, cl_uint, cl_ulong,
    cl_mem_migration_flags,
};
use libc::{c_void, intptr_t, size_t};
use std::mem;
use std::ptr;

/// An OpenCL command-queue.  
/// Operations on OpenCL memory and kernel objects are performed using a
/// command-queue.
pub struct CommandQueue {
    queue: cl_command_queue,
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        release_command_queue(self.queue).unwrap();
        println!("CommandQueue::drop");
    }
}

impl CommandQueue {
    fn new(queue: cl_command_queue) -> CommandQueue {
        CommandQueue { queue }
    }

    pub fn get(&self) -> cl_command_queue {
        self.queue
    }

    pub fn create(
        context: cl_context,
        device: cl_device_id,
        properties: cl_command_queue_properties,
    ) -> Result<CommandQueue, cl_int> {
        let queue = create_command_queue(context, device, properties)?;
        Ok(CommandQueue::new(queue))
    }

    pub fn create_with_properties(
        context: cl_context,
        device: cl_device_id,
        properties: cl_command_queue_properties,
        queue_size: cl_uint,
    ) -> Result<CommandQueue, cl_int> {
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
            create_command_queue_with_properties(context, device, props.as_ptr())?
        } else {
            create_command_queue_with_properties(context, device, ptr::null())?
        };

        Ok(CommandQueue::new(queue))
    }

    pub fn flush(&self) -> Result<(), cl_int> {
        flush(self.queue)
    }

    pub fn finish(&self) -> Result<(), cl_int> {
        finish(self.queue)
    }

    pub fn enqueue_read_buffer<T>(
        &self,
        buffer: cl_mem,
        blocking_read: cl_bool,
        offset: size_t,
        data: &mut [T],
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_read_buffer(
            self.queue,
            buffer,
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

    pub fn enqueue_read_buffer_rect(
        &self,
        buffer: cl_mem,
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
    ) -> Result<Event, cl_int> {
        let event = enqueue_read_buffer_rect(
            self.queue,
            buffer,
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
        buffer: cl_mem,
        blocking_write: cl_bool,
        offset: size_t,
        data: &[T],
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_write_buffer(
            self.queue,
            buffer,
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

    pub fn enqueue_write_buffer_rect(
        &self,
        buffer: cl_mem,
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
    ) -> Result<Event, cl_int> {
        let event = enqueue_write_buffer_rect(
            self.queue,
            buffer,
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
        buffer: cl_mem,
        pattern: &[T],
        offset: size_t,
        size: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_fill_buffer(
            self.queue,
            buffer,
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

    pub fn enqueue_copy_buffer(
        &self,
        src_buffer: cl_mem,
        dst_buffer: cl_mem,
        src_offset: size_t,
        dst_offset: size_t,
        size: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_copy_buffer(
            self.queue,
            src_buffer,
            dst_buffer,
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
    
    pub fn enqueue_copy_buffer_rect(
        &self,
        src_buffer: cl_mem,
        dst_buffer: cl_mem,
        src_origin: *const size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        src_row_pitch: size_t,
        src_slice_pitch: size_t,
        dst_row_pitch: size_t,
        dst_slice_pitch: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_copy_buffer_rect(
            self.queue,
            src_buffer,
            dst_buffer,
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
        image: cl_mem,
        blocking_read: cl_bool,
        origin: *const size_t,
        region: *const size_t,
        row_pitch: size_t,
        slice_pitch: size_t,
        ptr: *mut c_void,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_read_image(
            self.queue,
            image,
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
        image: cl_mem,
        blocking_write: cl_bool,
        origin: *const size_t,
        region: *const size_t,
        row_pitch: size_t,
        slice_pitch: size_t,
        ptr: *mut c_void,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_write_image(
            self.queue,
            image,
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
        image: cl_mem,
        fill_color: *const c_void,
        origin: *const size_t,
        region: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_fill_image(
            self.queue,
            image,
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
        src_image: cl_mem,
        dst_image: cl_mem,
        src_origin: *const size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_copy_image(
            self.queue,
            src_image,
            dst_image,
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

    pub fn enqueue_copy_image_to_buffer(
        &self,
        src_image: cl_mem,
        dst_buffer: cl_mem,
        src_origin: *const size_t,
        region: *const size_t,
        dst_offset: size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_copy_image_to_buffer(
            self.queue,
            src_image,
            dst_buffer,
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

    pub fn enqueue_copy_buffer_to_image(
        &self,
        src_buffer: cl_mem,
        dst_image: cl_mem,
        src_offset: size_t,
        dst_origin: *const size_t,
        region: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_copy_buffer_to_image(
            self.queue,
            src_buffer,
            dst_image,
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

    pub fn enqueue_map_buffer(
        &self,
        buffer: cl_mem,
        blocking_map: cl_bool,
        map_flags: cl_map_flags,
        offset: size_t,
        size: size_t,
        buffer_ptr: &mut cl_mem,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_map_buffer(
            self.queue,
            buffer,
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
        image: cl_mem,
        blocking_map: cl_bool,
        map_flags: cl_map_flags,
        origin: *const size_t,
        region: *const size_t,
        image_row_pitch: *mut size_t,
        image_slice_pitch: *mut size_t,
        image_ptr: &mut cl_mem,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
        let event = enqueue_map_image(
            self.queue,
            image,
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
    ) -> Result<Event, cl_int> {
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
    ) -> Result<Event, cl_int> {
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

    pub fn enqueue_nd_range_kernel(
        &self,
        kernel: cl_kernel,
        work_dim: cl_uint,
        global_work_offsets: *const size_t,
        global_work_sizes: *const size_t,
        local_work_sizes: *const size_t,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
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

    pub fn enqueue_task(
        &self,
        kernel: cl_kernel,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
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
    ) -> Result<Event, cl_int> {
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

    pub fn enqueue_marker_with_wait_list(
        &self,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
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

    pub fn enqueue_barrier_with_wait_list(
        &self,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
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
    ) -> Result<Event, cl_int> {
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
    ) -> Result<Event, cl_int> {
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
    ) -> Result<Event, cl_int> {
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
    ) -> Result<Event, cl_int> {
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

    pub fn enqueue_svm_unmap<T>(
        &self,
        svm: &[T],
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
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

    pub fn enqueue_svm_migrate_mem(
        &self,
        svm_pointers: &[*const c_void],
        sizes: *const size_t,
        flags: cl_mem_migration_flags,
        event_wait_list: &[cl_event],
    ) -> Result<Event, cl_int> {
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

    pub fn context(&self) -> Result<intptr_t, cl_int> {
        Ok(get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_CONTEXT)?.to_ptr())
    }

    pub fn device(&self) -> Result<intptr_t, cl_int> {
        Ok(get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_DEVICE)?.to_ptr())
    }

    pub fn reference_count(&self) -> Result<cl_uint, cl_int> {
        Ok(
            get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_REFERENCE_COUNT)?
                .to_uint(),
        )
    }

    pub fn properties(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_PROPERTIES)?.to_ulong())
    }

    pub fn size(&self) -> Result<cl_uint, cl_int> {
        Ok(get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_SIZE)?.to_uint())
    }

    // CL_VERSION_2_1
    pub fn device_default(&self) -> Result<intptr_t, cl_int> {
        Ok(get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_DEVICE_DEFAULT)?.to_ptr())
    }

    // CL_VERSION_3_0
    pub fn properties_array(&self) -> Result<Vec<cl_ulong>, cl_int> {
        Ok(
            get_command_queue_info(self.queue, CommandQueueInfo::CL_QUEUE_PROPERTIES_ARRAY)?
                .to_vec_ulong(),
        )
    }
}
