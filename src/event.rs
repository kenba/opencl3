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

pub use cl3::event::*;

use super::Result;
use cl3::types::{
    cl_command_queue, cl_context, cl_event, cl_event_info, cl_int, cl_profiling_info, cl_uint,
    cl_ulong,
};
use libc::c_void;

/// An OpenCL event object.  
/// Has methods to return information from calls to clGetEventInfo and
/// clGetEventProfilingInfo with the appropriate parameters.  
/// Implements the Drop trait to call release_event when the object is dropped.
#[derive(Debug)]
pub struct Event {
    event: cl_event,
}

impl From<cl_event> for Event {
    fn from(event: cl_event) -> Self {
        Event { event }
    }
}

impl From<Event> for cl_event {
    fn from(value: Event) -> Self {
        value.event as cl_event
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { release_event(self.event).expect("Error: clReleaseEvent") };
    }
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    /// Create an Event from an OpenCL cl_event.
    ///
    /// * `event` - a valid OpenCL cl_event.
    ///
    /// returns the new Event
    pub fn new(event: cl_event) -> Self {
        Self { event }
    }

    /// Get the underlying OpenCL cl_event.
    pub fn get(&self) -> cl_event {
        self.event
    }

    /// Wait for the event to complete.
    pub fn wait(&self) -> Result<()> {
        let events = [self.get()];
        Ok(wait_for_events(&events)?)
    }

    pub fn command_execution_status(&self) -> Result<CommandExecutionStatus> {
        Ok(CommandExecutionStatus(
            get_event_info(self.event, CL_EVENT_COMMAND_EXECUTION_STATUS)?.into(),
        ))
    }

    pub fn command_type(&self) -> Result<EventCommandType> {
        Ok(EventCommandType(
            get_event_info(self.event, CL_EVENT_COMMAND_TYPE)?.into(),
        ))
    }

    pub fn reference_count(&self) -> Result<cl_uint> {
        Ok(get_event_info(self.event, CL_EVENT_REFERENCE_COUNT)?.into())
    }

    pub fn command_queue(&self) -> Result<cl_command_queue> {
        Ok(isize::from(get_event_info(self.event, CL_EVENT_COMMAND_QUEUE)?) as cl_command_queue)
    }

    pub fn context(&self) -> Result<cl_context> {
        Ok(isize::from(get_event_info(self.event, CL_EVENT_CONTEXT)?) as cl_context)
    }

    /// Get data about an OpenCL event.
    /// Calls clGetEventInfo to get the desired data about the event.
    pub fn get_data(&self, param_name: cl_event_info) -> Result<Vec<u8>> {
        Ok(get_event_data(self.event, param_name)?)
    }

    pub fn set_callback(
        &self,
        command_exec_callback_type: cl_int,
        pfn_notify: extern "C" fn(cl_event, cl_int, *mut c_void),
        user_data: *mut c_void,
    ) -> Result<()> {
        Ok(set_event_callback(
            self.event,
            command_exec_callback_type,
            pfn_notify,
            user_data,
        )?)
    }

    pub fn profiling_command_queued(&self) -> Result<cl_ulong> {
        Ok(get_event_profiling_info(self.event, CL_PROFILING_COMMAND_QUEUED)?.into())
    }

    pub fn profiling_command_submit(&self) -> Result<cl_ulong> {
        Ok(get_event_profiling_info(self.event, CL_PROFILING_COMMAND_SUBMIT)?.into())
    }

    pub fn profiling_command_start(&self) -> Result<cl_ulong> {
        Ok(get_event_profiling_info(self.event, CL_PROFILING_COMMAND_START)?.into())
    }

    pub fn profiling_command_end(&self) -> Result<cl_ulong> {
        Ok(get_event_profiling_info(self.event, CL_PROFILING_COMMAND_END)?.into())
    }

    /// CL_VERSION_2_0
    pub fn profiling_command_complete(&self) -> Result<cl_ulong> {
        Ok(get_event_profiling_info(self.event, CL_PROFILING_COMMAND_COMPLETE)?.into())
    }

    /// Get profiling data about an OpenCL event.
    /// Calls clGetEventProfilingInfo to get the desired profiling data about the event.
    pub fn profiling_data(&self, param_name: cl_profiling_info) -> Result<Vec<u8>> {
        Ok(get_event_profiling_data(self.event, param_name)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
    use crate::context::Context;
    use crate::device::{Device, CL_DEVICE_TYPE_GPU};
    use crate::memory::{Buffer, CL_MEM_READ_ONLY};
    use crate::platform::get_platforms;
    use crate::types::{cl_float, CL_NON_BLOCKING};
    use std::ptr;

    extern "C" fn event_callback_function(
        _event: cl_event,
        event_command_status: cl_int,
        _user_data: *mut c_void,
    ) {
        println!(
            "OpenCL event callback command status: {}",
            event_command_status
        );
    }

    #[test]
    fn test_event() {
        let platforms = get_platforms().unwrap();
        assert!(0 < platforms.len());

        // Get the first platform
        let platform = &platforms[0];

        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
        assert!(0 < devices.len());

        // Get the first device
        let device = Device::new(devices[0]);
        let context = Context::from_device(&device).unwrap();

        // Create a command_queue on the Context's default device
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("CommandQueue::create_default failed");

        const ARRAY_SIZE: usize = 1024;
        let ones: [cl_float; ARRAY_SIZE] = [1.0; ARRAY_SIZE];

        let mut buffer = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())
                .unwrap()
        };

        let events: Vec<cl_event> = Vec::default();

        // Non-blocking write, wait for event
        let event = unsafe {
            queue
                .enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, &ones, &events)
                .unwrap()
        };

        // Set a callback_function on the event (i.e. write) being completed.
        event
            .set_callback(CL_COMPLETE, event_callback_function, ptr::null_mut())
            .unwrap();

        let value = event.command_execution_status().unwrap();
        println!("event.command_execution_status(): {}", value);
        // assert_eq!(CL_QUEUED, value.0);

        let value = event.command_type().unwrap();
        println!("event.command_type(): {}", value);
        assert_eq!(CL_COMMAND_WRITE_BUFFER, value.0);

        let value = event.reference_count().unwrap();
        println!("event.reference_count(): {}", value);
        // assert_eq!(1, value);

        let value = event.command_queue().unwrap();
        assert!(queue.get() == value);

        let value = event.context().unwrap();
        assert!(context.get() == value);

        event.wait().unwrap();

        let value = event.command_execution_status().unwrap();
        println!("event.command_execution_status(): {}", value);
        assert_eq!(CL_COMPLETE, value.0);

        let value = event.profiling_command_queued().unwrap();
        println!("event.profiling_command_queued(): {}", value);
        assert!(0 < value);

        let value = event.profiling_command_submit().unwrap();
        println!("event.profiling_command_submit(): {}", value);
        assert!(0 < value);

        let value = event.profiling_command_start().unwrap();
        println!("event.profiling_command_start(): {}", value);
        assert!(0 < value);

        let value = event.profiling_command_end().unwrap();
        println!("event.profiling_command_end(): {}", value);
        assert!(0 < value);

        // CL_VERSION_2_0
        match event.profiling_command_complete() {
            Ok(value) => println!("event.profiling_command_complete(): {}", value),
            Err(e) => println!("OpenCL error, event.profiling_command_complete(): {}", e),
        }
    }
}
