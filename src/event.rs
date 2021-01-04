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

use cl3::types::{cl_event, cl_int, cl_uint, cl_ulong};
use libc::intptr_t;

/// An OpenCL event object.  
/// Has methods to return information from calls to clGetEventInfo and
/// clGetEventProfilingInfo with the appropriate parameters.  
/// Implements the Drop trait to call release_event when the object is dropped.
pub struct Event {
    event: cl_event,
}

impl Drop for Event {
    fn drop(&mut self) {
        release_event(self.event).unwrap();
    }
}

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

    pub fn command_execution_status(&self) -> Result<cl_int, cl_int> {
        Ok(get_event_info(self.event, EventInfo::CL_EVENT_COMMAND_EXECUTION_STATUS)?.to_int())
    }

    pub fn command_type(&self) -> Result<cl_uint, cl_int> {
        Ok(get_event_info(self.event, EventInfo::CL_EVENT_COMMAND_TYPE)?.to_uint())
    }

    pub fn reference_count(&self) -> Result<cl_uint, cl_int> {
        Ok(get_event_info(self.event, EventInfo::CL_EVENT_REFERENCE_COUNT)?.to_uint())
    }

    pub fn command_queue(&self) -> Result<intptr_t, cl_int> {
        Ok(get_event_info(self.event, EventInfo::CL_EVENT_COMMAND_QUEUE)?.to_ptr())
    }

    pub fn context(&self) -> Result<intptr_t, cl_int> {
        Ok(get_event_info(self.event, EventInfo::CL_EVENT_CONTEXT)?.to_ptr())
    }

    pub fn profiling_command_queued(&self) -> Result<cl_ulong, cl_int> {
        Ok(
            get_event_profiling_info(self.event, ProfilingInfo::CL_PROFILING_COMMAND_QUEUED)?
                .to_ulong(),
        )
    }

    pub fn profiling_command_submit(&self) -> Result<cl_ulong, cl_int> {
        Ok(
            get_event_profiling_info(self.event, ProfilingInfo::CL_PROFILING_COMMAND_SUBMIT)?
                .to_ulong(),
        )
    }

    pub fn profiling_command_start(&self) -> Result<cl_ulong, cl_int> {
        Ok(
            get_event_profiling_info(self.event, ProfilingInfo::CL_PROFILING_COMMAND_START)?
                .to_ulong(),
        )
    }

    pub fn profiling_command_end(&self) -> Result<cl_ulong, cl_int> {
        Ok(
            get_event_profiling_info(self.event, ProfilingInfo::CL_PROFILING_COMMAND_END)?
                .to_ulong(),
        )
    }

    pub fn profiling_command_complete(&self) -> Result<cl_ulong, cl_int> {
        Ok(
            get_event_profiling_info(self.event, ProfilingInfo::CL_PROFILING_COMMAND_COMPLETE)?
                .to_ulong(),
        )
    }
}
