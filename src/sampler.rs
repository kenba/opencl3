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

use cl3::sampler;
use cl3::types::{
    cl_addressing_mode, cl_bool, cl_context, cl_filter_mode, cl_int, cl_sampler,
    cl_sampler_properties,
};
use std::ptr;

pub struct Sampler {
    sampler: cl_sampler,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        sampler::release_sampler(self.sampler).unwrap();
        self.sampler = ptr::null_mut();
        println!("Sampler::drop");
    }
}

impl Sampler {
    pub fn new(sampler: cl_sampler) -> Sampler {
        Sampler { sampler }
    }

    pub fn create<T>(
        context: cl_context,
        normalize_coords: cl_bool,
        addressing_mode: cl_addressing_mode,
        filter_mode: cl_filter_mode,
    ) -> Result<Sampler, cl_int> {
        let sampler =
            sampler::create_sampler(context, normalize_coords, addressing_mode, filter_mode)?;
        Ok(Sampler::new(sampler))
    }

    pub fn create_with_properties<T>(
        context: cl_context,
        properties: *const cl_sampler_properties,
    ) -> Result<Sampler, cl_int> {
        let sampler = sampler::create_sampler_with_properties(context, properties)?;
        Ok(Sampler::new(sampler))
    }

    pub fn get(&self) -> cl_sampler {
        self.sampler
    }
}
