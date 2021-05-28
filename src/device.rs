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

pub use cl3::device::*;
pub use cl3::ffi::cl_ext::cl_amd_device_topology;

use super::Result;
use cl3::types::{
    cl_device_fp_config, cl_device_id, cl_device_partition_property, cl_device_svm_capabilities,
    cl_device_type, cl_name_version, cl_platform_id, cl_uint, cl_ulong,
};
use libc::{intptr_t, size_t};

/// A text representation of an OpenCL device type, see:
/// [Device Types](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#device-types-table).
pub fn device_type_text(dev_type: cl_device_type) -> &'static str {
    match dev_type {
        CL_DEVICE_TYPE_DEFAULT => "CL_DEVICE_TYPE_DEFAULT",
        CL_DEVICE_TYPE_CPU => "CL_DEVICE_TYPE_CPU",
        CL_DEVICE_TYPE_GPU => "CL_DEVICE_TYPE_GPU",
        CL_DEVICE_TYPE_ACCELERATOR => "CL_DEVICE_TYPE_ACCELERATOR",
        CL_DEVICE_TYPE_CUSTOM => "CL_DEVICE_TYPE_CUSTOM",
        CL_DEVICE_TYPE_ALL => "CL_DEVICE_TYPE_ALL",

        _ => "COMBINED_DEVICE_TYPE",
    }
}

#[derive(Debug)]
pub struct SubDevice {
    id: cl_device_id,
}

impl Drop for SubDevice {
    fn drop(&mut self) {
        release_device(self.id()).expect("Error: clReleaseDevice");
    }
}

unsafe impl Send for SubDevice {}

impl SubDevice {
    pub fn new(id: cl_device_id) -> SubDevice {
        SubDevice { id }
    }

    /// Accessor for the underlying device id.
    pub fn id(&self) -> cl_device_id {
        self.id
    }
}

/// An OpenCL device id and methods to query it.  
/// The query methods calls clGetDeviceInfo with the relevant param_name, see:
/// [Device Queries](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#device-queries-table).
#[derive(Copy, Clone, Debug)]
pub struct Device {
    id: intptr_t,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub fn new(id: cl_device_id) -> Device {
        Device { id: id as intptr_t }
    }

    /// Accessor for the underlying device id.
    pub fn id(&self) -> cl_device_id {
        self.id as cl_device_id
    }

    /// Create sub-devices by partitioning an OpenCL device.
    ///
    /// * `properties` - the slice of cl_device_partition_property, see
    /// [Subdevice Partition](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#subdevice-partition-table).
    ///
    /// returns a Result containing a vector of available SubDevices
    /// or the error code from the OpenCL C API function.
    pub fn create_sub_devices(
        &self,
        properties: &[cl_device_partition_property],
    ) -> Result<Vec<SubDevice>> {
        let sub_device_ids = create_sub_devices(self.id(), &properties)?;
        Ok(sub_device_ids
            .iter()
            .map(|id| SubDevice::new(*id))
            .collect::<Vec<SubDevice>>())
    }

    #[cfg(feature = "CL_VERSION_2_1")]
    #[inline]
    pub fn get_device_and_host_timer(&self) -> Result<[cl_ulong; 2]> {
        Ok(get_device_and_host_timer(self.id())?)
    }

    #[cfg(feature = "CL_VERSION_2_1")]
    #[inline]
    pub fn get_host_timer(&self) -> Result<cl_ulong> {
        Ok(get_host_timer(self.id())?)
    }

    /// The OpenCL device type, see
    /// [Device Types](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#device-types-table).  
    pub fn dev_type(&self) -> Result<cl_device_type> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_TYPE)?.to_ulong())
    }

    /// A unique device vendor identifier: a [PCI vendor ID](https://www.pcilookup.com/)
    /// or a Khronos vendor ID if the vendor does not have a PCI vendor ID.  
    pub fn vendor_id(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_VENDOR_ID)?.to_uint())
    }

    /// The number of parallel compute units on the device, minimum 1.  
    pub fn max_compute_units(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_COMPUTE_UNITS)?.to_uint())
    }

    /// Maximum dimensions for global and local work-item IDs, minimum 3
    /// if device is not CL_DEVICE_TYPE_CUSTOM.  
    pub fn max_work_item_dimensions(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)?.to_uint())
    }

    /// Maximum number of work-items for each dimension of a work-group,
    /// minimum [1, 1, 1] if device is not CL_DEVICE_TYPE_CUSTOM.  
    pub fn max_work_group_size(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_WORK_GROUP_SIZE)?.to_size())
    }

    pub fn max_work_item_sizes(&self) -> Result<Vec<size_t>> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_WORK_ITEM_SIZES)?.to_vec_size())
    }

    pub fn max_preferred_vector_width_char(&self) -> Result<cl_uint> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR)?
                .to_uint(),
        )
    }

    pub fn max_preferred_vector_width_short(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
        )?
        .to_uint())
    }

    pub fn max_preferred_vector_width_int(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT)?.to_uint())
    }

    pub fn max_preferred_vector_width_long(&self) -> Result<cl_uint> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG)?
                .to_uint(),
        )
    }

    pub fn max_preferred_vector_width_float(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
        )?
        .to_uint())
    }

    pub fn max_preferred_vector_width_double(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
        )?
        .to_uint())
    }

    pub fn max_clock_frequency(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_CLOCK_FREQUENCY)?.to_uint())
    }

    pub fn address_bits(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_ADDRESS_BITS)?.to_uint())
    }

    pub fn max_read_image_args(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_READ_IMAGE_ARGS)?.to_uint())
    }

    pub fn max_write_image_args(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_WRITE_IMAGE_ARGS)?.to_uint())
    }

    pub fn max_mem_alloc_size(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_MEM_ALLOC_SIZE)?.to_ulong())
    }

    pub fn image2d_max_width(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE2D_MAX_WIDTH)?.to_size())
    }

    pub fn image2d_max_height(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE2D_MAX_HEIGHT)?.to_size())
    }

    pub fn image3d_max_width(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE3D_MAX_WIDTH)?.to_size())
    }

    pub fn image3d_max_height(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE3D_MAX_HEIGHT)?.to_size())
    }

    pub fn image3d_max_depth(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE3D_MAX_DEPTH)?.to_size())
    }

    pub fn image_support(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE_SUPPORT)?.to_uint())
    }

    pub fn max_parameter_size(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_PARAMETER_SIZE)?.to_size())
    }

    pub fn max_device_samples(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_SAMPLERS)?.to_uint())
    }

    pub fn mem_base_addr_align(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MEM_BASE_ADDR_ALIGN)?.to_uint())
    }

    pub fn min_data_type_align_size(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE)?.to_uint())
    }

    pub fn single_fp_config(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_SINGLE_FP_CONFIG)?.to_ulong())
    }

    pub fn global_mem_cache_type(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GLOBAL_MEM_CACHE_TYPE)?.to_uint())
    }

    pub fn global_mem_cacheline_size(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE)?.to_uint())
    }

    pub fn global_mem_cache_size(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)?.to_ulong())
    }

    pub fn global_mem_size(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GLOBAL_MEM_SIZE)?.to_ulong())
    }

    pub fn max_constant_buffer_size(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)?.to_ulong())
    }

    pub fn max_constant_args(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_CONSTANT_ARGS)?.to_uint())
    }

    pub fn local_mem_type(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_LOCAL_MEM_TYPE)?.to_uint())
    }

    pub fn local_mem_size(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_LOCAL_MEM_SIZE)?.to_ulong())
    }

    pub fn error_correction_support(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_ERROR_CORRECTION_SUPPORT)?.to_uint())
    }

    pub fn profiling_timer_resolution(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PROFILING_TIMER_RESOLUTION)?.to_size())
    }

    pub fn endian_little(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_ENDIAN_LITTLE)?.to_uint())
    }

    pub fn available(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_AVAILABLE)?.to_uint())
    }

    pub fn compiler_available(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_COMPILER_AVAILABLE)?.to_uint())
    }

    pub fn execution_capabilities(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_EXECUTION_CAPABILITIES)?.to_ulong())
    }

    pub fn queue_on_host_properties(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_QUEUE_ON_HOST_PROPERTIES)?.to_ulong())
    }

    pub fn name(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NAME)?.to_string())
    }

    pub fn vendor(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_VENDOR)?.to_string())
    }

    pub fn driver_version(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DRIVER_VERSION)?.to_string())
    }

    pub fn profile(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PROFILE)?.to_string())
    }

    pub fn version(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_VERSION)?.to_string())
    }

    pub fn extensions(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_EXTENSIONS)?.to_string())
    }

    pub fn platform(&self) -> Result<cl_platform_id> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PLATFORM)?.to_ptr() as cl_platform_id)
    }

    pub fn double_fp_config(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_DOUBLE_FP_CONFIG)?.to_ulong())
    }

    pub fn half_fp_config(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_HALF_FP_CONFIG)?.to_ulong())
    }

    pub fn preferred_vector_width_half(&self) -> Result<cl_uint> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF)?
                .to_uint(),
        )
    }

    // DEPRECATED 2.0
    pub fn host_unified_memory(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_HOST_UNIFIED_MEMORY)?.to_uint())
    }

    pub fn native_vector_width_char(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR)?.to_uint())
    }

    pub fn native_vector_width_short(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT)?.to_uint())
    }

    pub fn native_vector_width_int(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_INT)?.to_uint())
    }

    pub fn native_vector_width_long(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG)?.to_uint())
    }

    pub fn native_vector_width_float(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT)?.to_uint())
    }

    pub fn native_vector_width_double(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE)?.to_uint())
    }

    pub fn native_vector_width_half(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF)?.to_uint())
    }

    pub fn opencl_c_version(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_OPENCL_C_VERSION)?.to_string())
    }

    pub fn linker_available(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_LINKER_AVAILABLE)?.to_uint())
    }

    pub fn built_in_kernels(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_BUILT_IN_KERNELS)?.to_string())
    }

    pub fn image_max_buffer_size(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE_MAX_BUFFER_SIZE)?.to_size())
    }

    pub fn image_max_array_size(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE_MAX_ARRAY_SIZE)?.to_size())
    }

    pub fn parent_device(&self) -> Result<cl_device_id> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_PARENT_DEVICE)?.to_ptr()
                as cl_device_id,
        )
    }

    pub fn partition_max_sub_devices(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PARTITION_MAX_SUB_DEVICES)?.to_uint())
    }
    pub fn partition_properties(&self) -> Result<Vec<intptr_t>> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PARTITION_PROPERTIES)?.to_vec_intptr())
    }

    pub fn partition_affinity_domain(&self) -> Result<Vec<cl_ulong>> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_PARTITION_AFFINITY_DOMAIN)?
                .to_vec_ulong(),
        )
    }

    pub fn partition_type(&self) -> Result<Vec<intptr_t>> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PARTITION_TYPE)?.to_vec_intptr())
    }

    pub fn reference_count(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_REFERENCE_COUNT)?.to_uint())
    }

    pub fn preferred_interop_user_sync(&self) -> Result<cl_uint> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_PREFERRED_INTEROP_USER_SYNC)?
                .to_uint(),
        )
    }

    pub fn printf_buffer_size(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PRINTF_BUFFER_SIZE)?.to_size())
    }

    // CL_VERSION_2_0
    pub fn image_pitch_alignment(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IMAGE_PITCH_ALIGNMENT)?.to_uint())
    }

    pub fn image_base_address_alignment(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT,
        )?
        .to_uint())
    }

    pub fn max_read_write_image_args(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS)?.to_uint())
    }

    pub fn max_global_variable_size(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE)?.to_size())
    }

    pub fn queue_on_device_properties(&self) -> Result<Vec<intptr_t>> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES)?
                .to_vec_intptr(),
        )
    }

    pub fn queue_on_device_preferred_size(&self) -> Result<size_t> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE,
        )?
        .to_size())
    }

    pub fn queue_on_device_max_size(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE)?.to_size())
    }

    pub fn max_on_device_queues(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_ON_DEVICE_QUEUES)?.to_uint())
    }

    pub fn max_on_device_events(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_ON_DEVICE_EVENTS)?.to_uint())
    }

    pub fn svm_capabilities(&self) -> Result<cl_device_svm_capabilities> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_SVM_CAPABILITIES)?.to_ulong())
    }

    pub fn global_variable_preferred_total_size(&self) -> Result<size_t> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE,
        )?
        .to_size())
    }

    pub fn max_pipe_args(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_PIPE_ARGS)?.to_uint())
    }

    pub fn pipe_max_active_reservations(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS,
        )?
        .to_uint())
    }

    pub fn pipe_max_packet_size(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PIPE_MAX_PACKET_SIZE)?.to_uint())
    }

    pub fn preferred_platform_atomic_alignment(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT,
        )?
        .to_uint())
    }

    pub fn preferred_global_atomic_alignment(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT,
        )?
        .to_uint())
    }

    pub fn preferred_local_atomic_alignment(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT,
        )?
        .to_uint())
    }

    // CL_VERSION_2_1
    pub fn il_version(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_IL_VERSION)?.to_string())
    }

    pub fn max_num_sub_groups(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_NUM_SUB_GROUPS)?.to_uint())
    }

    pub fn sub_group_independent_forward_progress(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
        )?
        .to_uint())
    }

    // CL_VERSION_3_0
    pub fn numeric_version(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NUMERIC_VERSION)?.to_uint())
    }

    pub fn extensions_with_version(&self) -> Result<Vec<cl_name_version>> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_EXTENSIONS_WITH_VERSION)?
                .to_vec_name_version(),
        )
    }

    pub fn ils_with_version(&self) -> Result<Vec<cl_name_version>> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_ILS_WITH_VERSION)?
                .to_vec_name_version(),
        )
    }

    pub fn built_in_kernels_with_version(&self) -> Result<Vec<cl_name_version>> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION,
        )?
        .to_vec_name_version())
    }
    pub fn atomic_memory_capabilities(&self) -> Result<cl_ulong> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES)?
                .to_ulong(),
        )
    }

    pub fn atomic_fence_capabilities(&self) -> Result<cl_ulong> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_ATOMIC_FENCE_CAPABILITIES)?.to_ulong())
    }

    pub fn non_uniform_work_group_support(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT,
        )?
        .to_uint())
    }

    pub fn opencl_c_all_versions(&self) -> Result<Vec<cl_name_version>> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_OPENCL_C_ALL_VERSIONS)?
                .to_vec_name_version(),
        )
    }

    pub fn preferred_work_group_size_multiple(&self) -> Result<size_t> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        )?
        .to_size())
    }

    pub fn work_group_collective_functions_support(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT,
        )?
        .to_uint())
    }

    pub fn generic_address_space_support(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
        )?
        .to_uint())
    }

    pub fn uuid_khr(&self) -> Result<Vec<u8>> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_UUID_KHR)?.to_vec_uchar())
    }

    pub fn driver_uuid_khr(&self) -> Result<Vec<u8>> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DRIVER_UUID_KHR)?.to_vec_uchar())
    }

    pub fn luid_valid_khr(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_LUID_VALID_KHR)?.to_uint())
    }

    pub fn luid_khr(&self) -> Result<Vec<u8>> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_LUID_KHR)?.to_vec_uchar())
    }

    pub fn node_mask_khr(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_NODE_MASK_KHR)?.to_uint())
    }

    pub fn opencl_c_features(&self) -> Result<Vec<cl_name_version>> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_OPENCL_C_FEATURES)?
                .to_vec_name_version(),
        )
    }

    pub fn device_enqueue_capabilities(&self) -> Result<cl_ulong> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES)?
                .to_ulong(),
        )
    }

    pub fn pipe_support(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PIPE_SUPPORT)?.to_uint())
    }

    pub fn latest_conformance_version_passed(&self) -> Result<String> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED,
        )?
        .to_string())
    }

    pub fn compute_capability_major_nv(&self) -> Result<cl_uint> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV)?
                .to_uint(),
        )
    }

    pub fn compute_capability_minor_nv(&self) -> Result<cl_uint> {
        Ok(
            get_device_info(self.id(), DeviceInfo::CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV)?
                .to_uint(),
        )
    }

    pub fn registers_per_block_nv(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_REGISTERS_PER_BLOCK_NV)?.to_uint())
    }

    pub fn wrap_size_nv(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_WARP_SIZE_NV)?.to_uint())
    }

    pub fn gpu_overlap_nv(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GPU_OVERLAP_NV)?.to_uint())
    }

    pub fn compute_kernel_exec_timeout_nv(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV)?.to_uint())
    }

    pub fn integrated_memory_nv(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_INTEGRATED_MEMORY_NV)?.to_uint())
    }

    pub fn pci_bus_id_nv(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PCI_BUS_ID_NV)?.to_uint())
    }

    pub fn pci_slot_id_nv(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PCI_SLOT_ID_NV)?.to_uint())
    }

    pub fn profiling_timer_offset_amd(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PROFILING_TIMER_OFFSET_AMD)?.to_size())
    }

    pub fn topology_amd(&self) -> Result<cl_amd_device_topology> {
        let value = get_device_info(self.id(), DeviceInfo::CL_DEVICE_TOPOLOGY_AMD)?.to_vec_uchar();
        Ok(get_amd_device_topology(&value))
    }

    pub fn pci_bus_id_amd(&self) -> Result<cl_uint> {
        let value = self.topology_amd()?;
        Ok(value.bus as cl_uint)
    }

    pub fn board_name_amd(&self) -> Result<String> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_BOARD_NAME_AMD)?.to_string())
    }

    pub fn global_free_memory_amd(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GLOBAL_FREE_MEMORY_AMD)?.to_size())
    }

    pub fn simd_per_compute_unit_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD)?.to_uint())
    }

    pub fn simd_width_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_SIMD_WIDTH_AMD)?.to_uint())
    }

    pub fn simd_instruction_width_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD)?.to_uint())
    }

    pub fn wavefront_width_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_WAVEFRONT_WIDTH_AMD)?.to_uint())
    }

    pub fn global_mem_channels_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD)?.to_uint())
    }

    pub fn global_mem_channel_banks_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD,
        )?
        .to_uint())
    }

    pub fn global_mem_channel_bank_width_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD,
        )?
        .to_uint())
    }

    pub fn local_mem_size_per_compute_unit_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD,
        )?
        .to_uint())
    }

    pub fn local_mem_banks_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_LOCAL_MEM_BANKS_AMD)?.to_uint())
    }

    pub fn thread_trace_supported_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD)?.to_uint())
    }

    pub fn gfxip_major_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GFXIP_MAJOR_AMD)?.to_uint())
    }

    pub fn gfxip_minor_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_GFXIP_MINOR_AMD)?.to_uint())
    }

    pub fn available_async_queues_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD)?.to_uint())
    }

    pub fn preferred_work_group_size_amd(&self) -> Result<size_t> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD,
        )?
        .to_size())
    }

    pub fn max_work_group_size_amd(&self) -> Result<size_t> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD)?.to_size())
    }

    pub fn preferred_constant_buffer_size_amd(&self) -> Result<size_t> {
        Ok(get_device_info(
            self.id(),
            DeviceInfo::CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD,
        )?
        .to_size())
    }

    pub fn pcie_id_amd(&self) -> Result<cl_uint> {
        Ok(get_device_info(self.id(), DeviceInfo::CL_DEVICE_PCIE_ID_AMD)?.to_uint())
    }

    /// Determine if the device supports the given half floating point capability.  
    /// Returns true if the device supports it, false otherwise.
    pub fn supports_half(&self, min_fp_capability: cl_device_fp_config) -> bool {
        if let Ok(fp) = self.half_fp_config() {
            0 < fp & min_fp_capability
        } else {
            false
        }
    }
    /// Determine if the device supports the given double floating point capability.  
    /// Returns true if the device supports it, false otherwise.
    pub fn supports_double(&self, min_fp_capability: cl_device_fp_config) -> bool {
        if let Ok(fp) = self.double_fp_config() {
            0 < fp & min_fp_capability
        } else {
            false
        }
    }

    /// Determine if the device supports SVM and, if so, what kind of SVM.  
    /// Returns zero if the device does not support SVM.
    pub fn svm_mem_capability(&self) -> cl_device_svm_capabilities {
        if let Ok(svm) = self.svm_capabilities() {
            svm
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::get_platforms;
    use std::ptr;

    #[test]
    fn test_get_devices() {
        let platforms = get_platforms().unwrap();
        println!("Number of platforms: {}", platforms.len());
        assert!(0 < platforms.len());

        for platform in platforms {
            println!("CL_PLATFORM_NAME: {}", platform.name().unwrap());

            let devices = platform.get_devices(CL_DEVICE_TYPE_ALL).unwrap();
            for device_id in devices {
                let device = Device::new(device_id);

                println!("Device Debug Trait: {:?}", device);
                println!("\tCL_DEVICE_NAME: {}", device.name().unwrap());
                println!("\tCL_DEVICE_TYPE: {:X}", device.dev_type().unwrap());
                println!("\tCL_DEVICE_VENDOR_ID: {:X}", device.vendor_id().unwrap());
                println!("\tCL_DEVICE_VENDOR: {}", device.vendor().unwrap());
                println!(
                    "\tCL_DEVICE_OPENCL_C_VERSION: {:?}",
                    device.opencl_c_version().unwrap()
                );
                println!("");
            }
        }
    }

    #[test]
    fn test_get_sub_devices() {
        let platforms = get_platforms().unwrap();
        println!("Number of platforms: {}", platforms.len());
        assert!(0 < platforms.len());

        // Find an OpenCL device with sub devices
        let mut device_id = ptr::null_mut();
        let mut has_sub_devices: bool = false;

        for platform in platforms {
            let device_ids = platform.get_devices(CL_DEVICE_TYPE_CPU).unwrap();

            for dev_id in device_ids {
                let device = Device::new(dev_id);
                let max_sub_devices = device.partition_max_sub_devices().unwrap();

                has_sub_devices = 1 < max_sub_devices;
                if has_sub_devices {
                    device_id = dev_id;
                    break;
                }
            }
        }

        if has_sub_devices {
            let device = Device::new(device_id);
            let properties: [cl_device_partition_property; 3] = [CL_DEVICE_PARTITION_EQUALLY, 2, 0];
            let sub_devices = device.create_sub_devices(&properties).unwrap();

            println!("sub_devices len: {}", sub_devices.len());
            assert!(0 < sub_devices.len());
        } else {
            println!("OpenCL device capable of sub division not found");
        }
    }

    #[test]
    fn test_device_info() {
        let platforms = get_platforms().unwrap();
        assert!(!platforms.is_empty());

        // Choose the first platform
        let platform = &platforms[0];

        let devices = platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap();
        println!("CL_DEVICE_TYPE_GPU count: {}", devices.len());
        assert!(!devices.is_empty());

        // Choose the first device
        let device_id = devices[0];
        let device = Device::new(device_id);

        let value = device.dev_type().unwrap();
        assert_eq!(CL_DEVICE_TYPE_GPU, value);

        let value = device.vendor_id().unwrap();
        println!("CL_DEVICE_VENDOR_ID: {:X}", value);
        assert!(0 < value);

        let vendor_text = match value {
            0x1002 => "AMD",
            0x10DE => "Nvidia",
            0x8086 => "Intel",
            _ => "unknown",
        };
        println!("Device vendor is: {}", vendor_text);

        let value = device.max_compute_units().unwrap();
        println!("CL_DEVICE_MAX_COMPUTE_UNITS: {}", value);
        assert!(0 < value);

        let value = device.max_work_item_dimensions().unwrap();
        println!("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: {}", value);
        assert!(0 < value);

        let value = device.max_work_group_size().unwrap();
        println!("CL_DEVICE_MAX_WORK_GROUP_SIZE: {}", value);
        assert!(0 < value);

        let value = device.max_work_item_sizes().unwrap();
        println!("CL_DEVICE_MAX_WORK_ITEM_SIZES: {}", value.len());
        println!("CL_DEVICE_MAX_WORK_ITEM_SIZES: {:?}", value);
        assert!(0 < value.len());

        let value = device.max_preferred_vector_width_char().unwrap();
        println!("CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: {}", value);
        assert!(0 < value);

        let value = device.max_preferred_vector_width_short().unwrap();
        println!("CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: {}", value);
        assert!(0 < value);

        let value = device.max_preferred_vector_width_int().unwrap();
        println!("CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: {}", value);
        assert!(0 < value);

        let value = device.max_preferred_vector_width_long().unwrap();
        println!("CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: {}", value);
        assert!(0 < value);

        let value = device.max_preferred_vector_width_float().unwrap();
        println!("CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: {}", value);
        assert!(0 < value);

        let value = device.max_preferred_vector_width_double().unwrap();
        println!("CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: {}", value);
        assert!(0 < value);

        let value = device.max_clock_frequency().unwrap();
        println!("CL_DEVICE_MAX_CLOCK_FREQUENCY: {}", value);
        assert!(0 < value);

        let value = device.address_bits().unwrap();
        println!("CL_DEVICE_ADDRESS_BITS: {}", value);
        assert!(0 < value);

        let value = device.max_read_image_args().unwrap();
        println!("CL_DEVICE_MAX_READ_IMAGE_ARGS: {}", value);
        assert!(0 < value);

        let value = device.max_write_image_args().unwrap();
        println!("CL_DEVICE_MAX_WRITE_IMAGE_ARGS: {}", value);
        assert!(0 < value);

        let value = device.max_mem_alloc_size().unwrap();
        println!("CL_DEVICE_MAX_MEM_ALLOC_SIZE: {}", value);
        assert!(0 < value);

        let value = device.image2d_max_width().unwrap();
        println!("CL_DEVICE_IMAGE2D_MAX_WIDTH: {}", value);
        assert!(0 < value);

        let value = device.image2d_max_height().unwrap();
        println!("CL_DEVICE_IMAGE2D_MAX_HEIGHT: {}", value);
        assert!(0 < value);

        let value = device.image3d_max_width().unwrap();
        println!("CL_DEVICE_IMAGE3D_MAX_WIDTH: {}", value);
        assert!(0 < value);

        let value = device.image3d_max_height().unwrap();
        println!("CL_DEVICE_IMAGE3D_MAX_HEIGHT: {}", value);
        assert!(0 < value);

        let value = device.image3d_max_depth().unwrap();
        println!("CL_DEVICE_IMAGE3D_MAX_DEPTH: {}", value);
        assert!(0 < value);

        let value = device.image_support().unwrap();
        println!("CL_DEVICE_IMAGE_SUPPORT: {}", value);
        assert!(0 < value);

        let value = device.max_parameter_size().unwrap();
        println!("CL_DEVICE_MAX_PARAMETER_SIZE: {}", value);
        assert!(0 < value);

        let value = device.max_device_samples().unwrap();
        println!("CL_DEVICE_MAX_SAMPLERS: {}", value);
        assert!(0 < value);

        let value = device.mem_base_addr_align().unwrap();
        println!("CL_DEVICE_MEM_BASE_ADDR_ALIGN: {}", value);
        assert!(0 < value);

        let value = device.min_data_type_align_size().unwrap();
        println!("CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: {}", value);
        assert!(0 < value);
        let value = device.single_fp_config().unwrap();
        println!("CL_DEVICE_SINGLE_FP_CONFIG: {:X}", value);
        assert!(0 < value);

        let value = device.global_mem_cache_type().unwrap();
        println!("CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: {:X}", value);
        assert!(0 < value);

        let value = device.global_mem_cacheline_size().unwrap();
        println!("CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: {}", value);
        assert!(0 < value);

        let value = device.global_mem_cache_size().unwrap();
        println!("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: {}", value);
        assert!(0 < value);

        let value = device.global_mem_size().unwrap();
        println!("CL_DEVICE_GLOBAL_MEM_SIZE: {}", value);
        assert!(0 < value);
        let value = device.max_constant_buffer_size().unwrap();
        println!("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: {}", value);
        assert!(0 < value);

        let value = device.max_constant_args().unwrap();
        println!("CL_DEVICE_MAX_CONSTANT_ARGS: {}", value);
        assert!(0 < value);

        let value = device.local_mem_type().unwrap();
        println!("CL_DEVICE_LOCAL_MEM_TYPE: {:X}", value);
        assert!(0 < value);

        let value = device.local_mem_size().unwrap();
        println!("CL_DEVICE_LOCAL_MEM_SIZE: {}", value);
        assert!(0 < value);

        let value = device.error_correction_support().unwrap();
        println!("CL_DEVICE_ERROR_CORRECTION_SUPPORT: {}", value);

        let value = device.profiling_timer_resolution().unwrap();
        println!("CL_DEVICE_PROFILING_TIMER_RESOLUTION: {}", value);
        assert!(0 < value);

        let value = device.endian_little().unwrap();
        println!("CL_DEVICE_ENDIAN_LITTLE: {}", value);
        assert!(0 < value);

        let value = device.available().unwrap();
        println!("CL_DEVICE_AVAILABLE: {}", value);
        assert!(0 < value);

        let value = device.compiler_available().unwrap();
        println!("CL_DEVICE_COMPILER_AVAILABLE: {}", value);
        assert!(0 < value);

        let value = device.execution_capabilities().unwrap();
        println!("CL_DEVICE_EXECUTION_CAPABILITIES: {:X}", value);
        assert!(0 < value);

        // CL_VERSION_2_0
        match device.queue_on_host_properties() {
            Ok(value) => {
                println!("CL_DEVICE_QUEUE_ON_HOST_PROPERTIES: {:X}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES: {:?}, {}",
                e, e
            ),
        };

        let value = device.name().unwrap();
        println!("CL_DEVICE_NAME: {}", value);
        assert!(!value.is_empty());

        let value = device.vendor().unwrap();
        println!("CL_DEVICE_VENDOR: {}", value);
        assert!(!value.is_empty());

        let value = device.driver_version().unwrap();
        println!("CL_DRIVER_VERSION: {}", value);
        assert!(!value.is_empty());

        let value = device.profile().unwrap();
        println!("CL_DEVICE_PROFILE: {}", value);
        assert!(!value.is_empty());

        let value = device.version().unwrap();
        println!("CL_DEVICE_VERSION: {}", value);
        assert!(!value.is_empty());

        let value = device.extensions().unwrap();
        println!("CL_DEVICE_EXTENSIONS: {}", value);
        assert!(!value.is_empty());

        let value = device.platform().unwrap();
        println!("CL_DEVICE_PLATFORM: {:X}", value as intptr_t);
        assert_eq!(platform.id(), value);

        // Device may not support double fp precision
        match device.double_fp_config() {
            Ok(value) => {
                println!("CL_DEVICE_DOUBLE_FP_CONFIG: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_DOUBLE_FP_CONFIG: {:?}, {}", e, e),
        };

        // Device may not support half fp precision
        match device.half_fp_config() {
            Ok(value) => {
                println!("CL_DEVICE_HALF_FP_CONFIG: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_HALF_FP_CONFIG: {:?}, {}", e, e),
        };

        let value = device.preferred_vector_width_half().unwrap();
        println!("CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: {}", value);

        let value = device.host_unified_memory().unwrap();
        println!("CL_DEVICE_HOST_UNIFIED_MEMORY: {}", value);

        let value = device.native_vector_width_char().unwrap();
        println!("CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: {}", value);
        assert!(0 < value);

        let value = device.native_vector_width_short().unwrap();
        println!("CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: {}", value);
        assert!(0 < value);

        let value = device.native_vector_width_int().unwrap();
        println!("CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: {}", value);
        assert!(0 < value);

        let value = device.native_vector_width_long().unwrap();
        println!("CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: {}", value);
        assert!(0 < value);

        let value = device.native_vector_width_float().unwrap();
        println!("CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: {}", value);
        assert!(0 < value);

        let value = device.native_vector_width_double().unwrap();
        println!("CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: {}", value);

        let value = device.native_vector_width_half().unwrap();
        println!("CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF: {}", value);

        let value = device.opencl_c_version().unwrap();
        println!("CL_DEVICE_OPENCL_C_VERSION: {}", value);
        assert!(!value.is_empty());

        let value = device.linker_available().unwrap();
        println!("CL_DEVICE_LINKER_AVAILABLE: {}", value);

        let value = device.built_in_kernels().unwrap();
        println!("CL_DEVICE_BUILT_IN_KERNELS: {:?}", value);

        let value = device.image_max_buffer_size().unwrap();
        println!("CL_DEVICE_IMAGE_MAX_BUFFER_SIZE: {}", value);
        assert!(0 < value);

        let value = device.image_max_array_size().unwrap();
        println!("CL_DEVICE_IMAGE_MAX_ARRAY_SIZE: {}", value);
        assert!(0 < value);

        let value = device.parent_device().unwrap();
        println!("CL_DEVICE_PARENT_DEVICE: {:X}", value as intptr_t);
        let value = device.partition_max_sub_devices().unwrap();
        println!("CL_DEVICE_PARTITION_MAX_SUB_DEVICES: {}", value);

        let value = device.partition_properties().unwrap();
        println!("CL_DEVICE_PARTITION_PROPERTIES: {:?}", value);
        assert!(0 < value.len());

        let value = device.partition_affinity_domain().unwrap();
        println!("CL_DEVICE_PARTITION_AFFINITY_DOMAIN: {:?}", value);
        assert!(0 < value.len());

        let value = device.partition_type().unwrap();
        println!("CL_DEVICE_PARTITION_TYPE: {:?}", value);
        // assert!(0 < value.len());

        let value = device.reference_count().unwrap();
        println!("CL_DEVICE_REFERENCE_COUNT: {}", value);
        assert!(0 < value);

        let value = device.preferred_interop_user_sync().unwrap();
        println!("CL_DEVICE_PREFERRED_INTEROP_USER_SYNC: {:X}", value);

        let value = device.printf_buffer_size().unwrap();
        println!("CL_DEVICE_PRINTF_BUFFER_SIZE: {}", value);
        assert!(0 < value);

        //////////////////////////////////////////////////////////////////////
        // CL_VERSION_2_0 parameters
        match device.image_pitch_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_IMAGE_PITCH_ALIGNMENT: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_IMAGE_PITCH_ALIGNMENT: {:?}, {}",
                e, e
            ),
        };

        match device.image_base_address_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT: {:?}, {}",
                e, e
            ),
        };

        match device.max_read_write_image_args() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: {:?}, {}",
                e, e
            ),
        };

        match device.max_global_variable_size() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: {:?}, {}",
                e, e
            ),
        };

        match device.queue_on_device_properties() {
            Ok(value) => {
                println!("CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: {:?}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: {:?}, {}",
                e, e
            ),
        };

        match device.queue_on_device_preferred_size() {
            Ok(value) => {
                println!("CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: {:?}, {}",
                e, e
            ),
        };

        match device.queue_on_device_max_size() {
            Ok(value) => {
                println!("CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: {:?}, {}",
                e, e
            ),
        };

        match device.max_on_device_queues() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_ON_DEVICE_QUEUES: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_MAX_ON_DEVICE_QUEUES: {:?}, {}",
                e, e
            ),
        };

        match device.max_on_device_events() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_ON_DEVICE_EVENTS: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_MAX_ON_DEVICE_EVENTS: {:?}, {}",
                e, e
            ),
        };

        match device.svm_capabilities() {
            Ok(value) => {
                println!("CL_DEVICE_SVM_CAPABILITIES: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_SVM_CAPABILITIES: {:?}, {}", e, e),
        };

        match device.global_variable_preferred_total_size() {
            Ok(value) => {
                println!("CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: {:?}, {}",
                e, e
            ),
        };

        match device.max_pipe_args() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_PIPE_ARGS: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_MAX_PIPE_ARGS: {:?}, {}", e, e),
        };

        match device.pipe_max_active_reservations() {
            Ok(value) => {
                println!("CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS: {:?}, {}",
                e, e
            ),
        };

        match device.pipe_max_packet_size() {
            Ok(value) => {
                println!("CL_DEVICE_PIPE_MAX_PACKET_SIZE: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PIPE_MAX_PACKET_SIZE: {:?}, {}",
                e, e
            ),
        };

        match device.preferred_platform_atomic_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT: {:?}, {}",
                e, e
            ),
        };

        match device.preferred_global_atomic_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT: {:?}, {}",
                e, e
            ),
        };

        match device.preferred_local_atomic_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT: {:?}, {}",
                e, e
            ),
        };

        // //////////////////////////////////////////////////////////////////////
        // // CL_VERSION_2_1 parameters

        match device.il_version() {
            Ok(value) => {
                println!("CL_DEVICE_IL_VERSION: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_IL_VERSION: {:?}, {}", e, e),
        };

        match device.max_num_sub_groups() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_NUM_SUB_GROUPS: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_MAX_NUM_SUB_GROUPS: {:?}, {}", e, e),
        };

        match device.sub_group_independent_forward_progress() {
            Ok(value) => {
                println!(
                    "CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {:?}",
                    value
                )
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS:{:?}, {}",
                e, e
            ),
        };

        //////////////////////////////////////////////////////////////////////
        // CL_VERSION_3_0 parameters

        match device.numeric_version() {
            Ok(value) => {
                println!("CL_DEVICE_NUMERIC_VERSION: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_NUMERIC_VERSION: {:?}, {}", e, e),
        };

        match device.extensions_with_version() {
            Ok(value) => {
                println!("CL_DEVICE_EXTENSIONS_WITH_VERSION: {:?}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_EXTENSIONS_WITH_VERSION: {:?}, {}",
                e, e
            ),
        };

        match device.ils_with_version() {
            Ok(value) => {
                println!("CL_DEVICE_ILS_WITH_VERSION: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_ILS_WITH_VERSION: {:?}, {}", e, e),
        };

        match device.built_in_kernels_with_version() {
            Ok(value) => {
                println!("CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION: {:?}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION: {:?}, {}",
                e, e
            ),
        };

        match device.atomic_memory_capabilities() {
            Ok(value) => {
                println!("CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: {:X}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: {:?}, {}",
                e, e
            ),
        };

        match device.atomic_fence_capabilities() {
            Ok(value) => {
                println!("CL_DEVICE_ATOMIC_FENCE_CAPABILITIES: {:X}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES: {:?}, {}",
                e, e
            ),
        };

        match device.non_uniform_work_group_support() {
            Ok(value) => {
                println!("CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT: {:?}, {}",
                e, e
            ),
        };

        match device.opencl_c_all_versions() {
            Ok(value) => {
                println!("CL_DEVICE_OPENCL_C_ALL_VERSIONS: {:?}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_OPENCL_C_ALL_VERSIONS: {:?}, {}",
                e, e
            ),
        };

        match device.preferred_work_group_size_multiple() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {:?}, {}",
                e, e
            ),
        };

        match device.work_group_collective_functions_support() {
            Ok(value) => {
                println!(
                    "CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: {:X}",
                    value
                )
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: {:?}, {}",
                e, e
            ),
        };

        match device.generic_address_space_support() {
            Ok(value) => {
                println!("CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT: {:X}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT: {:?}, {}",
                e, e
            ),
        };

        match device.uuid_khr() {
            Ok(value) => {
                println!("CL_DEVICE_UUID_KHR: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_UUID_KHR: {:?}, {}", e, e),
        };

        match device.driver_uuid_khr() {
            Ok(value) => {
                println!("CL_DRIVER_UUID_KHR: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DRIVER_UUID_KHR: {:?}, {}", e, e),
        };

        match device.luid_valid_khr() {
            Ok(value) => {
                println!("CL_DEVICE_LUID_VALID_KHR: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_LUID_VALID_KHR: {:?}, {}", e, e),
        };

        match device.luid_khr() {
            Ok(value) => {
                println!("CL_DEVICE_LUID_KHR: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_LUID_KHR: {:?}, {}", e, e),
        };

        match device.node_mask_khr() {
            Ok(value) => {
                println!("CL_DEVICE_NODE_MASK_KHR: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_NODE_MASK_KHR: {:?}, {}", e, e),
        };

        match device.opencl_c_features() {
            Ok(value) => {
                println!("CL_DEVICE_OPENCL_C_FEATURES: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_OPENCL_C_FEATURES: {:?}, {}", e, e),
        };

        match device.device_enqueue_capabilities() {
            Ok(value) => {
                println!("CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES: {:X}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES: {:?}, {}",
                e, e
            ),
        };

        match device.pipe_support() {
            Ok(value) => {
                println!("CL_DEVICE_PIPE_SUPPORT: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PIPE_SUPPORT: {:?}, {}", e, e),
        };

        match device.latest_conformance_version_passed() {
            Ok(value) => {
                println!("CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED: {:?}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED: {:?}, {}",
                e, e
            ),
        };

        match device.compute_capability_major_nv() {
            Ok(value) => {
                println!("CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV: {:?}, {}",
                e, e
            ),
        };

        match device.compute_capability_minor_nv() {
            Ok(value) => {
                println!("CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV: {:?}, {}",
                e, e
            ),
        };

        match device.registers_per_block_nv() {
            Ok(value) => {
                println!("CL_DEVICE_REGISTERS_PER_BLOCK_NV: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_REGISTERS_PER_BLOCK_NV: {:?}, {}",
                e, e
            ),
        };

        match device.wrap_size_nv() {
            Ok(value) => {
                println!("CL_DEVICE_WARP_SIZE_NV: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_WARP_SIZE_NV: {:?}, {}", e, e),
        };

        match device.gpu_overlap_nv() {
            Ok(value) => {
                println!("CL_DEVICE_GPU_OVERLAP_NV: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_GPU_OVERLAP_NV: {:?}, {}", e, e),
        };

        match device.compute_kernel_exec_timeout_nv() {
            Ok(value) => {
                println!("CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV: {:?}, {}",
                e, e
            ),
        };

        match device.integrated_memory_nv() {
            Ok(value) => {
                println!("CL_DEVICE_INTEGRATED_MEMORY_NV: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_INTEGRATED_MEMORY_NV: {:?}, {}",
                e, e
            ),
        };

        match device.pci_bus_id_nv() {
            Ok(value) => {
                println!("CL_DEVICE_PCI_BUS_ID_NV: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PCI_BUS_ID_NV: {:?}, {}", e, e),
        };

        match device.profiling_timer_offset_amd() {
            Ok(value) => {
                println!("CL_DEVICE_PROFILING_TIMER_OFFSET_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PROFILING_TIMER_OFFSET_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.topology_amd() {
            Ok(value) => {
                println!("CL_DEVICE_TOPOLOGY_AMD: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_TOPOLOGY_AMD: {:?}, {}", e, e),
        };

        match device.pci_bus_id_amd() {
            Ok(value) => {
                println!("pci_bus_id_amd: {}", value)
            }
            Err(e) => println!("OpenCL error, pci_bus_id_amd: {:?}, {}", e, e),
        };

        match device.board_name_amd() {
            Ok(value) => {
                println!("CL_DEVICE_BOARD_NAME_AMD: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_BOARD_NAME_AMD: {:?}, {}", e, e),
        };

        match device.global_free_memory_amd() {
            Ok(value) => {
                println!("CL_DEVICE_GLOBAL_FREE_MEMORY_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_GLOBAL_FREE_MEMORY_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.simd_per_compute_unit_amd() {
            Ok(value) => {
                println!("CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.simd_width_amd() {
            Ok(value) => {
                println!("CL_DEVICE_SIMD_WIDTH_AMD: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_SIMD_WIDTH_AMD: {:?}, {}", e, e),
        };

        match device.simd_instruction_width_amd() {
            Ok(value) => {
                println!("CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.wavefront_width_amd() {
            Ok(value) => {
                println!("CL_DEVICE_WAVEFRONT_WIDTH_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_WAVEFRONT_WIDTH_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.global_mem_channels_amd() {
            Ok(value) => {
                println!("CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.global_mem_channel_banks_amd() {
            Ok(value) => {
                println!("CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.global_mem_channel_bank_width_amd() {
            Ok(value) => {
                println!("CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.local_mem_size_per_compute_unit_amd() {
            Ok(value) => {
                println!("CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.local_mem_banks_amd() {
            Ok(value) => {
                println!("CL_DEVICE_LOCAL_MEM_BANKS_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_LOCAL_MEM_BANKS_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.thread_trace_supported_amd() {
            Ok(value) => {
                println!("CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.gfxip_major_amd() {
            Ok(value) => {
                println!("CL_DEVICE_GFXIP_MAJOR_AMD: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_GFXIP_MAJOR_AMD: {:?}, {}", e, e),
        };

        match device.gfxip_minor_amd() {
            Ok(value) => {
                println!("CL_DEVICE_GFXIP_MINOR_AMD: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_GFXIP_MINOR_AMD: {:?}, {}", e, e),
        };

        match device.available_async_queues_amd() {
            Ok(value) => {
                println!("CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.preferred_work_group_size_amd() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.max_work_group_size_amd() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.preferred_constant_buffer_size_amd() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD: {}", value)
            }
            Err(e) => println!(
                "OpenCL error, CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD: {:?}, {}",
                e, e
            ),
        };

        match device.pcie_id_amd() {
            Ok(value) => {
                println!("CL_DEVICE_PCIE_ID_AMD: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PCIE_ID_AMD: {:?}, {}", e, e),
        };
    }
}
