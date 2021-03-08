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

pub use cl3::device::*;

use cl3::types::{
    cl_device_fp_config, cl_device_id, cl_device_partition_property, cl_device_svm_capabilities,
    cl_device_type, cl_int, cl_uint, cl_ulong, cl_name_version,
};
use libc::{intptr_t, size_t};
use std::ffi::CString;

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

pub struct SubDevice {
    id: cl_device_id,
}

impl Drop for SubDevice {
    fn drop(&mut self) {
        release_device(self.id).unwrap();
        println!("SubDevice::drop");
    }
}

impl SubDevice {
    pub fn new(id: cl_device_id) -> SubDevice {
        SubDevice { id }
    }

    /// Accessor for the underlying device id.
    pub fn id(&self) -> cl_device_id {
        self.id
    }
}

unsafe impl Send for SubDevice {}
unsafe impl Sync for SubDevice {}

/// An OpenCL device id and methods to query it.  
/// The query methods calls clGetDeviceInfo with the relevant param_name, see:
/// [Device Queries](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#device-queries-table).
pub struct Device {
    id: cl_device_id,
}

impl Device {
    pub fn new(id: cl_device_id) -> Device {
        Device { id }
    }

    /// Accessor for the underlying device id.
    pub fn id(&self) -> cl_device_id {
        self.id
    }

    /// Create sub-devices by partitioning an OpenCL device.
    ///
    /// * `properties` - the slice of cl_device_partition_property, see
    /// [Subdevice Partition](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#subdevice-partition-table).
    ///
    /// returns a Result containing a vector of available sub-device ids
    /// or the error code from the OpenCL C API function.
    pub fn create_sub_devices(
        &self,
        properties: &[cl_device_partition_property],
    ) -> Result<Vec<cl_device_id>, cl_int> {
        create_sub_devices(self.id, &properties)
    }

    #[cfg(feature = "CL_VERSION_2_1")]
    #[inline]
    pub fn get_device_and_host_timer(&self) -> Result<[cl_ulong; 2], cl_int> {
        get_device_and_host_timer(self.id)
    }

    #[cfg(feature = "CL_VERSION_2_1")]
    #[inline]
    pub fn get_host_timer(&self) -> Result<cl_ulong, cl_int> {
        get_host_timer(self.id)
    }

    /// The OpenCL device type, see
    /// [Device Types](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#device-types-table).  
    pub fn dev_type(&self) -> Result<cl_device_type, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_TYPE)?.to_ulong())
    }

    /// A unique device vendor identifier: a [PCI vendor ID](https://www.pcilookup.com/)
    /// or a Khronos vendor ID if the vendor does not have a PCI vendor ID.  
    pub fn vendor_id(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_VENDOR_ID)?.to_uint())
    }

    /// The number of parallel compute units on the device, minimum 1.  
    pub fn max_compute_units(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_COMPUTE_UNITS)?.to_uint())
    }

    /// Maximum dimensions for global and local work-item IDs, minimum 3
    /// if device is not CL_DEVICE_TYPE_CUSTOM.  
    pub fn max_work_item_dimensions(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)?.to_uint())
    }

    /// Maximum number of work-items for each dimension of a work-group,
    /// minimum [1, 1, 1] if device is not CL_DEVICE_TYPE_CUSTOM.  
    pub fn max_work_group_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_WORK_GROUP_SIZE)?.to_size())
    }

    pub fn max_work_item_sizes(&self) -> Result<Vec<size_t>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_WORK_ITEM_SIZES)?.to_vec_size())
    }

    pub fn max_preferred_vector_width_char(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR)?.to_uint())
    }

    pub fn max_preferred_vector_width_short(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT)?.to_uint())
    }

    pub fn max_preferred_vector_width_int(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT)?.to_uint())
    }

    pub fn max_preferred_vector_width_long(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG)?.to_uint())
    }

    pub fn max_preferred_vector_width_float(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?.to_uint())
    }

    pub fn max_preferred_vector_width_double(&self) -> Result<cl_uint, cl_int> {
        Ok(
            get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)?
                .to_uint(),
        )
    }

    pub fn max_clock_frequency(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_CLOCK_FREQUENCY)?.to_uint())
    }

    pub fn address_bits(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_ADDRESS_BITS)?.to_uint())
    }

    pub fn max_read_image_args(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_READ_IMAGE_ARGS)?.to_uint())
    }

    pub fn max_write_image_args(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_WRITE_IMAGE_ARGS)?.to_uint())
    }

    pub fn max_mem_alloc_size(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_MEM_ALLOC_SIZE)?.to_ulong())
    }

    pub fn image2d_max_width(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE2D_MAX_WIDTH)?.to_size())
    }

    pub fn image2d_max_height(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE2D_MAX_HEIGHT)?.to_size())
    }

    pub fn image3d_max_width(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE3D_MAX_WIDTH)?.to_size())
    }

    pub fn image3d_max_height(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE3D_MAX_HEIGHT)?.to_size())
    }

    pub fn image3d_max_depth(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE3D_MAX_DEPTH)?.to_size())
    }

    pub fn image_support(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE_SUPPORT)?.to_uint())
    }

    pub fn max_parameter_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_PARAMETER_SIZE)?.to_size())
    }

    pub fn max_device_samples(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_SAMPLERS)?.to_uint())
    }

    pub fn mem_base_addr_align(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MEM_BASE_ADDR_ALIGN)?.to_uint())
    }

    pub fn min_data_type_align_size(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE)?.to_uint())
    }

    pub fn single_fp_config(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_SINGLE_FP_CONFIG)?.to_ulong())
    }

    pub fn global_mem_cache_type(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_GLOBAL_MEM_CACHE_TYPE)?.to_uint())
    }

    pub fn global_mem_cacheline_size(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE)?.to_uint())
    }

    pub fn global_mem_cache_size(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)?.to_ulong())
    }

    pub fn global_mem_size(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_GLOBAL_MEM_SIZE)?.to_ulong())
    }

    pub fn max_constant_buffer_size(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)?.to_ulong())
    }

    pub fn max_constant_args(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_CONSTANT_ARGS)?.to_uint())
    }

    pub fn local_mem_type(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_LOCAL_MEM_TYPE)?.to_uint())
    }

    pub fn local_mem_size(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_LOCAL_MEM_SIZE)?.to_ulong())
    }

    pub fn error_correction_support(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_ERROR_CORRECTION_SUPPORT)?.to_uint())
    }

    pub fn profiling_timer_resolution(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PROFILING_TIMER_RESOLUTION)?.to_size())
    }

    pub fn endian_little(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_ENDIAN_LITTLE)?.to_uint())
    }

    pub fn available(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_AVAILABLE)?.to_uint())
    }

    pub fn compiler_available(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_COMPILER_AVAILABLE)?.to_uint())
    }

    pub fn execution_capabilities(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_EXECUTION_CAPABILITIES)?.to_ulong())
    }

    pub fn queue_on_host_properties(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_QUEUE_ON_HOST_PROPERTIES)?.to_ulong())
    }

    pub fn name(&self) -> Result<CString, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NAME)?
            .to_str()
            .unwrap())
    }

    pub fn vendor(&self) -> Result<CString, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_VENDOR)?
            .to_str()
            .unwrap())
    }

    pub fn driver_version(&self) -> Result<CString, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DRIVER_VERSION)?
            .to_str()
            .unwrap())
    }

    pub fn profile(&self) -> Result<CString, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PROFILE)?
            .to_str()
            .unwrap())
    }

    pub fn version(&self) -> Result<CString, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_VERSION)?
            .to_str()
            .unwrap())
    }

    pub fn extensions(&self) -> Result<CString, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_EXTENSIONS)?
            .to_str()
            .unwrap())
    }

    pub fn platform(&self) -> Result<intptr_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PLATFORM)?.to_ptr())
    }

    pub fn double_fp_config(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_DOUBLE_FP_CONFIG)?.to_ulong())
    }

    pub fn half_fp_config(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_HALF_FP_CONFIG)?.to_ulong())
    }

    pub fn preferred_vector_width_half(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF)?.to_uint())
    }

    // DEPRECATED 2.0
    pub fn host_unified_memory(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_HOST_UNIFIED_MEMORY)?.to_uint())
    }

    pub fn native_vector_width_char(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR)?.to_uint())
    }

    pub fn native_vector_width_short(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT)?.to_uint())
    }

    pub fn native_vector_width_int(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_INT)?.to_uint())
    }

    pub fn native_vector_width_long(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG)?.to_uint())
    }

    pub fn native_vector_width_float(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT)?.to_uint())
    }

    pub fn native_vector_width_double(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE)?.to_uint())
    }

    pub fn native_vector_width_half(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF)?.to_uint())
    }

    pub fn opencl_c_version(&self) -> Result<CString, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_OPENCL_C_VERSION)?
            .to_str()
            .unwrap())
    }

    pub fn linker_available(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_LINKER_AVAILABLE)?.to_uint())
    }

    pub fn built_in_kernels(&self) -> Result<CString, cl_int> {
        Ok(
            get_device_info(self.id, DeviceInfo::CL_DEVICE_BUILT_IN_KERNELS)?
                .to_str()
                .unwrap(),
        )
    }

    pub fn image_max_buffer_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE_MAX_BUFFER_SIZE)?.to_size())
    }

    pub fn image_max_array_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE_MAX_ARRAY_SIZE)?.to_size())
    }

    pub fn parent_device(&self) -> Result<intptr_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PARENT_DEVICE)?.to_ptr())
    }

    pub fn partition_max_sub_devices(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PARTITION_MAX_SUB_DEVICES)?.to_uint())
    }
    
    pub fn partition_properties(&self) -> Result<Vec<intptr_t>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PARTITION_PROPERTIES)?.to_vec_intptr())
    }

    pub fn partition_affinity_domain(&self) -> Result<Vec<cl_ulong>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PARTITION_AFFINITY_DOMAIN)?.to_vec_ulong())
    }

    pub fn partition_type(&self) -> Result<Vec<intptr_t>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PARTITION_TYPE)?.to_vec_intptr())
    }

    pub fn reference_count(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_REFERENCE_COUNT)?.to_uint())
    }

    pub fn preferred_interop_user_sync(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_INTEROP_USER_SYNC)?.to_uint())
    }

    pub fn printf_buffer_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PRINTF_BUFFER_SIZE)?.to_size())
    }

    // CL_VERSION_2_0
    pub fn image_pitch_alignment(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE_PITCH_ALIGNMENT)?.to_uint())
    }

    pub fn image_base_address_alignment(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT)?.to_uint())
    }

    pub fn max_read_write_image_args(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS)?.to_uint())
    }

    pub fn max_global_variable_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE)?.to_size())
    }

    pub fn queue_on_device_properties(&self) -> Result<Vec<intptr_t>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES)?.to_vec_intptr())
    }

    pub fn queue_on_device_preferred_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE)?.to_size())
    }

    pub fn queue_on_device_max_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE)?.to_size())
    }

    pub fn max_on_device_queues(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_ON_DEVICE_QUEUES)?.to_uint())
    }

    pub fn max_on_device_events(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_ON_DEVICE_EVENTS)?.to_uint())
    }

    pub fn svm_capabilities(&self) -> Result<cl_device_svm_capabilities, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_SVM_CAPABILITIES)?.to_ulong())
    }

    pub fn global_variable_preferred_total_size(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE)?.to_size())
    }

    pub fn max_pipe_args(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_PIPE_ARGS)?.to_uint())
    }

    pub fn pipe_max_active_reservations(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS)?.to_uint())
    }

    pub fn pipe_max_packet_size(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PIPE_MAX_PACKET_SIZE)?.to_uint())
    }

    pub fn preferred_platform_atomic_alignment(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT)?.to_uint())
    }

    pub fn preferred_global_atomic_alignment(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT)?.to_uint())
    }

    pub fn preferred_local_atomic_alignment(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT)?.to_uint())
    }

    // CL_VERSION_2_1
    pub fn il_version(&self) -> Result<CString, cl_int> {
        Ok(
            get_device_info(self.id, DeviceInfo::CL_DEVICE_IL_VERSION)?
                .to_str()
                .unwrap(),
        )
    }

    pub fn max_num_sub_groups(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_MAX_NUM_SUB_GROUPS)?.to_uint())
    }

    pub fn sub_group_independent_forward_progress(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS)?.to_uint())
    }

    // CL_VERSION_3_0
    pub fn numeric_version(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NUMERIC_VERSION)?.to_uint())
    }

    pub fn extensions_with_version(&self) -> Result<Vec<cl_name_version>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_EXTENSIONS_WITH_VERSION)?.to_vec_name_version())
    }

    pub fn ils_with_version(&self) -> Result<Vec<cl_name_version>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_ILS_WITH_VERSION)?.to_vec_name_version())
    }

    pub fn built_in_kernels_with_version(&self) -> Result<Vec<cl_name_version>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION)?.to_vec_name_version())
    }

    pub fn atomic_memory_capabilities(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES)?.to_ulong())
    }

    pub fn atomic_fence_capabilities(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_ATOMIC_FENCE_CAPABILITIES)?.to_ulong())
    }

    pub fn non_uniform_work_group_support(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT)?.to_uint())
    }

    pub fn opencl_c_all_versions(&self) -> Result<Vec<cl_name_version>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_OPENCL_C_ALL_VERSIONS)?.to_vec_name_version())
    }

    pub fn preferred_work_group_size_multiple(&self) -> Result<size_t, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)?.to_size())
    }

    pub fn work_group_collective_functions_support(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT)?.to_uint())
    }

    pub fn generic_address_space_support(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT)?.to_uint())
    }

    pub fn opencl_c_features(&self) -> Result<Vec<cl_name_version>, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_OPENCL_C_FEATURES)?.to_vec_name_version())
    }

    pub fn device_enqueue_capabilities(&self) -> Result<cl_ulong, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES)?.to_ulong())
    }

    pub fn pipe_support(&self) -> Result<cl_uint, cl_int> {
        Ok(get_device_info(self.id, DeviceInfo::CL_DEVICE_PIPE_SUPPORT)?.to_uint())
    }

    pub fn latest_conformance_version_passed(&self) -> Result<CString, cl_int> {
        Ok(
            get_device_info(self.id, DeviceInfo::CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED)?
                .to_str()
                .unwrap(),
        )
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

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::cl_platform_id;
    use crate::platform::get_platforms;
    use crate::error_codes::error_text;

    #[test]
    fn test_get_devices() {
        let platforms = get_platforms().unwrap();
        println!("Number of platforms: {}", platforms.len());
        assert!(0 < platforms.len());

        for platform in platforms {
            println!("CL_PLATFORM_NAME: {:?}", platform.name().unwrap());

            let devices = platform.get_devices(CL_DEVICE_TYPE_ALL).unwrap();
            for device_id in devices {
                let device = Device::new(device_id);

                println!("\tCL_DEVICE_NAME: {:?}", device.name().unwrap());
                println!("\tCL_DEVICE_TYPE: {:X}", device.dev_type().unwrap());
                println!("\tCL_DEVICE_VENDOR_ID: {:X}", device.vendor_id().unwrap());
                println!("\tCL_DEVICE_VENDOR: {:?}", device.vendor().unwrap());
                println!(
                    "\tCL_DEVICE_OPENCL_C_VERSION: {:?}",
                    device.opencl_c_version().unwrap()
                );
                println!("");
            }
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
            Err(e) => println!("OpenCL error, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES: {}", error_text(e))
        };

        let value = device.name().unwrap();
        println!("CL_DEVICE_NAME: {:?}", value);
        assert!(!value.to_bytes().is_empty());

        let value = device.vendor().unwrap();
        println!("CL_DEVICE_VENDOR: {:?}", value);
        assert!(!value.to_bytes().is_empty());

        let value = device.driver_version().unwrap();
        println!("CL_DRIVER_VERSION: {:?}", value);
        assert!(!value.to_bytes().is_empty());

        let value = device.profile().unwrap();
        println!("CL_DEVICE_PROFILE: {:?}", value);
        assert!(!value.to_bytes().is_empty());

        let value = device.version().unwrap();
        println!("CL_DEVICE_VERSION: {:?}", value);
        assert!(!value.to_bytes().is_empty());

        let value = device.extensions().unwrap();
        println!("CL_DEVICE_EXTENSIONS: {:?}", value);
        assert!(!value.to_bytes().is_empty());

        let value = device.platform().unwrap();
        println!("CL_DEVICE_PLATFORM: {:X}", value);
        assert_eq!(platform.id(), value as cl_platform_id);

        // Device may not support double fp precision
        match device.double_fp_config() {
            Ok(value) => {
                println!("CL_DEVICE_DOUBLE_FP_CONFIG: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_DOUBLE_FP_CONFIG: {}", error_text(e))
        };

        // Device may not support half fp precision
        match device.half_fp_config() {
            Ok(value) => {
                println!("CL_DEVICE_HALF_FP_CONFIG: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_HALF_FP_CONFIG: {}", error_text(e))
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
        println!("CL_DEVICE_OPENCL_C_VERSION: {:?}", value);
        assert!(!value.to_bytes().is_empty());

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
        println!("CL_DEVICE_PARENT_DEVICE: {:X}", value);
        
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
            Err(e) => println!("OpenCL error, CL_DEVICE_IMAGE_PITCH_ALIGNMENT: {}", error_text(e))
        };

        match device.image_base_address_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT: {}", error_text(e))
        };

        match device.max_read_write_image_args() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: {}", error_text(e))
        };

        match device.max_global_variable_size() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: {}", error_text(e))
        };

        match device.queue_on_device_properties() {
            Ok(value) => {
                println!("CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: {}", error_text(e))
        };

        match device.queue_on_device_preferred_size() {
            Ok(value) => {
                println!("CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: {}", error_text(e))
        };

        match device.queue_on_device_max_size() {
            Ok(value) => {
                println!("CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: {}", error_text(e))
        };

        match device.max_on_device_queues() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_ON_DEVICE_QUEUES: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_MAX_ON_DEVICE_QUEUES: {}", error_text(e))
        };

        match device.max_on_device_events() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_ON_DEVICE_EVENTS: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_MAX_ON_DEVICE_EVENTS: {}", error_text(e))
        };

        match device.svm_capabilities() {
            Ok(value) => {
                println!("CL_DEVICE_SVM_CAPABILITIES: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_SVM_CAPABILITIES: {}", error_text(e))
        };

        match device.global_variable_preferred_total_size() {
            Ok(value) => {
                println!("CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: {}", error_text(e))
        };

        match device.max_pipe_args() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_PIPE_ARGS: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_MAX_PIPE_ARGS: {}", error_text(e))
        };
        
        match device.pipe_max_active_reservations() {
            Ok(value) => {
                println!("CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS: {}", error_text(e))
        };

        match device.pipe_max_packet_size() {
            Ok(value) => {
                println!("CL_DEVICE_PIPE_MAX_PACKET_SIZE: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PIPE_MAX_PACKET_SIZE: {}", error_text(e))
        };

        match device.preferred_platform_atomic_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT: {}", error_text(e))
        };

        match device.preferred_global_atomic_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT: {}", error_text(e))
        };

        match device.preferred_local_atomic_alignment() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT: {}", error_text(e))
        };

        //////////////////////////////////////////////////////////////////////
        // CL_VERSION_2_1 parameters

        match device.il_version() {
            Ok(value) => {
                println!("CL_DEVICE_IL_VERSION: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_IL_VERSION: {}", error_text(e))
        };

        match device.max_num_sub_groups() {
            Ok(value) => {
                println!("CL_DEVICE_MAX_NUM_SUB_GROUPS: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_MAX_NUM_SUB_GROUPS: {}", error_text(e))
        };

        match device.sub_group_independent_forward_progress() {
            Ok(value) => {
                println!("CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {}", error_text(e))
        };

        //////////////////////////////////////////////////////////////////////
        // CL_VERSION_3_0 parameters

        match device.numeric_version() {
            Ok(value) => {
                println!("CL_DEVICE_NUMERIC_VERSION: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_NUMERIC_VERSION: {}", error_text(e))
        };

        match device.extensions_with_version() {
            Ok(value) => {
                println!("CL_DEVICE_EXTENSIONS_WITH_VERSION: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_EXTENSIONS_WITH_VERSION: {}", error_text(e))
        };

        match device.ils_with_version() {
            Ok(value) => {
                println!("CL_DEVICE_ILS_WITH_VERSION: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_ILS_WITH_VERSION: {}", error_text(e))
        };

        match device.built_in_kernels_with_version() {
            Ok(value) => {
                println!("CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION: {}", error_text(e))
        };

        match device.atomic_memory_capabilities() {
            Ok(value) => {
                println!("CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: {}", error_text(e))
        };

        match device.atomic_fence_capabilities() {
            Ok(value) => {
                println!("CL_DEVICE_ATOMIC_FENCE_CAPABILITIES: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES: {}", error_text(e))
        };

        match device.non_uniform_work_group_support() {
            Ok(value) => {
                println!("CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT: {}", error_text(e))
        };

        match device.opencl_c_all_versions() {
            Ok(value) => {
                println!("CL_DEVICE_OPENCL_C_ALL_VERSIONS: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_OPENCL_C_ALL_VERSIONS: {}", error_text(e))
        };

        match device.preferred_work_group_size_multiple() {
            Ok(value) => {
                println!("CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {}", error_text(e))
        };
        
        match device.work_group_collective_functions_support() {
            Ok(value) => {
                println!("CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: {}", error_text(e))
        };

        match device.generic_address_space_support() {
            Ok(value) => {
                println!("CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT: {}", error_text(e))
        };

        match device.opencl_c_features() {
            Ok(value) => {
                println!("CL_DEVICE_OPENCL_C_FEATURES: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_OPENCL_C_FEATURES: {}", error_text(e))
        };

        match device.device_enqueue_capabilities() {
            Ok(value) => {
                println!("CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES: {}", error_text(e))
        };

        match device.pipe_support() {
            Ok(value) => {
                println!("CL_DEVICE_PIPE_SUPPORT: {:X}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_PIPE_SUPPORT: {}", error_text(e))
        };

        match device.latest_conformance_version_passed() {
            Ok(value) => {
                println!("CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED: {:?}", value)
            }
            Err(e) => println!("OpenCL error, CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED: {}", error_text(e))
        };
    }
}
