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

use super::context::Context;
use super::Result;

use cl3::device::{
    CL_DEVICE_SVM_ATOMICS, CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
    CL_DEVICE_SVM_FINE_GRAIN_SYSTEM,
};
use cl3::memory::{
    svm_alloc, svm_free, CL_MEM_READ_WRITE, CL_MEM_SVM_ATOMICS, CL_MEM_SVM_FINE_GRAIN_BUFFER,
};
use cl3::types::{cl_device_svm_capabilities, cl_svm_mem_flags, cl_uint};
use libc::c_void;
#[cfg(feature = "serde")]
use serde::de::{Deserialize, DeserializeSeed, Deserializer, Error, SeqAccess, Visitor};
#[cfg(feature = "serde")]
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::alloc::{self, Layout};
use std::fmt;
use std::fmt::Debug;
use std::iter::IntoIterator;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;
#[allow(unused_imports)]
use std::result;

struct SvmRawVec<'a, T> {
    ptr: *mut T,
    cap: usize,
    context: &'a Context,
    fine_grain_buffer: bool,
    fine_grain_system: bool,
    atomics: bool,
}

unsafe impl<'a, T: Send> Send for SvmRawVec<'a, T> {}
unsafe impl<'a, T: Sync> Sync for SvmRawVec<'a, T> {}

impl<'a, T> SvmRawVec<'a, T> {
    fn new(context: &'a Context, svm_capabilities: cl_device_svm_capabilities) -> Self {
        assert!(0 < mem::size_of::<T>(), "No Zero Sized Types!");

        assert!(
            0 != svm_capabilities
                & (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER),
            "No OpenCL SVM, use OpenCL buffers"
        );

        let fine_grain_buffer: bool = svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0;
        let fine_grain_system: bool = svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM != 0;
        let atomics: bool = (fine_grain_buffer || fine_grain_system)
            && (svm_capabilities & CL_DEVICE_SVM_ATOMICS != 0);
        SvmRawVec {
            ptr: ptr::null_mut(),
            cap: 0,
            context,
            fine_grain_buffer,
            fine_grain_system,
            atomics,
        }
    }

    fn with_capacity(
        context: &'a Context,
        svm_capabilities: cl_device_svm_capabilities,
        capacity: usize,
    ) -> Result<Self> {
        let mut v = Self::new(context, svm_capabilities);
        v.grow(capacity)?;

        Ok(v)
    }

    fn with_capacity_zeroed(
        context: &'a Context,
        svm_capabilities: cl_device_svm_capabilities,
        capacity: usize,
    ) -> Result<Self> {
        let mut v = Self::with_capacity(context, svm_capabilities, capacity)?;
        v.zero(capacity);

        Ok(v)
    }

    fn grow(&mut self, count: usize) -> Result<()> {
        let elem_size = mem::size_of::<T>();

        let mut new_cap = count;
        // if pushing or inserting, double the capacity
        if (0 < self.cap) && (count - self.cap == 1) {
            new_cap = 2 * self.cap;
        }

        let size = elem_size * new_cap;

        // Ensure within capacity.
        assert!(size <= (isize::MAX as usize) / 2, "capacity overflow");

        // allocation, determine whether to use svm_alloc or not
        let ptr = if self.fine_grain_system {
            let new_layout = Layout::array::<T>(new_cap).unwrap();
            let new_ptr = unsafe { alloc::alloc(new_layout) as *mut c_void };
            if new_ptr.is_null() {
                alloc::handle_alloc_error(new_layout);
            }
            new_ptr
        } else {
            let svm_mem_flags: cl_svm_mem_flags = if self.fine_grain_buffer {
                if self.atomics {
                    CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_READ_WRITE | CL_MEM_SVM_ATOMICS
                } else {
                    CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_READ_WRITE
                }
            } else {
                CL_MEM_READ_WRITE
            };
            let alignment = mem::align_of::<T>();
            unsafe {
                svm_alloc(
                    self.context.get(),
                    svm_mem_flags,
                    size,
                    alignment as cl_uint,
                )?
            }
        };

        // reallocation, copy old data to new pointer and free old memory
        if 0 < self.cap {
            unsafe { ptr::copy(self.ptr, ptr as *mut T, self.cap) };
            if self.fine_grain_system {
                let layout = Layout::array::<T>(self.cap).unwrap();
                unsafe {
                    alloc::dealloc(self.ptr as *mut u8, layout);
                }
            } else {
                unsafe { svm_free(self.context.get(), self.ptr as *mut c_void) };
            }
        }

        self.ptr = ptr as *mut T;
        self.cap = new_cap;

        Ok(())
    }

    fn zero(&mut self, count: usize) {
        unsafe { ptr::write_bytes(self.ptr, 0u8, count) };
    }
}

impl<'a, T> Drop for SvmRawVec<'a, T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if self.fine_grain_system {
                let layout = Layout::array::<T>(self.cap).unwrap();
                unsafe {
                    alloc::dealloc(self.ptr as *mut u8, layout);
                }
            } else {
                unsafe { svm_free(self.context.get(), self.ptr as *mut c_void) };
            }
            self.ptr = ptr::null_mut();
        }
    }
}

/// An OpenCL Shared Virtual Memory (SVM) vector.
/// It has the lifetime of the [Context] that it was constructed from.  
/// Note: T cannot be a "zero sized type" (ZST).
///
/// There are three types of Shared Virtual Memory:
/// - CL_DEVICE_SVM_COARSE_GRAIN_BUFFER: OpenCL buffer memory objects can be shared.
/// - CL_DEVICE_SVM_FINE_GRAIN_BUFFER: individual memory objects in an OpenCL buffer can be shared.
/// - CL_DEVICE_SVM_FINE_GRAIN_SYSTEM: individual memory objects *anywhere* in **host** memory can be shared.
///
/// This `SvmVec` struct is designed to support CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
/// and CL_DEVICE_SVM_FINE_GRAIN_BUFFER.  
/// A [Context] that supports CL_DEVICE_SVM_FINE_GRAIN_SYSTEM can (and should!)
/// use a standard Rust vector instead.
///
/// Intel provided an excellent overview of Shared Virtual Memory here:
/// [OpenCL 2.0 Shared Virtual Memory Overview](https://software.intel.com/content/www/us/en/develop/articles/opencl-20-shared-virtual-memory-overview.html).  
/// A PDF version is available here: [SVM Overview](https://github.com/kenba/opencl3/blob/main/docs/svmoverview.pdf).
///
/// To summarise, a CL_DEVICE_SVM_COARSE_GRAIN_BUFFER requires the SVM to be *mapped*
/// before being read or written by the host and *unmapped* afterward, while
/// CL_DEVICE_SVM_FINE_GRAIN_BUFFER can be used like a standard Rust vector.
///
/// The `is_fine_grained method` can be used to determine whether an `SvmVec` supports
/// CL_DEVICE_SVM_FINE_GRAIN_BUFFER and should be used to control SVM map and unmap
/// operations, e.g.:
/// ```no_run
/// # use cl3::device::CL_DEVICE_TYPE_GPU;
/// # use opencl3::command_queue::CommandQueue;
/// # use opencl3::context::Context;
/// # use opencl3::device::Device;
/// # use opencl3::kernel::{ExecuteKernel, Kernel};
/// # use opencl3::memory::{CL_MAP_WRITE};
/// # use opencl3::platform::get_platforms;
/// # use opencl3::svm::SvmVec;
/// # use opencl3::types::*;
/// # use opencl3::Result;
///
/// # fn main() -> Result<()> {
/// # let platforms = get_platforms().unwrap();
/// # let devices = platforms[0].get_devices(CL_DEVICE_TYPE_GPU).unwrap();
/// # let device = Device::new(devices[0]);
/// # let context = Context::from_device(&device).unwrap();
/// # let queue = CommandQueue::create_default_with_properties(&context, 0, 0).unwrap();
/// // The input data
/// const ARRAY_SIZE: usize = 8;
/// let value_array: [cl_int; ARRAY_SIZE] = [3, 2, 5, 9, 7, 1, 4, 2];
///
/// // Create an OpenCL SVM vector
/// let mut test_values = SvmVec::<cl_int>::allocate(&context, ARRAY_SIZE)?;
///
/// // Map test_values if not an CL_MEM_SVM_FINE_GRAIN_BUFFER
/// if !test_values.is_fine_grained() {
///      unsafe { queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut test_values, &[])?};
/// }
///
/// // Copy input data into the OpenCL SVM vector
/// test_values.clone_from_slice(&value_array);
///
/// // Unmap test_values if not an CL_MEM_SVM_FINE_GRAIN_BUFFER
/// if !test_values.is_fine_grained() {
///     let unmap_test_values_event =  unsafe { queue.enqueue_svm_unmap(&test_values, &[])?};
///     unmap_test_values_event.wait()?;
/// }
/// # Ok(())
/// # }
/// ```

pub struct SvmVec<'a, T> {
    buf: SvmRawVec<'a, T>,
    len: usize,
}

impl<'a, T> SvmVec<'a, T> {
    fn ptr(&self) -> *mut T {
        self.buf.ptr
    }

    /// The capacity of the vector.
    pub fn cap(&self) -> usize {
        self.buf.cap
    }

    /// The length of the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the vector is fine grain buffer
    pub fn is_fine_grain_buffer(&self) -> bool {
        self.buf.fine_grain_buffer
    }

    /// Whether the vector is fine grain system
    pub fn is_fine_grain_system(&self) -> bool {
        self.buf.fine_grain_system
    }

    /// Whether the vector is fine grained
    pub fn is_fine_grained(&self) -> bool {
        self.buf.fine_grain_buffer || self.buf.fine_grain_system
    }

    /// Whether the vector can use atomics
    pub fn has_atomics(&self) -> bool {
        self.buf.atomics
    }

    /// Clear the vector, i.e. empty it.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Set the length of the vector.
    /// If new_len > len, the new memory will be uninitialised.
    ///
    /// # Safety
    /// May fail to grow buf if memory is not available for new_len.
    pub fn set_len(&mut self, new_len: usize) -> Result<()> {
        if self.cap() < new_len {
            self.buf.grow(new_len)?;
        }
        self.len = new_len;
        Ok(())
    }

    /// Construct an empty SvmVec from a [Context].  
    /// The SvmVec has the lifetime of the [Context].
    ///
    /// # Panics
    ///
    /// The cl_device_svm_capabilities of the [Context] must include
    /// CL_DEVICE_SVM_COARSE_GRAIN_BUFFER or CL_DEVICE_SVM_FINE_GRAIN_BUFFER.  
    /// The cl_device_svm_capabilities must *not* include CL_DEVICE_SVM_FINE_GRAIN_SYSTEM,
    /// a standard Rust `Vec!` should be used instead.
    pub fn new(context: &'a Context) -> Self {
        let svm_capabilities = context.get_svm_mem_capability();
        SvmVec {
            buf: SvmRawVec::new(context, svm_capabilities),
            len: 0,
        }
    }

    /// Construct an SvmVec with the given len of values from a [Context].
    ///
    /// returns a Result containing an SvmVec with len values of **uninitialised**
    /// memory, or the OpenCL error.
    ////
    /// # Panics
    ///
    /// The cl_device_svm_capabilities of the [Context] must include
    /// CL_DEVICE_SVM_COARSE_GRAIN_BUFFER or CL_DEVICE_SVM_FINE_GRAIN_BUFFER.  
    /// The cl_device_svm_capabilities must *not* include CL_DEVICE_SVM_FINE_GRAIN_SYSTEM,
    /// a standard Rust `Vec!` should be used instead.
    pub fn allocate(context: &'a Context, len: usize) -> Result<Self> {
        let svm_capabilities = context.get_svm_mem_capability();
        Ok(SvmVec {
            buf: SvmRawVec::with_capacity(context, svm_capabilities, len)?,
            len,
        })
    }

    /// Construct an empty SvmVec with the given capacity from a [Context].
    ///
    /// returns a Result containing an empty SvmVec, or the OpenCL error.
    ///
    /// # Panics
    ///
    /// The cl_device_svm_capabilities of the [Context] must include
    /// CL_DEVICE_SVM_COARSE_GRAIN_BUFFER or CL_DEVICE_SVM_FINE_GRAIN_BUFFER.  
    /// The cl_device_svm_capabilities must *not* include CL_DEVICE_SVM_FINE_GRAIN_SYSTEM,
    /// a standard Rust `Vec!` should be used instead.
    pub fn with_capacity(context: &'a Context, capacity: usize) -> Result<Self> {
        let svm_capabilities = context.get_svm_mem_capability();
        Ok(SvmVec {
            buf: SvmRawVec::with_capacity(context, svm_capabilities, capacity)?,
            len: 0,
        })
    }

    /// Construct an SvmVec with the given len of values from a [Context] and
    /// the svm_capabilities of the device (or devices) in the [Context].
    ///
    /// # Panics
    ///
    /// The function will panic if the cl_device_svm_capabilities of the [Context]
    /// does **not** include CL_DEVICE_SVM_FINE_GRAIN_BUFFER.
    ///
    /// returns a Result containing an SvmVec with len values of zeroed
    /// memory, or the OpenCL error.
    pub fn allocate_zeroed(context: &'a Context, len: usize) -> Result<Self> {
        let svm_capabilities = context.get_svm_mem_capability();
        let fine_grain_buffer: bool = svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0;
        assert!(
            fine_grain_buffer,
            "SVM is not fine grained, use `allocate` instead."
        );
        Ok(SvmVec {
            buf: SvmRawVec::with_capacity_zeroed(context, svm_capabilities, len)?,
            len,
        })
    }

    /// Reserve vector capacity.  
    /// returns an empty Result or the OpenCL error.
    pub fn reserve(&mut self, capacity: usize) -> Result<()> {
        self.buf.grow(capacity)
    }

    /// Push a value onto the vector.
    ///
    /// # Panics
    ///
    /// The function will panic if a coarse grain buffer attempts to grow the vector.
    pub fn push(&mut self, elem: T) {
        if self.len == self.cap() {
            assert!(
                self.is_fine_grained(),
                "SVM is not fine grained, cannot grow the vector."
            );
            self.buf.grow(self.len + 1).unwrap();
        }

        unsafe {
            ptr::write(self.ptr().add(self.len), elem);
        }

        // Can't fail, we'll OOM first.
        self.len += 1;
    }

    /// Pop a value from the vector.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(ptr::read(self.ptr().add(self.len))) }
        }
    }

    /// Insert a value into the vector at index.
    ///
    /// # Panics
    ///
    /// The function will panic if the index is out of bounds or
    /// if a coarse grain buffer attempts to grow the vector.
    pub fn insert(&mut self, index: usize, elem: T) {
        assert!(index <= self.len, "index out of bounds");
        if self.cap() == self.len {
            assert!(
                self.is_fine_grained(),
                "SVM is not fine grained, cannot grow the vector."
            );
            self.buf.grow(self.len + 1).unwrap();
        }

        unsafe {
            if index < self.len {
                ptr::copy(
                    self.ptr().add(index),
                    self.ptr().add(index + 1),
                    self.len - index,
                );
            }
            ptr::write(self.ptr().add(index), elem);
            self.len += 1;
        }
    }

    /// Remove a value from the vector at index.
    ///
    /// # Panics
    ///
    /// The function will panic if the index is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "index out of bounds");
        unsafe {
            self.len -= 1;
            let result = ptr::read(self.ptr().add(index));
            ptr::copy(
                self.ptr().add(index + 1),
                self.ptr().add(index),
                self.len - index,
            );
            result
        }
    }

    /// Drain the vector.
    pub fn drain(&mut self) -> Drain<T> {
        unsafe {
            let iter = RawValIter::new(self);

            // this is a mem::forget safety thing. If Drain is forgotten, we just
            // leak the whole Vec's contents. Also we need to do this *eventually*
            // anyway, so why not do it now?
            self.len = 0;

            Drain {
                iter,
                vec: PhantomData,
            }
        }
    }
}

impl<'a, T> IntoIterator for SvmVec<'a, T> {
    type Item = T;
    type IntoIter = IntoIter<'a, Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let iter = RawValIter::new(&self);
            let buf = ptr::read(&self.buf);
            mem::forget(self);

            Self::IntoIter { iter, _buf: buf }
        }
    }
}

impl<'a, T> Drop for SvmVec<'a, T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
        // allocation is handled by SvmRawVec
    }
}

impl<'a, T> Deref for SvmVec<'a, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len) }
    }
}

impl<'a, T> DerefMut for SvmVec<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), self.len) }
    }
}

impl<'a, T: Debug> fmt::Debug for SvmVec<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

/// A DeserializeSeed implementation  that uses stateful deserialization to
/// append array elements onto the end of an existing SvmVec.
/// The pre-existing state ("seed") in this case is the SvmVec<'b, T>.
#[cfg(feature = "serde")]
pub struct ExtendSvmVec<'a, 'b, T: 'a>(pub &'a mut SvmVec<'b, T>);

#[cfg(feature = "serde")]
impl<'de, 'a, 'b, T> DeserializeSeed<'de> for ExtendSvmVec<'a, 'b, T>
where
    T: Deserialize<'de>,
{
    // The return type of the `deserialize` method. Since this implementation
    // appends onto an existing SvmVec the return type is ().
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Visitor implementation to walk an array of the deserializer input.
        struct ExtendSvmVecVisitor<'a, 'b, T: 'a>(&'a mut SvmVec<'b, T>);

        impl<'de, 'a, 'b, T> Visitor<'de> for ExtendSvmVecVisitor<'a, 'b, T>
        where
            T: Deserialize<'de>,
        {
            type Value = ();

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an array")
            }

            fn visit_seq<A>(self, mut seq: A) -> result::Result<(), A::Error>
            where
                A: SeqAccess<'de>,
            {
                // reserve SvmVec memory if the size of the deserializer array is known
                if let Some(size) = seq.size_hint() {
                    let len = self.0.len + size;
                    self.0.reserve(len).map_err(A::Error::custom)?;
                }

                // Visit each element in the array and push it onto the existing SvmVec
                while let Some(elem) = seq.next_element()? {
                    self.0.push(elem);
                }
                Ok(())
            }
        }

        deserializer.deserialize_seq(ExtendSvmVecVisitor(self.0))
    }
}

#[cfg(feature = "serde")]
impl<'a, T> Serialize for SvmVec<'a, T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;

        for element in self.iter() {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

struct RawValIter<T> {
    start: *const T,
    end: *const T,
}

unsafe impl<T: Send> Send for RawValIter<T> {}

impl<T> RawValIter<T> {
    unsafe fn new(slice: &[T]) -> Self {
        RawValIter {
            start: slice.as_ptr(),
            end: if mem::size_of::<T>() == 0 {
                ((slice.as_ptr() as usize) + slice.len()) as *const _
            } else if slice.is_empty() {
                slice.as_ptr()
            } else {
                slice.as_ptr().add(slice.len())
            },
        }
    }
}

impl<T> Iterator for RawValIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                let result = ptr::read(self.start);
                self.start = if mem::size_of::<T>() == 0 {
                    (self.start as usize + 1) as *const _
                } else {
                    self.start.offset(1)
                };
                Some(result)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let elem_size = mem::size_of::<T>();
        let len =
            (self.end as usize - self.start as usize) / if elem_size == 0 { 1 } else { elem_size };
        (len, Some(len))
    }
}

impl<T> DoubleEndedIterator for RawValIter<T> {
    fn next_back(&mut self) -> Option<T> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                self.end = if mem::size_of::<T>() == 0 {
                    (self.end as usize - 1) as *const _
                } else {
                    self.end.offset(-1)
                };
                Some(ptr::read(self.end))
            }
        }
    }
}

pub struct IntoIter<'a, T> {
    _buf: SvmRawVec<'a, T>, // we don't actually care about this. Just need it to live.
    iter: RawValIter<T>,
}

impl<'a, T> Iterator for IntoIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for IntoIter<'a, T> {
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back()
    }
}

impl<'a, T> Drop for IntoIter<'a, T> {
    fn drop(&mut self) {
        for _ in &mut *self {}
    }
}

pub struct Drain<'a, T: 'a> {
    vec: PhantomData<&'a mut SvmVec<'a, T>>,
    iter: RawValIter<T>,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back()
    }
}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // pre-drain the iter
        for _ in &mut self.iter {}
    }
}
