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

use super::context::Context;

use cl3::device::{
    CL_DEVICE_SVM_COARSE_GRAIN_BUFFER, CL_DEVICE_SVM_FINE_GRAIN_BUFFER,
    CL_DEVICE_SVM_FINE_GRAIN_SYSTEM,
};
use cl3::memory::{svm_alloc, svm_free, CL_MEM_READ_WRITE, CL_MEM_SVM_FINE_GRAIN_BUFFER};
use cl3::types::{cl_device_svm_capabilities, cl_svm_mem_flags, cl_uint};
use libc::c_void;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;

struct SvmRawVec<'a, T> {
    ptr: *mut T,
    cap: usize,
    context: &'a Context,
    fine_grain_buffer: bool,
}

impl<'a, T> SvmRawVec<'a, T> {
    fn new(context: &'a Context, svm_capabilities: cl_device_svm_capabilities) -> Self {
        assert!(0 < mem::size_of::<T>(), "No Zero Sized Types!");

        assert!(
            0 != svm_capabilities
                & (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER),
            "No OpenCL SVM, use OpenCL buffers"
        );

        let fine_grain_system: bool = svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM != 0;
        assert!(!fine_grain_system, "SVM supports system memory, use Vec!");

        let fine_grain_buffer: bool = svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0;
        SvmRawVec {
            ptr: ptr::null_mut(),
            cap: 0,
            context,
            fine_grain_buffer,
        }
    }

    fn grow(&mut self, count: usize) {
        let elem_size = mem::size_of::<T>();

        let mut new_cap = count;
        // if pushing or inserting, double the capacity
        if (0 < self.cap) && (count - self.cap == 1) {
            new_cap = 2 * self.cap;
        }

        // Ensure within capacity.
        assert!(new_cap <= (isize::MAX as usize) / 2, "capacity overflow");

        let svm_mem_flags: cl_svm_mem_flags = if true == self.fine_grain_buffer {
            CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_READ_WRITE
        } else {
            CL_MEM_READ_WRITE
        };
        let size = elem_size * count;
        let alignment = mem::align_of::<T>();
        let ptr = svm_alloc(
            self.context.get(),
            svm_mem_flags,
            size,
            alignment as cl_uint,
        )
        .unwrap();
        assert!(!ptr.is_null(), "svm_alloc failed");

        // reallocation, copy old data to new pointer and free old memory
        if 0 < self.cap {
            unsafe { ptr::copy(self.ptr, ptr as *mut T, self.cap) };
            svm_free(self.context.get(), self.ptr as *mut c_void);
        }

        self.ptr = ptr as *mut T;
        self.cap = new_cap;
    }
}

impl<'a, T> Drop for SvmRawVec<'a, T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            svm_free(self.context.get(), self.ptr as *mut c_void);
            self.ptr = ptr::null_mut();
            // println!("SvmRawVec::drop");
        }
    }
}

/// An OpenCL Shared Virtual Memory (SVM) vector.  
/// It has the lifetime of the [Context] that it was constructed from.  
/// Note: T cannot be a "zero sized type" (ZST).
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

    /// Construct an empty SvmVec from a [Context] and the svm_capabilities of
    /// the device (or devices) in the [Context].  
    /// The SvmVec has the lifetime of the [Context].
    pub fn new(context: &'a Context, svm_capabilities: cl_device_svm_capabilities) -> Self {
        SvmVec {
            buf: SvmRawVec::new(&context, svm_capabilities),
            len: 0,
        }
    }

    /// Reserve vector capacity.
    pub fn reserve(&mut self, capacity: usize) {
        self.buf.grow(capacity);
    }

    pub fn push(&mut self, elem: T) {
        if self.len == self.cap() {
            self.buf.grow(self.len + 1);
        }

        unsafe {
            ptr::write(self.ptr().offset(self.len as isize), elem);
        }

        // Can't fail, we'll OOM first.
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(ptr::read(self.ptr().offset(self.len as isize))) }
        }
    }

    pub fn insert(&mut self, index: usize, elem: T) {
        assert!(index <= self.len, "index out of bounds");
        if self.cap() == self.len {
            self.buf.grow(self.len + 1);
        }

        unsafe {
            if index < self.len {
                ptr::copy(
                    self.ptr().offset(index as isize),
                    self.ptr().offset(index as isize + 1),
                    self.len - index,
                );
            }
            ptr::write(self.ptr().offset(index as isize), elem);
            self.len += 1;
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len, "index out of bounds");
        unsafe {
            self.len -= 1;
            let result = ptr::read(self.ptr().offset(index as isize));
            ptr::copy(
                self.ptr().offset(index as isize + 1),
                self.ptr().offset(index as isize),
                self.len - index,
            );
            result
        }
    }

    pub fn into_iter(self) -> IntoIter<'a, T> {
        unsafe {
            let iter = RawValIter::new(&self);
            let buf = ptr::read(&self.buf);
            mem::forget(self);

            IntoIter {
                iter: iter,
                _buf: buf,
            }
        }
    }

    pub fn drain(&mut self) -> Drain<T> {
        unsafe {
            let iter = RawValIter::new(&self);

            // this is a mem::forget safety thing. If Drain is forgotten, we just
            // leak the whole Vec's contents. Also we need to do this *eventually*
            // anyway, so why not do it now?
            self.len = 0;

            Drain {
                iter: iter,
                vec: PhantomData,
            }
        }
    }
}

impl<'a, T> Drop for SvmVec<'a, T> {
    fn drop(&mut self) {
        while let Some(_) = self.pop() {}
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

struct RawValIter<T> {
    start: *const T,
    end: *const T,
}

impl<T> RawValIter<T> {
    unsafe fn new(slice: &[T]) -> Self {
        RawValIter {
            start: slice.as_ptr(),
            end: if mem::size_of::<T>() == 0 {
                ((slice.as_ptr() as usize) + slice.len()) as *const _
            } else if slice.len() == 0 {
                slice.as_ptr()
            } else {
                slice.as_ptr().offset(slice.len() as isize)
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
