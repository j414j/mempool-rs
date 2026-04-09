//! A fixed-capacity pool that returns movable slot handles.
//!
//! This is the loosest and most manual handle-based variant in the crate.
//! `MemPool<T>` owns a fixed number of slots and returns [`SlotHandle`] values
//! that may coexist and be moved around freely. Slots are returned to the pool
//! only when [`MemPool::free`] or [`MemPool::free_unchecked`] is called.
//!
//! # Design Choice
//!
//! [`SlotHandle`] is not lifetime-bound to the pool. That makes multiple live
//! handles easy to use, but it also means the compiler cannot prevent a handle
//! from outliving the [`MemPool`] that created it.
//!
//! Callers must therefore uphold these rules:
//! - a handle must not outlive its pool
//! - a handle must be freed at most once
//! - a handle must be freed back into the same pool
//!
//! Dropping a handle destroys the stored `T` if it was initialized, but it does
//! not return the slot to the freelist. Returning the slot is an explicit pool
//! operation.
//!
//! # Examples
//!
//! ```
//! use mempool::MemPool;
//!
//! let mut pool = MemPool::new(2);
//! let mut a = pool.alloc().unwrap();
//! let mut b = pool.alloc().unwrap();
//!
//! a.init(10);
//! b.init(20);
//! assert_eq!(*a.get(), 10);
//! assert_eq!(*b.get(), 20);
//!
//! pool.free(a);
//! pool.free(b);
//! assert!(pool.alloc().is_some());
//! ```
//!
use std::{cell::UnsafeCell, mem::MaybeUninit};

struct Slot<T> {
  elem: UnsafeCell<MaybeUninit<T>>,
  next: usize,
}

pub struct MemPool<T> {
  buf: Vec<Slot<T>>,
  head: usize,
}

impl<T> MemPool<T> {
  /// Creates a pool with space for at most `size` concurrently allocated slots.
  pub fn new(size: usize) -> Self {
    let mut buf = Vec::with_capacity(size);
    for i in 0..size {
      buf.push(Slot {
        elem: UnsafeCell::new(MaybeUninit::uninit()),
        next: i + 1,
      })
    }

    Self { buf, head: 0 }
  }

  /// Allocates one slot from the pool.
  ///
  /// The returned handle starts out uninitialized. Call [`SlotHandle::init`]
  /// before using [`SlotHandle::get`] or [`SlotHandle::get_mut`].
  ///
  /// Returns `None` if the pool is exhausted.
  pub fn alloc<'a>(&'a mut self) -> Option<SlotHandle<T>> {
    if self.head >= self.buf.len() {
      return None;
    }

    Some(unsafe { self.alloc_unchecked() })
  }

  /// Allocates one slot without checking whether the pool is exhausted.
  ///
  /// # Safety
  ///
  /// The caller must ensure that the pool still contains at least one free
  /// slot before calling this function.
  pub unsafe fn alloc_unchecked<'a>(&'a mut self) -> SlotHandle<T> {
    let idx = self.head;
    let slot = &self.buf[idx];
    self.head = slot.next;

    SlotHandle {
      idx,
      elem: slot.elem.get(),
      is_init: false,
    }
  }

  /// Frees a handle without validating that it belongs to this pool.
  ///
  /// # Safety
  ///
  /// The caller must ensure all of the following:
  /// - `handle` was returned by a previous call to [`MemPool::alloc`] on this
  ///   exact pool.
  /// - `handle` has not already been freed.
  /// - No references previously derived from the handle are used again after
  ///   this call.
  pub unsafe fn free_unchecked(&mut self, handle: SlotHandle<T>) {
    self.buf[handle.idx].next = self.head;
    self.head = handle.idx;
  }

  /// Frees a handle previously allocated from this pool.
  ///
  /// Panics if the handle does not belong to this pool.
  pub fn free<'a>(&'a mut self, handle: SlotHandle<T>) {
    if handle.idx >= self.buf.len() {
      panic!("Illegal free");
    }

    if handle.elem != self.buf[handle.idx].elem.get() {
      panic!("invalid handle");
    }

    unsafe {
      self.free_unchecked(handle);
    }
  }
}

pub struct SlotHandle<T> {
  idx: usize,
  elem: *mut MaybeUninit<T>,
  is_init: bool,
}

impl<T> SlotHandle<T> {
  /// Initializes the value stored in this slot.
  ///
  /// Panics if the slot has already been initialized.
  pub fn init(&mut self, value: T) {
    if self.is_init {
      panic!("cannot reinit value");
    }
    self.is_init = true;
    unsafe { &mut (*self.elem) }.write(value);
  }

  /// Returns an immutable reference to the initialized value.
  ///
  /// # Safety
  ///
  /// The caller must ensure the slot has already been initialized with
  /// [`SlotHandle::init`].
  pub unsafe fn get_unchecked(&self) -> &T {
    unsafe { &*((*self.elem).as_ptr()) }
  }

  /// Returns a mutable reference to the initialized value.
  ///
  /// # Safety
  ///
  /// The caller must ensure the slot has already been initialized with
  /// [`SlotHandle::init`].
  pub unsafe fn get_mut_unchecked(&mut self) -> &mut T {
    unsafe { &mut *((*self.elem).as_mut_ptr()) }
  }

  /// Returns an immutable reference to the value.
  ///
  /// Panics if the slot has not been initialized yet.
  pub fn get(&self) -> &T {
    if !self.is_init {
      panic!("tried to get uninit element");
    }
    unsafe { self.get_unchecked() }
  }

  /// Returns a mutable reference to the value.
  ///
  /// Panics if the slot has not been initialized yet.
  pub fn get_mut(&mut self) -> &mut T {
    if !self.is_init {
      panic!("tried to get uninit element");
    }
    unsafe { self.get_mut_unchecked() }
  }
}

impl<T> Drop for SlotHandle<T> {
  fn drop(&mut self) {
    if self.is_init {
      unsafe {
        (*self.elem).assume_init_drop();
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::MemPool;
  use std::sync::atomic::{AtomicUsize, Ordering};

  struct DropCounter<'a> {
    drops: &'a AtomicUsize,
  }

  impl Drop for DropCounter<'_> {
    fn drop(&mut self) {
      self.drops.fetch_add(1, Ordering::SeqCst);
    }
  }

  #[test]
  fn alloc_returns_none_when_exhausted() {
    let mut pool = MemPool::<u32>::new(2);
    assert!(pool.alloc().is_some());
    assert!(pool.alloc().is_some());
    assert!(pool.alloc().is_none());
  }

  #[test]
  fn free_recycles_slots() {
    let mut pool = MemPool::<u32>::new(1);
    let handle = pool.alloc().unwrap();
    pool.free(handle);
    assert!(pool.alloc().is_some());
  }

  #[test]
  fn get_and_get_mut_work_after_init() {
    let mut pool = MemPool::<u32>::new(1);
    let mut handle = pool.alloc().unwrap();
    handle.init(10);
    assert_eq!(*handle.get(), 10);
    *handle.get_mut() += 5;
    assert_eq!(*handle.get(), 15);
    pool.free(handle);
  }

  #[test]
  fn checked_free_rejects_foreign_handle() {
    let mut a = MemPool::<u32>::new(1);
    let mut b = MemPool::<u32>::new(1);
    let handle = a.alloc().unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      b.free(handle);
    }));

    assert!(result.is_err());
  }

  #[test]
  fn dropping_handle_drops_value_but_does_not_free_slot() {
    let drops = AtomicUsize::new(0);
    let mut pool = MemPool::<DropCounter<'_>>::new(1);

    {
      let mut handle = pool.alloc().unwrap();
      handle.init(DropCounter { drops: &drops });
    }

    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(pool.alloc().is_none());
  }
}
