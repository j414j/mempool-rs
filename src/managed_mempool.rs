//! A fixed-capacity pool with lifetime-bound handles and RAII slot return.
//!
//! This variant ties each [`SlotHandle`] to the lifetime of the originating
//! [`MemPool`]. That prevents handles from outliving the pool while still
//! allowing multiple handles to exist concurrently by using interior mutability
//! inside the pool.
//!
//! Slots are returned to the freelist automatically when their handles are
//! dropped. There is no public `free` method in this module.
//!
//! # Examples
//!
//! ```
//! use mempool::managed_mempool::MemPool;
//!
//! let pool = MemPool::new(2);
//! let mut a = pool.alloc().unwrap();
//! let mut b = pool.alloc().unwrap();
//!
//! a.init(10);
//! b.init(20);
//! assert_eq!(*a.get(), 10);
//! assert_eq!(*b.get(), 20);
//!
//! drop(a);
//! assert!(pool.alloc().is_some());
//! ```
//!
use std::{
  cell::{Cell, UnsafeCell},
  mem::MaybeUninit,
};

struct Slot<T> {
  elem: UnsafeCell<MaybeUninit<T>>,
  next: Cell<usize>,
}

pub struct MemPool<T> {
  buf: Vec<Slot<T>>,
  head: Cell<usize>,
}

impl<T> MemPool<T> {
  /// Creates a pool with space for at most `size` concurrently allocated slots.
  pub fn new(size: usize) -> Self {
    let mut buf = Vec::with_capacity(size);
    for i in 0..size {
      buf.push(Slot {
        elem: UnsafeCell::new(MaybeUninit::uninit()),
        next: Cell::new(i + 1),
      })
    }

    Self {
      buf,
      head: Cell::new(0),
    }
  }

  /// Allocates one slot from the pool.
  ///
  /// The returned handle starts out uninitialized. Call [`SlotHandle::init`]
  /// before using [`SlotHandle::get`] or [`SlotHandle::get_mut`].
  ///
  /// Returns `None` if the pool is exhausted.
  pub fn alloc<'a>(&'a self) -> Option<SlotHandle<'a, T>> {
    let head = self.head.get();
    if head >= self.buf.len() {
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
  pub unsafe fn alloc_unchecked<'a>(&'a self) -> SlotHandle<'a, T> {
    let idx = self.head.get();
    let slot = &self.buf[idx];
    self.head.set(slot.next.get());

    SlotHandle {
      idx,
      elem: slot.elem.get(),
      is_init: false,
      pool: &self,
    }
  }
  /// Returns a slot to the freelist.
  ///
  /// This is used internally from [`Drop`] for [`SlotHandle`].
  fn free_unchecked(&self, slot_idx: usize) {
    self.buf[slot_idx].next.set(self.head.get());
    self.head.set(slot_idx);
  }
}

pub struct SlotHandle<'a, T> {
  idx: usize,
  elem: *mut MaybeUninit<T>,
  is_init: bool,
  pool: &'a MemPool<T>,
}

impl<'a, T> SlotHandle<'a, T> {
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

impl<'a, T> Drop for SlotHandle<'a, T> {
  fn drop(&mut self) {
    if self.is_init {
      unsafe {
        (*self.elem).assume_init_drop();
      }
    }
    self.pool.free_unchecked(self.idx);
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
    let pool = MemPool::<u32>::new(2);
    let _a = pool.alloc().unwrap();
    let _b = pool.alloc().unwrap();
    assert!(pool.alloc().is_none());
  }

  #[test]
  fn dropping_handle_returns_slot_to_pool() {
    let pool = MemPool::<u32>::new(1);
    let handle = pool.alloc().unwrap();
    drop(handle);
    assert!(pool.alloc().is_some());
  }

  #[test]
  fn get_and_get_mut_work_after_init() {
    let pool = MemPool::<u32>::new(1);
    let mut handle = pool.alloc().unwrap();
    handle.init(3);
    assert_eq!(*handle.get(), 3);
    *handle.get_mut() *= 4;
    assert_eq!(*handle.get(), 12);
  }

  #[test]
  fn dropping_handle_drops_inner_value() {
    let drops = AtomicUsize::new(0);
    let pool = MemPool::<DropCounter<'_>>::new(1);

    {
      let mut handle = pool.alloc().unwrap();
      handle.init(DropCounter { drops: &drops });
    }

    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(pool.alloc().is_some());
  }
}
