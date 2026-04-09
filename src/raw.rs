//! A fixed-capacity pool that exposes raw pointers instead of typed handles.
//!
//! This is the most manual variant in the crate. `MemPool<T>` hands out
//! `*mut T` pointers to uninitialized slots. The caller is fully responsible for
//! initialization, reads, mutation, dropping the value in place, and returning
//! the slot to the pool.
//!
//! # Safety Model
//!
//! The pool only manages slot reuse. It does not track whether a slot is
//! initialized or whether a `T` still needs to be dropped.
//!
//! Callers must therefore uphold all of the following:
//! - initialize the pointee before reading it
//! - drop the pointee before reusing or freeing a slot that contains a live `T`
//! - never use a pointer after freeing its slot
//! - never let a pointer outlive the pool that created it
//! - only free pointers that came from this pool
//!
//! The checked [`MemPool::free`] variant verifies only that the pointer maps to
//! a slot index within the pool allocation. It does not prove pointer identity.
//!
//! # Examples
//!
//! ```
//! use mempool::raw::MemPool;
//!
//! let mut pool = MemPool::new(1);
//! let ptr: *mut i32 = pool.alloc().unwrap();
//!
//! unsafe {
//!   ptr.write(42);
//!   assert_eq!(*ptr, 42);
//!   std::ptr::drop_in_place(ptr);
//! }
//!
//! pool.free(ptr);
//! assert!(pool.alloc().is_some());
//! ```
//!
use alloc::vec::Vec;
use core::{cell::UnsafeCell, mem::MaybeUninit};

struct Slot<T> {
  idx: usize,
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
        idx: i,
        elem: UnsafeCell::new(MaybeUninit::uninit()),
        next: i + 1,
      })
    }

    Self { buf, head: 0 }
  }

  /// Allocates one slot from the pool.
  ///
  /// The returned pointer targets uninitialized storage for one `T`.
  ///
  /// Returns `None` if the pool is exhausted.
  pub fn alloc<'a>(&'a mut self) -> Option<*mut T> {
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
  pub unsafe fn alloc_unchecked<'a>(&'a mut self) -> *mut T {
    let idx = self.head;
    let slot = &self.buf[idx];
    self.head = slot.next;

    unsafe { (*slot.elem.get()).as_mut_ptr() }
  }

  fn calculate_idx_from_ptr(&mut self, ptr: *mut T) -> usize {
    unsafe { (*(ptr.byte_sub(core::mem::offset_of!(Slot<T>, elem)) as *const Slot<T>)).idx }
  }

  /// Frees a pointer without validating that it belongs to this pool.
  ///
  /// # Safety
  ///
  /// The caller must ensure all of the following:
  /// - `handle` was returned by a previous call to [`MemPool::alloc`] on this
  ///   exact pool.
  /// - `handle` has not already been freed.
  /// - the pointee has already been dropped if it contains a live `T`.
  /// - No references previously derived from the handle are used again after
  ///   this call.
  pub unsafe fn free_unchecked(&mut self, handle: *mut T) {
    let idx = self.calculate_idx_from_ptr(handle);
    self.buf[idx].next = self.head;
    self.head = idx;
  }

  /// Frees a pointer previously allocated from this pool.
  ///
  /// Panics if the pointer maps outside the pool allocation.
  pub fn free<'a>(&'a mut self, handle: *mut T) {
    let idx = self.calculate_idx_from_ptr(handle);
    if idx >= self.buf.len() {
      panic!("Illegal free");
    }

    unsafe {
      self.free_unchecked(handle);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::MemPool;
  use core::sync::atomic::{AtomicUsize, Ordering};

  #[derive(Debug, PartialEq, Eq)]
  struct Pair {
    left: u32,
    right: u32,
  }

  struct DropCounter<'a> {
    drops: &'a AtomicUsize,
  }

  impl Drop for DropCounter<'_> {
    fn drop(&mut self) {
      self.drops.fetch_add(1, Ordering::SeqCst);
    }
  }

  #[test]
  fn alloc_returns_none_when_pool_is_exhausted() {
    let mut pool = MemPool::<u32>::new(2);

    let a = pool.alloc();
    let b = pool.alloc();
    let c = pool.alloc();

    assert!(a.is_some());
    assert!(b.is_some());
    assert!(c.is_none());
  }

  #[test]
  fn free_makes_slot_available_again() {
    let mut pool = MemPool::<u32>::new(1);

    let ptr = pool.alloc().unwrap();
    unsafe {
      ptr.write(41);
      assert_eq!(*ptr, 41);
      core::ptr::drop_in_place(ptr);
    }
    pool.free(ptr);

    let reused = pool.alloc().unwrap();
    assert_eq!(reused, ptr);
  }

  #[test]
  fn raw_pointer_can_be_used_for_read_and_write() {
    let mut pool = MemPool::<Pair>::new(1);
    let ptr = pool.alloc().unwrap();

    unsafe {
      ptr.write(Pair {
        left: 10,
        right: 20,
      });
      assert_eq!((*ptr).left, 10);
      assert_eq!((*ptr).right, 20);

      (*ptr).left += 5;
      (*ptr).right += (*ptr).left;
      assert_eq!(
        *ptr,
        Pair {
          left: 15,
          right: 35
        }
      );

      core::ptr::drop_in_place(ptr);
    }
    pool.free(ptr);
  }

  #[test]
  fn free_is_lifo() {
    let mut pool = MemPool::<u32>::new(3);

    let a = pool.alloc().unwrap();
    let b = pool.alloc().unwrap();
    let c = pool.alloc().unwrap();

    pool.free(b);
    pool.free(c);

    let first = pool.alloc().unwrap();
    let second = pool.alloc().unwrap();

    assert_eq!(first, c);
    assert_eq!(second, b);

    pool.free(first);
    pool.free(second);
    pool.free(a);
  }

  #[test]
  #[should_panic]
  fn free_panics_for_pointer_outside_pool() {
    let mut pool = MemPool::<u64>::new(1);
    let begin = pool.buf.as_ptr() as usize;
    let outside = (begin + pool.buf.len() * core::mem::size_of::<super::Slot<u64>>()) as *mut u64;

    pool.free(outside);
  }

  #[test]
  fn unchecked_alloc_and_free_round_trip() {
    let mut pool = MemPool::<u32>::new(1);

    let ptr = unsafe { pool.alloc_unchecked() };
    unsafe {
      ptr.write(77);
      assert_eq!(*ptr, 77);
      core::ptr::drop_in_place(ptr);
      pool.free_unchecked(ptr);
    }

    let reused = pool.alloc().unwrap();
    assert_eq!(reused, ptr);
  }

  #[test]
  fn caller_controls_value_drop_before_free() {
    let drops = AtomicUsize::new(0);
    let mut pool = MemPool::<DropCounter<'_>>::new(1);
    let ptr = pool.alloc().unwrap();

    unsafe {
      ptr.write(DropCounter { drops: &drops });
    }
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    unsafe {
      core::ptr::drop_in_place(ptr);
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    pool.free(ptr);
  }
}
