//! A raw-pointer mempool with a tagged atomic freelist head.
//!
//! This is the most manual multithreaded variant in the crate. `MemPool<T>`
//! hands out `*mut T` pointers into fixed-capacity storage. The freelist head
//! is coordinated through atomics, and the head value is stored as a tagged
//! index to reduce ABA issues on push/pop of the free-slot stack.
//!
//! The pool only manages slot allocation and reuse. It does not track whether a
//! slot currently holds a live `T`, and it does not drop values automatically.
//!
//! Callers must therefore uphold all of the following:
//! - initialize the pointee before reading it
//! - drop the pointee before freeing or reusing a slot that contains a live `T`
//! - never use a pointer after freeing its slot
//! - only free pointers that came from this pool
//!
//! # Examples
//!
//! ```
//! use mempool::threadsafe_raw::MemPool;
//!
//! let pool = MemPool::new(1);
//! let ptr: *mut i32 = pool.alloc().unwrap();
//!
//! unsafe {
//!   ptr.write(42);
//!   assert_eq!(*ptr, 42);
//!   core::ptr::drop_in_place(ptr);
//! }
//!
//! unsafe { pool.free(ptr) };
//! assert!(pool.alloc().is_some());
//! ```
//!
use alloc::{boxed::Box, vec::Vec};
use core::{
  cell::UnsafeCell,
  mem::MaybeUninit,
  sync::atomic::{AtomicUsize, Ordering},
};

const INDEX_BITS: usize = usize::BITS as usize / 2;
const INDEX_MASK: usize = (1usize << INDEX_BITS) - 1;

const fn pack(tag: usize, index: usize) -> usize {
  tag << INDEX_BITS | index
}

const fn unpack(packed: usize) -> (usize, usize) {
  (packed >> INDEX_BITS, packed & INDEX_MASK)
}

pub enum TryAllocFailReason {
  BufferFull,
  CASFailed,
}

struct Slot<T> {
  idx: usize,
  elem: UnsafeCell<MaybeUninit<T>>,
  next: AtomicUsize,
}

pub struct MemPool<T> {
  buf: Box<[Slot<T>]>,
  head: AtomicUsize,
}

impl<T> MemPool<T> {
  /// Creates a pool with space for at most `size` concurrently allocated slots.
  pub fn new(size: usize) -> Self {
    assert!(size < INDEX_MASK);
    let mut buf = Vec::with_capacity(size);
    for i in 0..size {
      buf.push(Slot {
        idx: i,
        elem: UnsafeCell::new(MaybeUninit::uninit()),
        next: AtomicUsize::new(i + 1),
      })
    }

    Self {
      buf: buf.into_boxed_slice(),
      head: AtomicUsize::new(pack(0, 0)),
    }
  }

  /// Allocates one slot from the pool.
  ///
  /// The returned pointer targets uninitialized storage for one `T`.
  ///
  /// Returns `None` if the pool is exhausted.
  pub fn alloc(&self) -> Option<*mut T> {
    loop {
      match self.try_alloc() {
        Ok(v) => return Some(v),
        Err(e) => match e {
          TryAllocFailReason::BufferFull => return None,
          TryAllocFailReason::CASFailed => continue,
        },
      }
    }
  }

  pub fn try_alloc(&self) -> Result<*mut T, TryAllocFailReason> {
    let head = self.head.load(Ordering::Acquire);
    if unpack(head).1 >= self.buf.len() {
      return Err(TryAllocFailReason::BufferFull);
    }

    unsafe {
      self
        .try_alloc_unchecked_with_head(head)
        .map_or_else(|| Err(TryAllocFailReason::CASFailed), |s| Ok(s))
    }
  }

  unsafe fn try_alloc_unchecked_with_head(&self, head: usize) -> Option<*mut T> {
    let (tag, index) = unpack(head);
    let next_head = self.buf[index].next.load(Ordering::Acquire);
    if self
      .head
      .compare_exchange(
        head,
        pack(tag + 1, next_head),
        Ordering::AcqRel,
        Ordering::Relaxed,
      )
      .is_err()
    {
      return None;
    }
    let slot = &self.buf[index];

    Some(unsafe { (*slot.elem.get()).as_mut_ptr() })
  }

  pub unsafe fn try_alloc_unchecked<'a>(&'a self) -> Option<*mut T> {
    let head = self.head.load(Ordering::Acquire);
    unsafe { self.try_alloc_unchecked_with_head(head) }
  }

  /// Allocates one slot without checking whether the pool is exhausted.
  ///
  /// # Safety
  ///
  /// The caller must ensure that the pool still contains at least one free
  /// slot before calling this function.
  pub unsafe fn alloc_unchecked<'a>(&'a self) -> *mut T {
    loop {
      match unsafe { self.try_alloc_unchecked() } {
        Some(v) => return v,
        None => continue,
      }
    }
  }

  unsafe fn slot_from_elem(ptr: *mut T) -> *mut Slot<T> {
    unsafe { ptr.byte_sub(core::mem::offset_of!(Slot<T>, elem)) as *mut Slot<T> }
  }

  unsafe fn calculate_idx_from_ptr(ptr: *mut T) -> usize {
    unsafe { (*Self::slot_from_elem(ptr)).idx }
  }

  pub unsafe fn try_free_unchecked(&self, handle: *mut T) -> bool {
    let idx = unsafe { Self::calculate_idx_from_ptr(handle) };
    let head = self.head.load(Ordering::Acquire);
    let (tag, index) = unpack(head);
    self.buf[idx].next.store(index, Ordering::Release);
    self
      .head
      .compare_exchange(
        head,
        pack(tag + 1, idx),
        Ordering::AcqRel,
        Ordering::Relaxed,
      )
      .is_ok()
  }

  /// Returns a slot to the freelist without validating the pointer first.
  ///
  /// # Safety
  ///
  /// The caller must ensure all of the following:
  /// - `handle` was returned by a previous call to [`MemPool::alloc`] on this
  ///   exact pool
  /// - `handle` has not already been freed
  /// - the pointee has already been dropped if it contains a live `T`
  /// - no references previously derived from `handle` are used again after this
  ///   call
  pub unsafe fn free_unchecked(&self, handle: *mut T) {
    while !unsafe { self.try_free_unchecked(handle) } {}
  }

  /// Attempts to free a pointer after checking that it maps into this pool's
  /// storage range.
  ///
  /// This is still unsafe because preventing double free and use-after-free on *mut T
  /// is caller's responsibility and is not enforced by the pool.
  pub unsafe fn try_free(&self, handle: *mut T) -> bool {
    let range = self.buf.as_ptr_range();
    let slot = unsafe { Self::slot_from_elem(handle) } as *const Slot<T>;
    if !range.contains(&slot) {
      panic!("Illegal pointer");
    }

    unsafe { self.try_free_unchecked(handle) }
  }

  /// Frees a pointer after checking that it maps into this pool's storage
  /// range.
  ///
  /// Panics if the pointer does not belong to this pool.
  ///
  /// This is still unsafe because preventing double free and use-after-free on *mut T
  /// is caller's responsibility and is not enforced by the pool.
  pub unsafe fn free(&self, handle: *mut T) {
    let range = self.buf.as_ptr_range();
    let slot = unsafe { Self::slot_from_elem(handle) } as *const Slot<T>;
    if !range.contains(&slot) {
      panic!("Illegal pointer");
    }

    unsafe { self.free_unchecked(handle) };
  }
}

unsafe impl<T> Sync for MemPool<T> {}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
  use super::std::{panic, sync::Arc, thread};
  use super::{MemPool, TryAllocFailReason};
  use alloc::vec::Vec;
  use core::sync::atomic::{AtomicUsize, Ordering};

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
  fn try_alloc_reports_buffer_full() {
    let pool = MemPool::<u32>::new(1);
    let _handle = pool.alloc().unwrap();
    assert!(matches!(
      pool.try_alloc(),
      Err(TryAllocFailReason::BufferFull)
    ));
  }

  #[test]
  fn dropping_handle_returns_slot_to_pool() {
    let pool = MemPool::<u32>::new(1);
    let handle = pool.alloc().unwrap();
    unsafe {
      pool.free(handle);
    }
    assert!(pool.alloc().is_some());
  }

  #[test]
  fn unchecked_alloc_round_trips() {
    let pool = MemPool::<u32>::new(1);
    let handle = unsafe { pool.alloc_unchecked() };
    unsafe { handle.write(17) };
    assert_eq!(unsafe { *handle }, 17);
    unsafe { core::ptr::drop_in_place(handle) };
    unsafe {
      pool.free(handle);
    }
    assert!(pool.alloc().is_some());
  }

  #[test]
  fn caller_controls_value_drop_before_free() {
    let drops = AtomicUsize::new(0);
    let pool = MemPool::<DropCounter<'_>>::new(1);

    let handle = pool.alloc().unwrap();
    unsafe { handle.write(DropCounter { drops: &drops }) };
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    unsafe { core::ptr::drop_in_place(handle) };
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    unsafe {
      pool.free(handle);
    }
  }

  #[test]
  fn free_panics_for_pointer_outside_pool() {
    let pool = MemPool::<u64>::new(1);
    let begin = pool.buf.as_ptr() as usize;
    let outside = (begin + pool.buf.len() * core::mem::size_of::<super::Slot<u64>>()) as *mut u64;

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| unsafe {
      pool.free(outside);
    }));

    assert!(result.is_err());
  }

  #[test]
  fn supports_multiple_threads_allocating() {
    let pool = Arc::new(MemPool::<u32>::new(4));
    let sum = Arc::new(AtomicUsize::new(0));
    let mut joins = Vec::new();

    for i in 0..4 {
      let pool = Arc::clone(&pool);
      let sum = Arc::clone(&sum);
      joins.push(thread::spawn(move || {
        let handle = pool.alloc().unwrap();
        unsafe {
          handle.write(i as u32);
          sum.fetch_add(*handle as usize, Ordering::SeqCst);
          core::ptr::drop_in_place(handle);
        }
        unsafe {
          pool.free(handle);
        }
      }));
    }

    for join in joins {
      join.join().unwrap();
    }

    assert_eq!(sum.load(Ordering::SeqCst), 6);
    assert!(pool.alloc().is_some());
  }
}
