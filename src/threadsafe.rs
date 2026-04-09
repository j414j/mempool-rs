//! A lifetime-bound pool with an atomic freelist head.
//!
//! This variant is the lock-free freelist mempool in the crate. Shared pool
//! state is coordinated through atomics, and the freelist head is stored as a
//! tagged index to reduce ABA issues on push/pop of the free-slot stack.
//!
//! Slots are returned to the freelist automatically when their handles are
//! dropped.
//!
//! # Examples
//!
//! ```
//! use mempool::threadsafe::MemPool;
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
  /// The returned handle starts out uninitialized. Call [`SlotHandle::init`]
  /// before using [`SlotHandle::get`] or [`SlotHandle::get_mut`].
  ///
  /// Returns `None` if the pool is exhausted.
  pub fn alloc<'a>(&'a self) -> Option<SlotHandle<'a, T>> {
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

  pub fn try_alloc<'a>(&'a self) -> Result<SlotHandle<'a, T>, TryAllocFailReason> {
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

  unsafe fn try_alloc_unchecked_with_head<'a>(&'a self, head: usize) -> Option<SlotHandle<'a, T>> {
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

    Some(SlotHandle {
      idx: index,
      elem: slot.elem.get(),
      is_init: false,
      pool: &self,
    })
  }

  pub unsafe fn try_alloc_unchecked<'a>(&'a self) -> Option<SlotHandle<'a, T>> {
    let head = self.head.load(Ordering::Acquire);
    unsafe { self.try_alloc_unchecked_with_head(head) }
  }

  /// Allocates one slot without checking whether the pool is exhausted.
  ///
  /// # Safety
  ///
  /// The caller must ensure that the pool still contains at least one free
  /// slot before calling this function.
  pub unsafe fn alloc_unchecked<'a>(&'a self) -> SlotHandle<'a, T> {
    loop {
      match unsafe { self.try_alloc_unchecked() } {
        Some(v) => return v,
        None => continue,
      }
    }
  }

  fn try_free_unchecked(&self, slot_idx: usize) -> bool {
    let head = self.head.load(Ordering::Acquire);
    let (tag, index) = unpack(head);
    self.buf[slot_idx].next.store(index, Ordering::Release);
    self
      .head
      .compare_exchange(
        head,
        pack(tag + 1, slot_idx),
        Ordering::AcqRel,
        Ordering::Relaxed,
      )
      .is_ok()
  }

  /// Returns a slot to the freelist.
  ///
  /// This is used internally from [`Drop`] for [`SlotHandle`].
  fn free_unchecked(&self, slot_idx: usize) {
    while !self.try_free_unchecked(slot_idx) {}
  }
}

unsafe impl<T> Sync for MemPool<T> {}

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
  use super::{MemPool, TryAllocFailReason};
  use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  };
  use std::thread;

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
    drop(handle);
    assert!(pool.alloc().is_some());
  }

  #[test]
  fn unchecked_alloc_round_trips() {
    let pool = MemPool::<u32>::new(1);
    let mut handle = unsafe { pool.alloc_unchecked() };
    handle.init(17);
    assert_eq!(unsafe { *handle.get_unchecked() }, 17);
    drop(handle);
    assert!(pool.alloc().is_some());
  }

  #[test]
  fn supports_multiple_threads_allocating() {
    let pool = Arc::new(MemPool::<u32>::new(4));
    let sum = AtomicUsize::new(0);

    thread::scope(|scope| {
      for i in 0..4 {
        let pool = Arc::clone(&pool);
        let sum_ref = &sum;
        scope.spawn(move || {
          let mut handle = pool.alloc().unwrap();
          handle.init(i as u32);
          sum_ref.fetch_add(*handle.get() as usize, Ordering::SeqCst);
        });
      }
    });

    assert_eq!(sum.load(Ordering::SeqCst), 6);
    assert!(pool.alloc().is_some());
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
  }
}
