# mempool

`mempool` is a small `no_std` crate containing a few fixed-capacity memory pool implementations with different ownership and synchronization tradeoffs.

## Pool Variants

- `basic`: single-threaded pool with movable handles and explicit `free`
- `managed`: single-threaded pool with lifetime-bound handles and RAII slot return
- `raw`: single-threaded raw-pointer pool where the caller drops `T`
- `raw_compact`: single-threaded raw-pointer pool with a more compact slot layout; the pool drops `T` on free
- `threadsafe`: multithreaded pool with managed handles and atomic freelist bookkeeping
- `threadsafe_raw`: multithreaded raw-pointer pool with a tagged atomic freelist head

## Design Notes

All pools are fixed-capacity. They allocate backing storage up front and then reuse slots instead of allocating per element.

The modules do not all offer the same safety guarantees:

- handle-based pools provide a higher-level API around initialization and access
- raw-pointer pools are intentionally low-level and place more responsibility on the caller
- checked APIs generally perform shallow validation
- unchecked APIs assume the caller is maintaining the pool invariants correctly

In particular, the raw variants are performance-oriented primitives. Callers are responsible for:

- initializing values before reading them
- avoiding double-free and use-after-free
- preventing aliasing and data races when sharing raw pointers

For `raw`, callers also drop `T` manually before freeing. For `raw_compact`, the pool drops `T` when the slot is freed.

## Usage

Add the crate as a dependency and pick the module that matches the ownership model you want.

```rust
use mempool::managed::MemPool;

let pool = MemPool::new(2);

let mut a = pool.alloc().unwrap();
let mut b = pool.alloc().unwrap();

a.init(10);
b.init(20);

assert_eq!(*a.get(), 10);
assert_eq!(*b.get(), 20);
```

For a raw-pointer variant:

```rust
use mempool::raw::MemPool;

let pool = MemPool::new(1);
let ptr: *mut i32 = pool.alloc().unwrap();

unsafe {
    ptr.write(42);
    assert_eq!(*ptr, 42);
    core::ptr::drop_in_place(ptr);
    pool.free(ptr);
}
```

For the compact raw variant:

```rust
use mempool::raw_compact::MemPool;

let mut pool = MemPool::new(1);
let ptr: *mut i32 = pool.alloc().unwrap();

unsafe {
    ptr.write(42);
    assert_eq!(*ptr, 42);
    pool.free(ptr);
}
```

## Testing

Run the library test suite with:

```sh
cargo test --lib
```
