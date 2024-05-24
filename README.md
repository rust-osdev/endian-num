# endian-num

[![Crates.io](https://img.shields.io/crates/v/endian-num)](https://crates.io/crates/endian-num)
[![docs.rs](https://img.shields.io/docsrs/endian-num)](https://docs.rs/endian-num)
[![CI](https://github.com/rust-osdev/endian-num/actions/workflows/ci.yml/badge.svg)](https://github.com/rust-osdev/endian-num/actions/workflows/ci.yml)

This crate provides the [`Be`] (big-endian) and [`Le`] (little-endian) byte-order-aware numeric types.

Unlike the popular [`byteorder`] crate, which focuses on the action of encoding and decoding numbers to and from byte streams, this crate focuses on the state of numbers.
This is useful to create structs that contain fields of a specific endianness for interoperability, such as in virtio.
In comparison to other crates that focus on state, this crate closely follows naming conventions from [`core::num`], has rich functionality, and extensive documentation of each method.

[`Be`]: https://docs.rs/endian-num/latest/endian_num/struct.Be.html
[`Le`]: https://docs.rs/endian-num/latest/endian_num/struct.Le.html
[`byteorder`]: https://docs.rs/byteorder
[`core::num`]: https://doc.rust-lang.org/stable/core/num/index.html

The core API looks _roughly_ like this (correspondingly for `Be`):

```rust
#[repr(transparent)]
pub struct<T> Le(pub T);

impl Le<T: Integer> {
    pub const fn from_ne(n: T) -> Self;
    pub const fn from_be(n: Be<T>) -> Self;

    pub const fn to_ne(self) -> T;
    pub const fn to_be(self) -> Be<T>;

    pub const fn to_be_bytes(self) -> [u8; mem::size_of::<Self>()];
    pub const fn to_le_bytes(self) -> [u8; mem::size_of::<Self>()];
    pub const fn to_ne_bytes(self) -> [u8; mem::size_of::<Self>()];

    pub const fn from_be_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self;
    pub const fn from_le_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self;
    pub const fn from_ne_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self;
}
```

The types also implement appropriate traits from [`core::cmp`], [`core::convert`], [`core::fmt`], and [`core::ops`] and provide additional helper methods for computations.

For API documentation, see the [docs].

[docs]: https://docs.rs/endian-num
[`core::cmp`]: https://doc.rust-lang.org/stable/core/cmp/index.html
[`core::convert`]: https://doc.rust-lang.org/stable/core/convert/index.html
[`core::fmt`]: https://doc.rust-lang.org/stable/core/fmt/index.html
[`core::ops`]: https://doc.rust-lang.org/stable/core/ops/index.html


## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
