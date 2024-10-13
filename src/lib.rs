//! Byte-order-aware numeric types.
//!
//! This crate provides the [`Be`] (big-endian) and [`Le`] (little-endian) byte-order-aware numeric types.
//!
//! Unlike the popular [`byteorder`] crate, which focuses on the action of encoding and decoding numbers to and from byte streams, this crate focuses on the state of numbers.
//! This is useful to create structs that contain fields of a specific endianness for interoperability, such as in virtio.
//! In comparison to other crates that focus on state, this crate closely follows naming conventions from [`core::num`], has rich functionality, and extensive documentation of each method.
//!
//! [`byteorder`]: https://docs.rs/byteorder
//!
//! The core API looks _roughly_ like this (correspondingly for `Be`):
//!
//! ```ignore
//! #[repr(transparent)]
//! pub struct<T> Le(pub T);
//!
//! impl Le<T: Integer> {
//!     pub const fn from_ne(n: T) -> Self;
//!     pub const fn from_be(n: Be<T>) -> Self;
//!
//!     pub const fn to_ne(self) -> T;
//!     pub const fn to_be(self) -> Be<T>;
//!
//!     pub const fn to_be_bytes(self) -> [u8; mem::size_of::<Self>()];
//!     pub const fn to_le_bytes(self) -> [u8; mem::size_of::<Self>()];
//!     pub const fn to_ne_bytes(self) -> [u8; mem::size_of::<Self>()];
//!
//!     pub const fn from_be_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self;
//!     pub const fn from_le_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self;
//!     pub const fn from_ne_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self;
//! }
//! ```
//!
//! The types also implement appropriate traits from [`core::cmp`], [`core::convert`], [`core::fmt`], and [`core::ops`] and provide additional helper methods for computations.
//!
//! In addition to widening and byte-reordering [`From`] implementations, the endian number types implement conversions to and from arrays of smaller number types of the same ordering.
//! This is useful in situations, where a larger field has to be treated as multiple smaller field.
//!
//! # Examples
//!
//! ```
//! use endian_num::Le;
//!
//! let a = Le::<u32>::from_ne(0x1A);
//! let b = Le::<u32>::from_ne(0x2B00);
//!
//! assert_eq!((a + b).to_le_bytes(), [0x1A, 0x2B, 0x00, 0x00]);
//! ```
//!
//! # Optional features
//!
//! This crate has the following optional features:
//!
//! - [`bitflags`] — `Be` and `Le` implement [`Bits`], [`ParseHex`], and [`WriteHex`].
//! - [`bytemuck`] — `Be` and `Le` implement [`Zeroable`] and [`Pod`].
//! - `linux-types` — Type aliases like in [`linux/types.h`], such as [`le32`].
//! - [`zerocopy`] — `Be` and `Le` implement [`KnownLayout`], [`Immutable`], [`FromBytes`], and [`IntoBytes`].
//!
//! [`Bits`]: bitflags::Bits
//! [`ParseHex`]: bitflags::parser::ParseHex
//! [`WriteHex`]: bitflags::parser::WriteHex
//! [`Zeroable`]: bytemuck::Zeroable
//! [`Pod`]: bytemuck::Pod
//! [`linux/types.h`]: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/linux/types.h?h=v6.9#n36
//! [`KnownLayout`]: zerocopy::KnownLayout
//! [`Immutable`]: zerocopy::Immutable
//! [`FromBytes`]: zerocopy::FromBytes
//! [`IntoBytes`]: zerocopy::IntoBytes
//!
//! # Related crates
//!
//! Several crates provide alternative approaches to byte-order-aware numeric types:
//!
//! - [endian-type](https://docs.rs/endian-type)
//! - [endian-type-rs](https://docs.rs/endian-type-rs) — Depends on `num`.
//! - [endiantype](https://docs.rs/endiantype)
//! - [nora_endian](https://docs.rs/nora_endian)
//! - [simple_endian](https://docs.rs/simple_endian) — Also provides `f32`, `f64`, and `bool` types.
//! - [`zerocopy::byteorder`] — These types are [`Unaligned`](zerocopy::Unaligned), which makes them unsuitable for volatile memory operations.
//!
//! [`zerocopy::byteorder`]: https://docs.rs/zerocopy/0.7/zerocopy/byteorder/index.html

#![no_std]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

#[macro_use]
mod internal_macros;

use core::cmp::Ordering;
use core::iter::{Product, Sum};
use core::num::TryFromIntError;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use core::{fmt, mem};

/// An integer stored in big-endian byte order.
///
/// # Examples
///
/// ```
/// use endian_num::Be;
///
/// let n = 0x1Au32;
///
/// if cfg!(target_endian = "big") {
///     assert_eq!(Be::<u32>::from_ne(n).0, n);
/// } else {
///     assert_eq!(Be::<u32>::from_ne(n).0, n.swap_bytes());
/// }
/// ```
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck_derive::Zeroable, bytemuck_derive::Pod)
)]
#[cfg_attr(
    feature = "zerocopy",
    derive(
        zerocopy_derive::KnownLayout,
        zerocopy_derive::Immutable,
        zerocopy_derive::FromBytes,
        zerocopy_derive::IntoBytes,
    )
)]
#[derive(Default, Hash, PartialEq, Eq, Clone, Copy)]
#[repr(transparent)]
pub struct Be<T>(pub T);

/// An integer stored in little-endian byte order.
///
/// # Examples
///
/// ```
/// use endian_num::Le;
///
/// let n = 0x1Au32;
///
/// if cfg!(target_endian = "little") {
///     assert_eq!(Le::<u32>::from_ne(n).0, n);
/// } else {
///     assert_eq!(Le::<u32>::from_ne(n).0, n.swap_bytes());
/// }
/// ```
#[cfg_attr(
    feature = "bytemuck",
    derive(bytemuck_derive::Zeroable, bytemuck_derive::Pod)
)]
#[cfg_attr(
    feature = "zerocopy",
    derive(
        zerocopy_derive::KnownLayout,
        zerocopy_derive::Immutable,
        zerocopy_derive::FromBytes,
        zerocopy_derive::IntoBytes,
    )
)]
#[derive(Default, Hash, PartialEq, Eq, Clone, Copy)]
#[repr(transparent)]
pub struct Le<T>(pub T);

macro_rules! impl_fmt {
    (impl<T> $Trait:ident for Xe<T>) => {
        impl_fmt!(impl<T> $Trait for Be<T>);
        impl_fmt!(impl<T> $Trait for Le<T>);
    };
    (impl<T> $Trait:ident for $SelfT:ident<T>) => {
        impl<T> fmt::$Trait for $SelfT<T>
        where
            Self: Copy + Into<T>,
            T: fmt::$Trait,
        {
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                (*self).into().fmt(f)
            }
        }
    };
}

impl_fmt! { impl<T> Debug for Xe<T> }
impl_fmt! { impl<T> Display for Xe<T> }
impl_fmt! { impl<T> Binary for Xe<T> }
impl_fmt! { impl<T> Octal for Xe<T> }
impl_fmt! { impl<T> LowerHex for Xe<T> }
impl_fmt! { impl<T> UpperHex for Xe<T> }
impl_fmt! { impl<T> LowerExp for Xe<T> }
impl_fmt! { impl<T> UpperExp for Xe<T> }

macro_rules! unop_impl {
    (impl $Trait:ident, $method:ident for $T:ty) => {
        impl $Trait for $T {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn $method(self) -> Self::Output {
                Self::Output::from_ne($Trait::$method(self.to_ne()))
            }
        }
    };
}

macro_rules! binop_impl {
    (impl $Trait:ident<Self>, $method:ident for $T:ty) => {
        impl $Trait<Self> for $T {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: Self) -> Self::Output {
                Self::Output::from_ne($Trait::$method(self.to_ne(), rhs.to_ne()))
            }
        }
    };
    (impl $Trait:ident<$Rhs:ty>, $method:ident for $T:ty) => {
        impl $Trait<$Rhs> for $T {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: $Rhs) -> Self::Output {
                Self::Output::from_ne($Trait::$method(self.to_ne(), rhs))
            }
        }
    };
}

macro_rules! op_assign_impl {
    (impl $Trait:ident<$Rhs:ty>, $method:ident for $T:ty, $binop:path) => {
        impl $Trait<$Rhs> for $T {
            #[inline]
            #[track_caller]
            fn $method(&mut self, rhs: $Rhs) {
                *self = $binop(*self, rhs);
            }
        }
    };
}

macro_rules! endian_impl {
    ($(Xe<$T:ty>)*) => {$(
        endian_impl! { Be<$T> }
        endian_impl! { Le<$T> }
    )*};
    ($($Xe:ident<$T:ty>)*) => {$(
        impl PartialOrd for $Xe<$T> {
            #[inline]
            fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
                Some(self.cmp(rhs))
            }
        }

        impl Ord for $Xe<$T> {
            #[inline]
            fn cmp(&self, rhs: &Self) -> Ordering {
                Ord::cmp(&self.to_ne(), &rhs.to_ne())
            }
        }

        unop_impl! { impl Not, not for $Xe<$T> }

        forward_ref_unop! { impl Not, not for $Xe<$T> }

        binop_impl! { impl Add<Self>, add for $Xe<$T> }
        binop_impl! { impl BitAnd<Self>, bitand for $Xe<$T> }
        binop_impl! { impl BitOr<Self>, bitor for $Xe<$T> }
        binop_impl! { impl BitXor<Self>, bitxor for $Xe<$T> }
        binop_impl! { impl Div<Self>, div for $Xe<$T> }
        binop_impl! { impl Mul<Self>, mul for $Xe<$T> }
        binop_impl! { impl Rem<Self>, rem for $Xe<$T> }
        binop_impl! { impl Sub<Self>, sub for $Xe<$T> }

        forward_ref_binop! { impl Add<$Xe<$T>>, add for $Xe<$T> }
        forward_ref_binop! { impl BitAnd<$Xe<$T>>, bitand for $Xe<$T> }
        forward_ref_binop! { impl BitOr<$Xe<$T>>, bitor for $Xe<$T> }
        forward_ref_binop! { impl BitXor<$Xe<$T>>, bitxor for $Xe<$T> }
        forward_ref_binop! { impl Div<$Xe<$T>>, div for $Xe<$T> }
        forward_ref_binop! { impl Mul<$Xe<$T>>, mul for $Xe<$T> }
        forward_ref_binop! { impl Rem<$Xe<$T>>, rem for $Xe<$T> }
        forward_ref_binop! { impl Sub<$Xe<$T>>, sub for $Xe<$T> }

        op_assign_impl! { impl AddAssign<Self>, add_assign for $Xe<$T>, Add::add }
        op_assign_impl! { impl BitAndAssign<Self>, bitand_assign for $Xe<$T>, BitAnd::bitand }
        op_assign_impl! { impl BitOrAssign<Self>, bitor_assign for $Xe<$T>, BitOr::bitor }
        op_assign_impl! { impl BitXorAssign<Self>, bitxor_assign for $Xe<$T>, BitXor::bitxor }
        op_assign_impl! { impl DivAssign<Self>, div_assign for $Xe<$T>, Div::div }
        op_assign_impl! { impl MulAssign<Self>, mul_assign for $Xe<$T>, Mul::mul }
        op_assign_impl! { impl RemAssign<Self>, rem_assign for $Xe<$T>, Rem::rem }
        op_assign_impl! { impl SubAssign<Self>, sub_assign for $Xe<$T>, Sub::sub }

        forward_ref_op_assign! { impl AddAssign<$Xe<$T>>, add_assign for $Xe<$T> }
        forward_ref_op_assign! { impl BitAndAssign<$Xe<$T>>, bitand_assign for $Xe<$T> }
        forward_ref_op_assign! { impl BitOrAssign<$Xe<$T>>, bitor_assign for $Xe<$T> }
        forward_ref_op_assign! { impl BitXorAssign<$Xe<$T>>, bitxor_assign for $Xe<$T> }
        forward_ref_op_assign! { impl DivAssign<$Xe<$T>>, div_assign for $Xe<$T> }
        forward_ref_op_assign! { impl MulAssign<$Xe<$T>>, mul_assign for $Xe<$T> }
        forward_ref_op_assign! { impl RemAssign<$Xe<$T>>, rem_assign for $Xe<$T> }
        forward_ref_op_assign! { impl SubAssign<$Xe<$T>>, sub_assign for $Xe<$T> }

        binop_impl! { impl Shl<u8>, shl for $Xe<$T> }
        binop_impl! { impl Shl<u16>, shl for $Xe<$T> }
        binop_impl! { impl Shl<u32>, shl for $Xe<$T> }
        binop_impl! { impl Shl<u64>, shl for $Xe<$T> }
        binop_impl! { impl Shl<u128>, shl for $Xe<$T> }
        binop_impl! { impl Shl<usize>, shl for $Xe<$T> }
        binop_impl! { impl Shl<i8>, shl for $Xe<$T> }
        binop_impl! { impl Shl<i16>, shl for $Xe<$T> }
        binop_impl! { impl Shl<i32>, shl for $Xe<$T> }
        binop_impl! { impl Shl<i64>, shl for $Xe<$T> }
        binop_impl! { impl Shl<i128>, shl for $Xe<$T> }
        binop_impl! { impl Shl<isize>, shl for $Xe<$T> }

        binop_impl! { impl Shr<u8>, shr for $Xe<$T> }
        binop_impl! { impl Shr<u16>, shr for $Xe<$T> }
        binop_impl! { impl Shr<u32>, shr for $Xe<$T> }
        binop_impl! { impl Shr<u64>, shr for $Xe<$T> }
        binop_impl! { impl Shr<u128>, shr for $Xe<$T> }
        binop_impl! { impl Shr<usize>, shr for $Xe<$T> }
        binop_impl! { impl Shr<i8>, shr for $Xe<$T> }
        binop_impl! { impl Shr<i16>, shr for $Xe<$T> }
        binop_impl! { impl Shr<i32>, shr for $Xe<$T> }
        binop_impl! { impl Shr<i64>, shr for $Xe<$T> }
        binop_impl! { impl Shr<i128>, shr for $Xe<$T> }
        binop_impl! { impl Shr<isize>, shr for $Xe<$T> }

        forward_ref_binop! { impl Shl<u8>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<u16>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<u32>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<u64>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<u128>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<usize>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<i8>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<i16>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<i32>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<i64>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<i128>, shl for $Xe<$T> }
        forward_ref_binop! { impl Shl<isize>, shl for $Xe<$T> }

        forward_ref_binop! { impl Shr<u8>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<u16>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<u32>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<u64>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<u128>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<usize>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<i8>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<i16>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<i32>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<i64>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<i128>, shr for $Xe<$T> }
        forward_ref_binop! { impl Shr<isize>, shr for $Xe<$T> }

        op_assign_impl! { impl ShlAssign<u8>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<u16>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<u32>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<u64>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<u128>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<usize>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<i8>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<i16>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<i32>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<i64>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<i128>, shl_assign for $Xe<$T>, Shl::shl }
        op_assign_impl! { impl ShlAssign<isize>, shl_assign for $Xe<$T>, Shl::shl }

        op_assign_impl! { impl ShrAssign<u8>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<u16>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<u32>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<u64>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<u128>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<usize>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<i8>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<i16>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<i32>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<i64>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<i128>, shr_assign for $Xe<$T>, Shr::shr }
        op_assign_impl! { impl ShrAssign<isize>, shr_assign for $Xe<$T>, Shr::shr }

        forward_ref_op_assign! { impl ShlAssign<u8>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<u16>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<u32>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<u64>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<u128>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<usize>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<i8>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<i16>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<i32>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<i64>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<i128>, shl_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShlAssign<isize>, shl_assign for $Xe<$T> }

        forward_ref_op_assign! { impl ShrAssign<u8>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<u16>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<u32>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<u64>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<u128>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<usize>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<i8>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<i16>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<i32>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<i64>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<i128>, shr_assign for $Xe<$T> }
        forward_ref_op_assign! { impl ShrAssign<isize>, shr_assign for $Xe<$T> }
    )*};
}

endian_impl! { Xe<u8> Xe<u16> Xe<u32> Xe<u64> Xe<u128> Xe<usize> Xe<i8> Xe<i16> Xe<i32> Xe<i64> Xe<i128> Xe<isize> }

macro_rules! impl_from {
    (Xe<$Small:ty> => Xe<$Large:ty>) => {
        impl_from!(Be<$Small> => Be<$Large>);
        impl_from!(Le<$Small> => Le<$Large>);
    };
    ($Small:ty => $Large:ty) => {
        impl From<$Small> for $Large {
            #[doc = concat!("Converts [`", stringify!($Small), "`] to [`", stringify!($Large), "`] losslessly.")]
            #[inline]
            fn from(small: $Small) -> Self {
                Self::from_ne(small.to_ne().into())
            }
        }
    };
}

// unsigned integer -> unsigned integer
impl_from!(Xe<u8> => Xe<u16>);
impl_from!(Xe<u8> => Xe<u32>);
impl_from!(Xe<u8> => Xe<u64>);
impl_from!(Xe<u8> => Xe<u128>);
impl_from!(Xe<u8> => Xe<usize>);
impl_from!(Xe<u16> => Xe<u32>);
impl_from!(Xe<u16> => Xe<u64>);
impl_from!(Xe<u16> => Xe<u128>);
impl_from!(Xe<u32> => Xe<u64>);
impl_from!(Xe<u32> => Xe<u128>);
impl_from!(Xe<u64> => Xe<u128>);

// signed integer -> signed integer
impl_from!(Xe<i8> => Xe<i16>);
impl_from!(Xe<i8> => Xe<i32>);
impl_from!(Xe<i8> => Xe<i64>);
impl_from!(Xe<i8> => Xe<i128>);
impl_from!(Xe<i8> => Xe<isize>);
impl_from!(Xe<i16> => Xe<i32>);
impl_from!(Xe<i16> => Xe<i64>);
impl_from!(Xe<i16> => Xe<i128>);
impl_from!(Xe<i32> => Xe<i64>);
impl_from!(Xe<i32> => Xe<i128>);
impl_from!(Xe<i64> => Xe<i128>);

// unsigned integer -> signed integer
impl_from!(Xe<u8> => Xe<i16>);
impl_from!(Xe<u8> => Xe<i32>);
impl_from!(Xe<u8> => Xe<i64>);
impl_from!(Xe<u8> => Xe<i128>);
impl_from!(Xe<u16> => Xe<i32>);
impl_from!(Xe<u16> => Xe<i64>);
impl_from!(Xe<u16> => Xe<i128>);
impl_from!(Xe<u32> => Xe<i64>);
impl_from!(Xe<u32> => Xe<i128>);
impl_from!(Xe<u64> => Xe<i128>);

// The C99 standard defines bounds on INTPTR_MIN, INTPTR_MAX, and UINTPTR_MAX
// which imply that pointer-sized integers must be at least 16 bits:
// https://port70.net/~nsz/c/c99/n1256.html#7.18.2.4
impl_from!(Xe<u16> => Xe<usize>);
impl_from!(Xe<u8> => Xe<isize>);
impl_from!(Xe<i16> => Xe<isize>);

macro_rules! impl_try_from {
    (Xe<$source:ty> => $(Xe<$target:ty>),+) => {$(
        impl_try_from!(Be<$source> => Be<$target>);
        impl_try_from!(Le<$source> => Le<$target>);
    )*};
    ($Xe:ident<$source:ty> => $Xe2:ident<$target:ty>) => {
        impl TryFrom<$Xe<$source>> for $Xe<$target> {
            type Error = TryFromIntError;

            /// Try to create the target number type from a source
            /// number type. This returns an error if the source value
            /// is outside of the range of the target type.
            #[inline]
            fn try_from(u: $Xe<$source>) -> Result<Self, Self::Error> {
                <$target>::try_from(u.to_ne()).map(Self::from_ne)
            }
        }
    };
}

// unsigned integer -> unsigned integer
impl_try_from!(Xe<u16> => Xe<u8>);
impl_try_from!(Xe<u32> => Xe<u8>, Xe<u16>);
impl_try_from!(Xe<u64> => Xe<u8>, Xe<u16>, Xe<u32>);
impl_try_from!(Xe<u128> => Xe<u8>, Xe<u16>, Xe<u32>, Xe<u64>);

// signed integer -> signed integer
impl_try_from!(Xe<i16> => Xe<i8>);
impl_try_from!(Xe<i32> => Xe<i8>, Xe<i16>);
impl_try_from!(Xe<i64> => Xe<i8>, Xe<i16>, Xe<i32>);
impl_try_from!(Xe<i128> => Xe<i8>, Xe<i16>, Xe<i32>, Xe<i64>);

// unsigned integer -> signed integer
impl_try_from!(Xe<u8> => Xe<i8>);
impl_try_from!(Xe<u16> => Xe<i8>, Xe<i16>);
impl_try_from!(Xe<u32> => Xe<i8>, Xe<i16>, Xe<i32>);
impl_try_from!(Xe<u64> => Xe<i8>, Xe<i16>, Xe<i32>, Xe<i64>);
impl_try_from!(Xe<u128> => Xe<i8>, Xe<i16>, Xe<i32>, Xe<i64>, Xe<i128>);

// signed integer -> unsigned integer
impl_try_from!(Xe<i8> => Xe<u8>, Xe<u16>, Xe<u32>, Xe<u64>, Xe<u128>);
impl_try_from!(Xe<i16> => Xe<u8>);
impl_try_from!(Xe<i16> => Xe<u16>, Xe<u32>, Xe<u64>, Xe<u128>);
impl_try_from!(Xe<i32> => Xe<u8>, Xe<u16>);
impl_try_from!(Xe<i32> => Xe<u32>, Xe<u64>, Xe<u128>);
impl_try_from!(Xe<i64> => Xe<u8>, Xe<u16>, Xe<u32>);
impl_try_from!(Xe<i64> => Xe<u64>, Xe<u128>);
impl_try_from!(Xe<i128> => Xe<u8>, Xe<u16>, Xe<u32>, Xe<u64>);
impl_try_from!(Xe<i128> => Xe<u128>);

// usize/isize
impl_try_from!(Xe<usize> => Xe<isize>);
impl_try_from!(Xe<isize> => Xe<usize>);

macro_rules! impl_sum_product {
    ($(Xe<$T:ty>)*) => {$(
        impl_sum_product!(Be<$T>);
        impl_sum_product!(Le<$T>);
    )*};
    ($Xe:ty) => (
        impl Sum for $Xe {
            fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
                iter.fold(<$Xe>::from_ne(0), |a, b| a + b)
            }
        }

        impl Product for $Xe {
            fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
                iter.fold(<$Xe>::from_ne(1), |a, b| a * b)
            }
        }

        impl<'a> Sum<&'a $Xe> for $Xe {
            fn sum<I: Iterator<Item=&'a Self>>(iter: I) -> Self {
                iter.fold(<$Xe>::from_ne(0), |a, b| a + b)
            }
        }

        impl<'a> Product<&'a $Xe> for $Xe {
            fn product<I: Iterator<Item=&'a Self>>(iter: I) -> Self {
                iter.fold(<$Xe>::from_ne(1), |a, b| a * b)
            }
        }
    );
}

impl_sum_product! { Xe<i8> Xe<i16> Xe<i32> Xe<i64> Xe<i128> Xe<isize> Xe<u8> Xe<u16> Xe<u32> Xe<u64> Xe<u128> Xe<usize> }

#[cfg(target_pointer_width = "16")]
macro_rules! usize_macro {
    ($macro:ident) => {
        $macro!(u16)
    };
}

#[cfg(target_pointer_width = "32")]
macro_rules! usize_macro {
    ($macro:ident) => {
        $macro!(u32)
    };
}

#[cfg(target_pointer_width = "64")]
macro_rules! usize_macro {
    ($macro:ident) => {
        $macro!(u64)
    };
}

#[rustfmt::skip]
macro_rules! rot {
    (u8) => { 2 };
    (u16) => { 4 };
    (u32) => { 8 };
    (u64) => { 12 };
    (u128) => { 16 };
    (usize) => { usize_macro!(rot) };
    (i8) => { rot!(u8) };
    (i16) => { rot!(u16) };
    (i32) => { rot!(u32) };
    (i64) => { rot!(u64) };
    (i128) => { rot!(u128) };
    (isize) => { rot!(usize) };
}

#[rustfmt::skip]
macro_rules! rot_op {
    (u8) => { "0x82" };
    (u16) => { "0xa003" };
    (u32) => { "0x10000b3" };
    (u64) => { "0xaa00000000006e1" };
    (u128) => { "0x13f40000000000000000000000004f76" };
    (usize) => { usize_macro!(rot_op) };
    (i8) => { "-0x7e" };
    (i16) => { "-0x5ffd" };
    (i32) => { rot_op!(u32) };
    (i64) => { rot_op!(u64) };
    (i128) => { rot_op!(u128) };
    (isize) => { rot_op!(usize) };
}

#[rustfmt::skip]
macro_rules! rot_result {
    (u8) => { "0xa" };
    (u16) => { "0x3a" };
    (u32) => { "0xb301" };
    (u64) => { "0x6e10aa" };
    (u128) => { "0x4f7613f4" };
    (usize) => { usize_macro!(rot_result) };
    (i8) => { rot_result!(u8) };
    (i16) => { rot_result!(u16) };
    (i32) => { rot_result!(u32) };
    (i64) => { rot_result!(u64) };
    (i128) => { rot_result!(u128) };
    (isize) => { rot_result!(usize) };
}

#[rustfmt::skip]
macro_rules! swap_op {
    (u8) => { "0x12" };
    (u16) => { "0x1234" };
    (u32) => { "0x12345678" };
    (u64) => { "0x1234567890123456" };
    (u128) => { "0x12345678901234567890123456789012" };
    (usize) => { usize_macro!(swap_op) };
    (i8) => { swap_op!(u8) };
    (i16) => { swap_op!(u16) };
    (i32) => { swap_op!(u32) };
    (i64) => { swap_op!(u64) };
    (i128) => { swap_op!(u128) };
    (isize) => { swap_op!(usize) };
}

#[rustfmt::skip]
macro_rules! swapped {
    (u8) => { "0x12" };
    (u16) => { "0x3412" };
    (u32) => { "0x78563412" };
    (u64) => { "0x5634129078563412" };
    (u128) => { "0x12907856341290785634129078563412" };
    (usize) => { usize_macro!(swapped) };
    (i8) => { swapped!(u8) };
    (i16) => { swapped!(u16) };
    (i32) => { swapped!(u32) };
    (i64) => { swapped!(u64) };
    (i128) => { swapped!(u128) };
    (isize) => { swapped!(usize) };
}

#[rustfmt::skip]
macro_rules! reversed {
    (u8) => { "0x48" };
    (u16) => { "0x2c48" };
    (u32) => { "0x1e6a2c48" };
    (u64) => { "0x6a2c48091e6a2c48" };
    (u128) => { "0x48091e6a2c48091e6a2c48091e6a2c48" };
    (usize) => { usize_macro!(reversed) };
    (i8) => { reversed!(u8) };
    (i16) => { reversed!(u16) };
    (i32) => { reversed!(u32) };
    (i64) => { reversed!(u64) };
    (i128) => { reversed!(u128) };
    (isize) => { reversed!(usize) };
}

macro_rules! be_bytes {
    (u8) => { "[0x12]" };
    (u16) => { "[0x12, 0x34]" };
    (u32) => { "[0x12, 0x34, 0x56, 0x78]" };
    (u64) => { "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]" };
    (u128) => { "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12]" };
    (usize) => { usize_macro!(be_bytes) };
    (i8) => { be_bytes!(u8) };
    (i16) => { be_bytes!(u16) };
    (i32) => { be_bytes!(u32) };
    (i64) => { be_bytes!(u64) };
    (i128) => { be_bytes!(u128) };
    (isize) => { be_bytes!(usize) };
}

macro_rules! le_bytes {
    (u8) => { "[0x12]" };
    (u16) => { "[0x34, 0x12]" };
    (u32) => { "[0x78, 0x56, 0x34, 0x12]" };
    (u64) => { "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]" };
    (u128) => { "[0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]" };
    (usize) => { usize_macro!(le_bytes) };
    (i8) => { le_bytes!(u8) };
    (i16) => { le_bytes!(u16) };
    (i32) => { le_bytes!(u32) };
    (i64) => { le_bytes!(u64) };
    (i128) => { le_bytes!(u128) };
    (isize) => { le_bytes!(usize) };
}

macro_rules! endian_int_impl {
    ($(Xe<$T:ident>)*) => {$(
        endian_int_impl! { Be<$T>, from_be, to_be, "big", Le, from_le, to_le, "little" }
        endian_int_impl! { Le<$T>, from_le, to_le, "little", Be, from_be, to_be, "big" }
    )*};
    ($Xe:ident<$T:ident>, $from_xe:ident, $to_xe:ident, $order:literal, $Other:ident, $from_other:ident, $to_other:ident, $order_other:literal) => {
        impl $Xe<$T> {
            /// The smallest value that can be represented by this integer type.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::MIN, ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", stringify!($T), "::MIN));")]
            /// ```
            pub const MIN: Self = Self::from_ne(<$T>::MIN);

            /// The largest value that can be represented by this integer type.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::MAX, ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", stringify!($T), "::MAX));")]
            /// ```
            pub const MAX: Self = Self::from_ne(<$T>::MAX);

            /// The size of this integer type in bits.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::BITS, ", stringify!($T), "::BITS);")]
            /// ```
            pub const BITS: u32 = <$T>::BITS;

            #[doc = concat!("Creates a new ", $order, "-endian integer from a native-endian integer.")]
            ///
            #[doc = concat!("On ", $order, " endian, this is a no-op. On ", $order_other, " endian, the bytes are swapped.")]
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = 0x1A", stringify!($T), ";")]
            ///
            #[doc = concat!("if cfg!(target_endian = \"", $order, "\") {")]
            #[doc = concat!("    assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(n).0, n);")]
            /// } else {
            #[doc = concat!("    assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(n).0, n.swap_bytes());")]
            /// }
            /// ```
            #[must_use]
            #[inline]
            pub const fn from_ne(n: $T) -> Self {
                Self(n.$to_xe())
            }

            #[doc = concat!("Creates a new ", $order, "-endian integer from a ", $order_other, "-endian integer.")]
            ///
            /// This always swaps the bytes.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use endian_num::{Be, Le};")]
            ///
            #[doc = concat!("let n = 0x1A", stringify!($T), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::", stringify!($from_other), "(", stringify!($Other), "(n)).0, n.swap_bytes());")]
            /// ```
            #[must_use]
            #[inline]
            pub const fn $from_other(n: $Other<$T>) -> Self {
                Self(n.0.swap_bytes())
            }

            /// Returns the integer in native-endian byte order.
            ///
            #[doc = concat!("On ", $order, " endian, this is a no-op. On ", $order_other, " endian, the bytes are swapped.")]
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = 0x1A", stringify!($T), ";")]
            ///
            #[doc = concat!("if cfg!(target_endian = \"", $order, "\") {")]
            #[doc = concat!("    assert_eq!(", stringify!($Xe), "(n).to_ne(), n);")]
            /// } else {
            #[doc = concat!("    assert_eq!(", stringify!($Xe), "(n).to_ne(), n.swap_bytes());")]
            /// }
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn to_ne(self) -> $T {
                <$T>::$from_xe(self.0)
            }

            #[doc = concat!("Returns the integer in ", $order_other, "-endian byte order.")]
            ///
            /// This always swaps the bytes.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = 0x1A", stringify!($T), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "(n).", stringify!($to_other), "().0, n.swap_bytes());")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn $to_other(self) -> $Other<$T> {
                $Other(self.0.swap_bytes())
            }

            /// Return the memory representation of this integer as a byte array in big-endian (network) byte order.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let bytes = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", swap_op!($T), ").to_be_bytes();")]
            #[doc = concat!("assert_eq!(bytes, ", be_bytes!($T), ");")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn to_be_bytes(self) -> [u8; mem::size_of::<Self>()] {
                self.to_ne().to_be_bytes()
            }

            /// Return the memory representation of this integer as a byte array in little-endian byte order.
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let bytes = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", swap_op!($T), ").to_le_bytes();")]
            #[doc = concat!("assert_eq!(bytes, ", le_bytes!($T), ");")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn to_le_bytes(self) -> [u8; mem::size_of::<Self>()] {
                self.to_ne().to_le_bytes()
            }

            /// Return the memory representation of this integer as a byte array in
            /// native byte order.
            ///
            /// As the target platform's native endianness is used, portable code
            /// should use [`to_be_bytes`] or [`to_le_bytes`], as appropriate,
            /// instead.
            ///
            /// [`to_be_bytes`]: Self::to_be_bytes
            /// [`to_le_bytes`]: Self::to_le_bytes
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let bytes = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", swap_op!($T), ").to_ne_bytes();")]
            /// assert_eq!(
            ///     bytes,
            ///     if cfg!(target_endian = "big") {
            #[doc = concat!("        ", be_bytes!($T))]
            ///     } else {
            #[doc = concat!("        ", le_bytes!($T))]
            ///     }
            /// );
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn to_ne_bytes(self) -> [u8; mem::size_of::<Self>()] {
                self.to_ne().to_ne_bytes()
            }

            #[doc = concat!("Create a ", $order, " endian integer value from its representation as a byte array in ", $order, " endian.")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let value = ", stringify!($Xe), "::<", stringify!($T), ">::from_be_bytes(", be_bytes!($T), ");")]
            #[doc = concat!("assert_eq!(value.to_ne(), ", swap_op!($T), ");")]
            /// ```
            ///
            /// When starting from a slice rather than an array, fallible conversion APIs can be used:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("fn read_be_", stringify!($T), "(input: &mut &[u8]) -> ", stringify!($Xe), "<", stringify!($T), "> {")]
            #[doc = concat!("    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($Xe), "<", stringify!($T), ">>());")]
            ///     *input = rest;
            #[doc = concat!("    ", stringify!($Xe), "::<", stringify!($T), ">::from_be_bytes(int_bytes.try_into().unwrap())")]
            /// }
            /// ```
            #[must_use]
            #[inline]
            pub const fn from_be_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
                Self::from_ne(<$T>::from_be_bytes(bytes))
            }

            #[doc = concat!("Create a ", $order, " endian integer value from its representation as a byte array in ", $order_other, " endian.")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let value = ", stringify!($Xe), "::<", stringify!($T), ">::from_le_bytes(", le_bytes!($T), ");")]
            #[doc = concat!("assert_eq!(value.to_ne(), ", swap_op!($T), ");")]
            /// ```
            ///
            /// When starting from a slice rather than an array, fallible conversion APIs can be used:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("fn read_le_", stringify!($T), "(input: &mut &[u8]) -> ", stringify!($Xe), "<", stringify!($T), "> {")]
            #[doc = concat!("    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($Xe), "<", stringify!($T), ">>());")]
            ///     *input = rest;
            #[doc = concat!("    ", stringify!($Xe), "::<", stringify!($T), ">::from_le_bytes(int_bytes.try_into().unwrap())")]
            /// }
            /// ```
            #[must_use]
            #[inline]
            pub const fn from_le_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
                Self::from_ne(<$T>::from_le_bytes(bytes))
            }

            #[doc = concat!("Create a ", $order, " endian integer value from its memory representation as a byte array in native endianness.")]
            ///
            /// As the target platform's native endianness is used, portable code
            /// likely wants to use [`from_be_bytes`] or [`from_le_bytes`], as
            /// appropriate instead.
            ///
            /// [`from_be_bytes`]: Self::from_be_bytes
            /// [`from_le_bytes`]: Self::from_le_bytes
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let value = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne_bytes(if cfg!(target_endian = \"big\") {")]
            #[doc = concat!("    ", be_bytes!($T))]
            /// } else {
            #[doc = concat!("    ", le_bytes!($T))]
            /// });
            #[doc = concat!("assert_eq!(value.to_ne(), ", swap_op!($T), ");")]
            /// ```
            ///
            /// When starting from a slice rather than an array, fallible conversion APIs can be used:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("fn read_ne_", stringify!($T), "(input: &mut &[u8]) -> ", stringify!($Xe), "<", stringify!($T), "> {")]
            #[doc = concat!("    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($Xe), "<", stringify!($T), ">>());")]
            ///     *input = rest;
            #[doc = concat!("    ", stringify!($Xe), "::<", stringify!($T), ">::from_ne_bytes(int_bytes.try_into().unwrap())")]
            /// }
            /// ```
            #[must_use]
            #[inline]
            pub const fn from_ne_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
                Self::from_ne(<$T>::from_ne_bytes(bytes))
            }

            /// Shifts the bits to the left by a specified amount, `n`,
            /// wrapping the truncated bits to the end of the resulting integer.
            ///
            /// Please note this isn't the same operation as the `<<` shifting operator!
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", rot_op!($T), ");")]
            #[doc = concat!("let m = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", rot_result!($T), ");")]
            ///
            #[doc = concat!("assert_eq!(n.rotate_left(", rot!($T), "), m);")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn rotate_left(self, n: u32) -> Self {
                Self::from_ne(self.to_ne().rotate_left(n))
            }

            /// Shifts the bits to the right by a specified amount, `n`,
            /// wrapping the truncated bits to the beginning of the resulting
            /// integer.
            ///
            /// Please note this isn't the same operation as the `>>` shifting operator!
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", rot_result!($T), ");")]
            #[doc = concat!("let m = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", rot_op!($T), ");")]
            ///
            #[doc = concat!("assert_eq!(n.rotate_right(", rot!($T), "), m);")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn rotate_right(self, n: u32) -> Self {
                Self::from_ne(self.to_ne().rotate_right(n))
            }

            /// Reverses the byte order of the integer.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "(", swap_op!($T), stringify!($T), ");")]
            /// let m = n.swap_bytes();
            ///
            #[doc = concat!("assert_eq!(m, ", stringify!($Xe), "(", swapped!($T), "));")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn swap_bytes(self) -> Self {
                Self(self.0.swap_bytes())
            }

            /// Reverses the order of bits in the integer.
            ///
            /// The least significant bit becomes the most significant bit,
            /// second least-significant bit becomes second most-significant bit, etc.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "(", swap_op!($T), stringify!($T), ");")]
            /// let m = n.reverse_bits();
            ///
            #[doc = concat!("assert_eq!(m, ", stringify!($Xe), "(", reversed!($T), "));")]
            #[doc = concat!("assert_eq!(", stringify!($Xe), "(0), ", stringify!($Xe), "(0", stringify!($T), ").reverse_bits());")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn reverse_bits(self) -> Self {
                Self(self.0.reverse_bits())
            }

            /// Raises self to the power of `exp`, using exponentiation by squaring.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(2).pow(5).to_ne(), 32);")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn pow(self, exp: u32) -> Self {
                Self::from_ne(self.to_ne().pow(exp))
            }
        }

        impl From<$T> for $Xe<$T> {
            #[inline]
            fn from(value: $T) -> Self {
                Self::from_ne(value)
            }
        }

        impl From<$Xe<$T>> for $T {
            #[inline]
            fn from(value: $Xe<$T>) -> Self {
                value.to_ne()
            }
        }

        impl From<$Other<$T>> for $Xe<$T> {
            #[inline]
            fn from(value: $Other<$T>) -> Self {
                Self::$from_other(value)
            }
        }

        #[cfg(feature = "bitflags")]
        impl bitflags::Bits for $Xe<$T> {
            const EMPTY: Self = Self::MIN;

            const ALL: Self = Self::MAX;
        }

        #[cfg(feature = "bitflags")]
        impl bitflags::parser::ParseHex for $Xe<$T>
        {
            fn parse_hex(input: &str) -> Result<Self, bitflags::parser::ParseError> {
                <$T>::parse_hex(input).map($Xe::<$T>::from_ne)
            }
        }

        #[cfg(feature = "bitflags")]
        impl bitflags::parser::WriteHex for $Xe<$T>
        {
            fn write_hex<W: fmt::Write>(&self, writer: W) -> fmt::Result {
                self.to_ne().write_hex(writer)
            }
        }
    };
}

endian_int_impl! { Xe<u8> Xe<u16> Xe<u32> Xe<u64> Xe<u128> Xe<usize> Xe<i8> Xe<i16> Xe<i32> Xe<i64> Xe<i128> Xe<isize> }

macro_rules! endian_int_impl_signed {
    ($(Xe<$T:ty>)*) => {$(
        endian_int_impl_signed! { Be<$T> }
        endian_int_impl_signed! { Le<$T> }
    )*};
    ($Xe:ident<$T:ty>) => {
        unop_impl! { impl Neg, neg for $Xe<$T> }
        forward_ref_unop! { impl Neg, neg for $Xe<$T> }

        impl $Xe<$T> {
            /// Computes the absolute value of self.
            ///
            #[doc = concat!("See [`", stringify!($T), "::abs`]")]
            /// for documentation on overflow behavior.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(10).abs(), ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(10));")]
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(-10).abs(), ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(10));")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn abs(self) -> Self {
                Self::from_ne(self.to_ne().abs())
            }

            /// Returns a number representing sign of `self`.
            ///
            ///  - `0` if the number is zero
            ///  - `1` if the number is positive
            ///  - `-1` if the number is negative
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(10).signum().to_ne(), 1);")]
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(0).signum().to_ne(), 0);")]
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(-10).signum().to_ne(), -1);")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn signum(self) -> Self {
                Self::from_ne(self.to_ne().signum())
            }

            /// Returns `true` if `self` is positive and `false` if the number is zero or
            /// negative.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(10).is_positive());")]
            #[doc = concat!("assert!(!", stringify!($Xe), "::<", stringify!($T), ">::from_ne(-10).is_positive());")]
            /// ```
            #[must_use]
            #[inline]
            pub const fn is_positive(self) -> bool {
                self.to_ne().is_positive()
            }

            /// Returns `true` if `self` is negative and `false` if the number is zero or
            /// positive.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(-10).is_negative());")]
            #[doc = concat!("assert!(!", stringify!($Xe), "::<", stringify!($T), ">::from_ne(10).is_negative());")]
            /// ```
            #[must_use]
            #[inline]
            pub const fn is_negative(self) -> bool {
                self.to_ne().is_negative()
            }

            /// Returns the number of ones in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "(0b100_0000", stringify!($T), ");")]
            ///
            /// assert_eq!(n.count_ones(), 1);
            /// ```
            #[doc(alias = "popcount")]
            #[doc(alias = "popcnt")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn count_ones(self) -> u32 {
                self.0.count_ones()
            }

            /// Returns the number of zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::MAX.count_zeros(), 1);")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn count_zeros(self) -> u32 {
                self.0.count_zeros()
            }

            /// Returns the number of leading zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(-1);")]
            ///
            /// assert_eq!(n.leading_zeros(), 0);
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn leading_zeros(self) -> u32 {
                self.to_ne().leading_zeros()
            }

            /// Returns the number of trailing zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(-4);")]
            ///
            /// assert_eq!(n.trailing_zeros(), 2);
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn trailing_zeros(self) -> u32 {
                self.to_ne().trailing_zeros()
            }
        }
    };
}

endian_int_impl_signed! { Xe<i8> Xe<i16> Xe<i32> Xe<i64> Xe<i128> Xe<isize> }

macro_rules! endian_int_impl_unsigned {
    ($(Xe<$T:ty>)*) => {$(
        endian_int_impl_unsigned! { Be<$T> }
        endian_int_impl_unsigned! { Le<$T> }
    )*};
    ($Xe:ident<$T:ty>) => {
        impl $Xe<$T> {
            /// Returns the number of ones in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "(0b01001100", stringify!($T), ");")]
            ///
            /// assert_eq!(n.count_ones(), 3);
            /// ```
            #[doc(alias = "popcount")]
            #[doc(alias = "popcnt")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn count_ones(self) -> u32 {
                self.0.count_ones()
            }

            /// Returns the number of zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert_eq!(", stringify!($Xe), "::<", stringify!($T), ">::MAX.count_zeros(), 0);")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn count_zeros(self) -> u32 {
                self.0.count_zeros()
            }

            /// Returns the number of leading zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(", stringify!($T), "::MAX >> 2);")]
            ///
            /// assert_eq!(n.leading_zeros(), 2);
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn leading_zeros(self) -> u32 {
                self.to_ne().leading_zeros()
            }

            /// Returns the number of trailing zeros in the binary representation of `self`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("let n = ", stringify!($Xe), "::<", stringify!($T), ">::from_ne(0b0101000);")]
            ///
            /// assert_eq!(n.trailing_zeros(), 3);
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn trailing_zeros(self) -> u32 {
                self.to_ne().trailing_zeros()
            }

            /// Returns `true` if and only if `self == 2^k` for some `k`.
            ///
            /// # Examples
            ///
            /// Basic usage:
            ///
            /// ```
            #[doc = concat!("use endian_num::", stringify!($Xe), ";")]
            ///
            #[doc = concat!("assert!(", stringify!($Xe), "::<", stringify!($T), ">::from_ne(16).is_power_of_two());")]
            #[doc = concat!("assert!(!", stringify!($Xe), "::<", stringify!($T), ">::from_ne(10).is_power_of_two());")]
            /// ```
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn is_power_of_two(self) -> bool {
                self.to_ne().is_power_of_two()
            }
        }
    };
}

endian_int_impl_unsigned! { Xe<u8> Xe<u16> Xe<u32> Xe<u64> Xe<u128> Xe<usize> }

macro_rules! impl_from_array {
    (Xe<$Large:ty>, [Xe<$Small:ty>; $i:literal]) => {
        impl_from_array!(Be<$Large>, [Be<$Small>; $i], "big");
        impl_from_array!(Le<$Large>, [Le<$Small>; $i], "little");
    };
    ($Large:ty, [$Small:ty; $i:literal], $order:literal) => {
        impl From<[$Small; $i]> for $Large {
            #[doc = concat!("Create an integer from its representation as a [`", stringify!($Small), "`] array in ", $order, " endian.")]
            #[inline]
            fn from(value: [$Small; $i]) -> Self {
                // SAFETY: integers are plain old datatypes so we can always transmute to them
                unsafe { mem::transmute(value) }
            }
        }

        impl From<$Large> for [$Small; $i] {
            #[doc = concat!("Return the memory representation of this integer as a [`", stringify!($Small), "`] array in ", $order, "-endian byte order.")]
            #[inline]
            fn from(value: $Large) -> Self {
                // SAFETY: integers are plain old datatypes so we can always transmute them to
                // arrays of bytes
                unsafe { mem::transmute(value) }
            }
        }
    };
}

impl_from_array!(Xe<u16>, [Xe<u8>; 2]);
impl_from_array!(Xe<u32>, [Xe<u8>; 4]);
impl_from_array!(Xe<u32>, [Xe<u16>; 2]);
impl_from_array!(Xe<u64>, [Xe<u8>; 8]);
impl_from_array!(Xe<u64>, [Xe<u16>; 4]);
impl_from_array!(Xe<u64>, [Xe<u32>; 2]);
impl_from_array!(Xe<u128>, [Xe<u8>; 16]);
impl_from_array!(Xe<u128>, [Xe<u16>; 8]);
impl_from_array!(Xe<u128>, [Xe<u32>; 4]);
impl_from_array!(Xe<u128>, [Xe<u64>; 2]);

macro_rules! type_alias {
    (#[$cfg:meta], $alias:ident = $SelfT:ty, $bits:expr, $order:expr) => {
        #[doc = concat!("A ", stringify!($bits), "-bit unsigned integer stored in ", $order, "-endian byte order.")]
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// ```
        #[doc = concat!("use endian_num::", stringify!($alias), ";")]
        ///
        #[doc = concat!("let n = 0x1A;")]
        ///
        #[doc = concat!("if cfg!(target_endian = \"", $order, "\") {")]
        #[doc = concat!("    assert_eq!(", stringify!($alias), "::from_ne(n).0, n);")]
        /// } else {
        #[doc = concat!("    assert_eq!(", stringify!($alias), "::from_ne(n).0, n.swap_bytes());")]
        /// }
        /// ```
        #[$cfg]
        #[allow(non_camel_case_types)]
        pub type $alias = $SelfT;
    };
}

type_alias!(#[cfg(feature = "linux-types")], be16 = Be<u16>, 16, "big");
type_alias!(#[cfg(feature = "linux-types")], be32 = Be<u32>, 32, "big");
type_alias!(#[cfg(feature = "linux-types")], be64 = Be<u64>, 64, "big");
type_alias!(#[cfg(feature = "linux-types")], be128 = Be<u128>, 128, "big");
type_alias!(#[cfg(feature = "linux-types")], le16 = Le<u16>, 16, "little");
type_alias!(#[cfg(feature = "linux-types")], le32 = Le<u32>, 32, "little");
type_alias!(#[cfg(feature = "linux-types")], le64 = Le<u64>, 64, "little");
type_alias!(#[cfg(feature = "linux-types")], le128 = Le<u128>, 128, "little");
