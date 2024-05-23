//! Based on <https://github.com/rust-lang/rust/blob/1.78.0/library/core/src/internal_macros.rs>.

// implements the unary operator "op &T"
// based on "op T" where T is expected to be `Copy`able
macro_rules! forward_ref_unop {
    (impl $Trait:ident, $method:ident for $T:ty) => {
        impl $Trait for &$T {
            type Output = <$T as $Trait>::Output;

            #[inline]
            #[track_caller]
            fn $method(self) -> <$T as $Trait>::Output {
                $Trait::$method(*self)
            }
        }
    };
}

// implements binary operators "&T op Rhs", "T op &Rhs", "&T op &Rhs"
// based on "T op Rhs" where T and Rhs are expected to be `Copy`able
macro_rules! forward_ref_binop {
    (impl $Trait:ident<$Rhs:ty>, $method:ident for $T:ty) => {
        impl<'a> $Trait<$Rhs> for &'a $T {
            type Output = <$T as $Trait<$Rhs>>::Output;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: $Rhs) -> <$T as $Trait<$Rhs>>::Output {
                $Trait::$method(*self, rhs)
            }
        }

        impl $Trait<&$Rhs> for $T {
            type Output = <$T as $Trait<$Rhs>>::Output;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: &$Rhs) -> <$T as $Trait<$Rhs>>::Output {
                $Trait::$method(self, *rhs)
            }
        }

        impl $Trait<&$Rhs> for &$T {
            type Output = <$T as $Trait<$Rhs>>::Output;

            #[inline]
            #[track_caller]
            fn $method(self, rhs: &$Rhs) -> <$T as $Trait<$Rhs>>::Output {
                $Trait::$method(*self, *rhs)
            }
        }
    };
}

// implements "T op= &Rhs", based on "T op= Rhs"
// where Rhs is expected to be `Copy`able
macro_rules! forward_ref_op_assign {
    (impl $Trait:ident<$Rhs:ty>, $method:ident for $T:ty) => {
        impl $Trait<&$Rhs> for $T {
            #[inline]
            #[track_caller]
            fn $method(&mut self, rhs: &$Rhs) {
                $Trait::$method(self, *rhs);
            }
        }
    };
}
