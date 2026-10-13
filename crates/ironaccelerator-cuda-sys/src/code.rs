//! Integer codes that a vendor library *writes* (status returns, out-params,
//! fields it fills in).
//!
//! These are `#[repr(transparent)]` newtypes over `u32`, not Rust enums: a C
//! library may return a value the bindings do not model — a newer release, a
//! code the header added — and materialising that as a Rust `enum` is
//! undefined behaviour. A newtype holds any value. The known codes are
//! associated constants, so `CUresult::Success`, `==` comparisons, and `match`
//! arms (with a `_` fallback) read the same as an enum would.

/// Declare a C status/code newtype with named associated constants.
macro_rules! ffi_code {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident = $value:expr, )*
        }
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Copy, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub u32);

        #[allow(non_upper_case_globals)]
        impl $name {
            $( $(#[$vmeta])* pub const $variant: Self = Self(($value) as u32); )*

            /// Wrap a raw code. Every `u32` is a valid value.
            #[inline]
            pub const fn from_raw(raw: u32) -> Self {
                Self(raw)
            }

            /// The raw code.
            #[inline]
            pub const fn raw(self) -> u32 {
                self.0
            }

            /// Name of the constant this code matches, or `None` for a code
            /// the bindings do not model.
            pub fn name(self) -> Option<&'static str> {
                $( if self.0 == (($value) as u32) { return Some(stringify!($variant)); } )*
                None
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                match self.name() {
                    Some(n) => write!(f, "{}::{}", stringify!($name), n),
                    None => write!(f, "{}({})", stringify!($name), self.0),
                }
            }
        }
    };
}
