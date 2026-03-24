//! Specialized binary input and output.
//! Uses the error handling for this crate.

#![doc(hidden)]
use alloc::vec::Vec;
use core::convert::TryFrom;

use ::half::f16;
use half::slice::HalfFloatSliceExt;
use lebe::prelude::*;
use no_std_io::io::{Cursor, ErrorKind as IoErrorKind};
pub use no_std_io::io::{Read, Seek, SeekFrom, Write};
use smallvec::{Array, SmallVec};

use crate::error::{Error, IoError, IoResult, Result, UnitResult};

/// A buffered reader with a compile-time buffer size.
#[derive(Debug)]
pub struct BufReader<R: Read, const S: usize = 8192> {
    inner: R,
    buf: [u8; S],
    /// bytes in `buf` that have been filled but not yet consumed
    filled: usize,
    /// read position within `buf`
    consumed: usize,
}

impl<R: Read, const S: usize> BufReader<R, S> {
    /// Wrap `inner` in a `BufReader`.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            buf: [0u8; S],
            filled: 0,
            consumed: 0,
        }
    }
}

impl<R: Read, const S: usize> Read for BufReader<R, S> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        if self.consumed >= self.filled {
            if buf.len() >= S {
                // bypass internal buffer for large reads
                return self.inner.read(buf);
            }
            self.filled = self.inner.read(&mut self.buf)?;
            self.consumed = 0;
            if self.filled == 0 {
                return Ok(0);
            }
        }
        let available = &self.buf[self.consumed..self.filled];
        let n = available.len().min(buf.len());
        buf[..n].copy_from_slice(&available[..n]);
        self.consumed += n;
        Ok(n)
    }
}

impl<R: Read + Seek, const S: usize> Seek for BufReader<R, S> {
    fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
        // discard buffer on any seek
        self.filled = 0;
        self.consumed = 0;
        self.inner.seek(pos)
    }
}

/// A cursor over a `Vec<u8>` that implements `Write`.
/// Needed because `no_std_io` does not provide `impl Write for
/// Cursor<Vec<u8>>`.
#[derive(Debug)]
pub struct WriteCursor(Cursor<Vec<u8>>);

impl WriteCursor {
    /// Create a new `WriteCursor` wrapping the given `Vec<u8>`.
    pub fn new(v: Vec<u8>) -> Self {
        WriteCursor(Cursor::new(v))
    }

    /// Return the current byte position of the cursor.
    pub fn position(&self) -> u64 {
        self.0.position()
    }

    /// Set the cursor position.
    pub fn set_position(&mut self, pos: u64) {
        self.0.set_position(pos);
    }

    /// Consume the cursor and return the underlying `Vec<u8>`.
    pub fn into_inner(self) -> Vec<u8> {
        self.0.into_inner()
    }
}

impl Seek for WriteCursor {
    fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
        self.0.seek(pos)
    }
}

impl Write for WriteCursor {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let pos = self.0.position() as usize;
        let inner = self.0.get_mut();
        let end = pos + buf.len();
        if inner.len() < end {
            inner.resize(end, 0);
        }
        inner[pos..end].copy_from_slice(buf);
        self.0.set_position(end as u64);
        Ok(buf.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

/// A buffered writer with a compile-time buffer size.
/// Flushes the internal buffer before any seek, ensuring correct behaviour
/// when the underlying writer requires seeking (e.g. EXR chunk-offset
/// patching).
#[derive(Debug)]
pub struct BufWriter<W: Write + Seek, const S: usize = 8192> {
    inner: W,
    buf: [u8; S],
    filled: usize,
}

impl<W: Write + Seek, const S: usize> BufWriter<W, S> {
    /// Wrap `inner` in a `BufWriter`.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            buf: [0u8; S],
            filled: 0,
        }
    }

    fn flush_buf(&mut self) -> IoResult<()> {
        if self.filled > 0 {
            self.inner.write_all(&self.buf[..self.filled])?;
            self.filled = 0;
        }
        Ok(())
    }
}

impl<W: Write + Seek, const S: usize> Write for BufWriter<W, S> {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        if self.filled + buf.len() >= S {
            self.flush_buf()?;
            if buf.len() >= S {
                return self.inner.write(buf);
            }
        }
        self.buf[self.filled..self.filled + buf.len()].copy_from_slice(buf);
        self.filled += buf.len();
        Ok(buf.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        self.flush_buf()?;
        self.inner.flush()
    }
}

impl<W: Write + Seek, const S: usize> Seek for BufWriter<W, S> {
    fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
        self.flush_buf()?;
        self.inner.seek(pos)
    }
}

/// Skip reading uninteresting bytes without allocating.
#[inline]
pub fn skip_bytes(read: &mut impl Read, count: usize) -> IoResult<()> {
    let mut buf = [0u8; 64];
    let mut remaining = count;
    while remaining > 0 {
        let to_read = remaining.min(buf.len());
        let n = read.read(&mut buf[..to_read])?;
        if n == 0 {
            return Err(IoError::from(IoErrorKind::UnexpectedEof));
        }
        remaining -= n;
    }
    Ok(())
}

/// If an error occurs while writing, attempts to delete the partially written
/// file. Creates a file just before the first write operation, not when this
/// function is called.
#[cfg(feature = "std")]
#[inline]
pub fn attempt_delete_file_on_write_error<'p>(
    path: &'p ::std::path::Path,
    write: impl FnOnce(LateFile<'p>) -> UnitResult,
) -> UnitResult {
    match write(LateFile::from(path)) {
        Err(error) => {
            // FIXME deletes existing file if creation of new file fails?
            let _deleted = ::std::fs::remove_file(path); // ignore deletion errors
            Err(error)
        }

        ok => ok,
    }
}

#[cfg(feature = "std")]
#[derive(Debug)]
pub struct LateFile<'p> {
    path: &'p ::std::path::Path,
    file: Option<::std::fs::File>,
}

#[cfg(feature = "std")]
impl<'p> From<&'p ::std::path::Path> for LateFile<'p> {
    fn from(path: &'p ::std::path::Path) -> Self {
        Self {
            path,
            file: None,
        }
    }
}

#[cfg(feature = "std")]
impl LateFile<'_> {
    fn file(&mut self) -> ::std::io::Result<&mut ::std::fs::File> {
        if self.file.is_none() {
            self.file = Some(::std::fs::File::create(self.path)?);
        }
        Ok(self.file.as_mut().unwrap()) // will not be reached if creation fails
    }
}

#[cfg(feature = "std")]
impl ::std::io::Write for LateFile<'_> {
    fn write(&mut self, buffer: &[u8]) -> ::std::io::Result<usize> {
        self.file()?.write(buffer)
    }

    fn flush(&mut self) -> ::std::io::Result<()> {
        if let Some(file) = &mut self.file {
            file.flush()
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "std")]
impl Seek for LateFile<'_> {
    fn seek(&mut self, position: SeekFrom) -> ::std::io::Result<u64> {
        self.file()?.seek(position)
    }
}

/// Peek a single byte without consuming it.
#[derive(Debug)]
pub struct PeekRead<T> {
    /// Cannot be exposed as it will not contain peeked values anymore.
    inner: T,

    peeked: Option<IoResult<u8>>,
}

impl<T: Read> PeekRead<T> {
    /// Wrap a reader to make it peekable.
    #[inline]
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            peeked: None,
        }
    }

    /// Read a single byte and return that without consuming it.
    /// The next `read` call will include that byte.
    #[inline]
    pub fn peek_u8(&mut self) -> &IoResult<u8> {
        self.peeked =
            self.peeked.take().or_else(|| Some(u8::read_from_little_endian(&mut self.inner)));
        self.peeked.as_ref().unwrap() // unwrap cannot fail because we just set
                                      // it
    }

    /// Skip a single byte if it equals the specified value.
    /// Returns whether the value was found.
    /// Consumes the peeked result if an error occurred.
    #[inline]
    pub fn skip_if_eq(&mut self, value: u8) -> IoResult<bool> {
        match self.peek_u8() {
            Ok(peeked) if *peeked == value => {
                self.peeked = None; // consume the byte
                Ok(true)
            }

            Ok(_) => Ok(false),

            // return the error otherwise.
            // unwrap is safe because this branch cannot be reached otherwise.
            // we need to take() from self because io errors cannot be cloned.
            Err(_) => Err(self.peeked.take().unwrap().err().unwrap()),
        }
    }
}

impl<T: Read> Read for PeekRead<T> {
    fn read(&mut self, target_buffer: &mut [u8]) -> IoResult<usize> {
        if target_buffer.is_empty() {
            return Ok(0);
        }

        match self.peeked.take() {
            None => self.inner.read(target_buffer),
            Some(peeked) => {
                target_buffer[0] = peeked?;

                // indexing [1..] is safe because an empty buffer already returned ok
                Ok(1 + self.inner.read(&mut target_buffer[1..])?)
            }
        }
    }
}

impl<T: Read + Seek> PeekRead<Tracking<T>> {
    /// Seek this read to the specified byte position.
    /// Discards any previously peeked value.
    pub fn skip_to(&mut self, position: usize) -> IoResult<()> {
        self.inner.seek_read_to(position)?;
        self.peeked = None;
        Ok(())
    }
}

impl<T: Read> PeekRead<Tracking<T>> {
    /// Current number of bytes read.
    pub const fn byte_position(&self) -> usize {
        self.inner.byte_position()
    }
}

/// Keep track of what byte we are at.
/// Used to skip back to a previous place after writing some information.
#[derive(Debug)]
pub struct Tracking<T> {
    /// Do not expose to prevent seeking without updating position
    inner: T,

    position: usize,
}

impl<T: Read> Read for Tracking<T> {
    fn read(&mut self, buffer: &mut [u8]) -> IoResult<usize> {
        let count = self.inner.read(buffer)?;
        self.position += count;
        Ok(count)
    }
}

impl<T: Write> Write for Tracking<T> {
    fn write(&mut self, buffer: &[u8]) -> IoResult<usize> {
        let count = self.inner.write(buffer)?;
        self.position += count;
        Ok(count)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.inner.flush()
    }
}

impl<T> Tracking<T> {
    /// If `inner` is a reference, if must never be seeked directly,
    /// but only through this `Tracking` instance.
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            position: 0,
        }
    }

    /// Current number of bytes written or read.
    pub const fn byte_position(&self) -> usize {
        self.position
    }
}

impl<T: Read + Seek> Tracking<T> {
    /// Set the reader to the specified byte position.
    /// If it is only a couple of bytes, no seek system call is performed.
    pub fn seek_read_to(&mut self, target_position: usize) -> IoResult<()> {
        let delta = target_position as i128 - self.position as i128; // FIXME  panicked at 'attempt to subtract with overflow'
        debug_assert!(delta.abs() < usize::MAX as i128);

        if delta > 0 && delta < 16 {
            // TODO profile that this is indeed faster than a syscall! (should be because of
            // bufread buffer discard)
            skip_bytes(self, delta as usize)?;
            self.position += delta as usize;
        } else if delta != 0 {
            self.inner.seek(SeekFrom::Start(u64::try_from(target_position).unwrap()))?;
            self.position = target_position;
        }

        Ok(())
    }
}

impl<T: Write + Seek> Tracking<T> {
    /// Move the writing cursor to the specified target byte index.
    /// If seeking forward, this will write zeroes.
    pub fn seek_write_to(&mut self, target_position: usize) -> IoResult<()> {
        if target_position < self.position {
            self.inner.seek(SeekFrom::Start(u64::try_from(target_position).unwrap()))?;
        } else if target_position > self.position {
            let zero_buf = [0u8; 64];
            let mut remaining = target_position - self.position;
            while remaining > 0 {
                let to_write = remaining.min(zero_buf.len());
                let n = self.write(&zero_buf[..to_write])?;
                if n == 0 {
                    break;
                }
                remaining -= n;
            }
        }

        self.position = target_position;
        Ok(())
    }
}

/// Generic trait that defines common binary operations such as reading and
/// writing for this type.
pub trait Data: Sized + Default + Clone {
    /// Number of bytes this would consume in an exr file.
    const BYTE_SIZE: usize = ::core::mem::size_of::<Self>();

    /// Read a value of type `Self` from a little-endian source.
    fn read_le(read: &mut impl Read) -> Result<Self>;

    /// Read a value of type `Self` from a **native-endian** source (no
    /// conversion).
    fn read_ne(read: &mut impl Read) -> Result<Self>;

    /// Read as many values of type `Self` as fit into the specified slice, from
    /// a little-endian source. If the slice cannot be filled completely,
    /// returns `Error::Invalid`.
    fn read_slice_le(read: &mut impl Read, slice: &mut [Self]) -> UnitResult;

    /// Read as many values of type `Self` as fit into the specified slice, from
    /// a **native-endian** source (no conversion). If the slice cannot be
    /// filled completely, returns `Error::Invalid`.
    fn read_slice_ne(read: &mut impl Read, slice: &mut [Self]) -> UnitResult;

    /// Read as many values of type `Self` as specified with `data_size`.
    ///
    /// This method will not allocate more memory than `soft_max` at once.
    /// If `hard_max` is specified, it will never read any more than that.
    /// Returns `Error::Invalid` if the reader does not contain the desired
    /// number of elements. Reads from little-endian byte source.
    #[inline]
    fn read_vec_le(
        read: &mut impl Read,
        data_size: usize,
        soft_max: usize,
        hard_max: Option<usize>,
        purpose: &'static str,
    ) -> Result<Vec<Self>> {
        if let Some(max) = hard_max {
            if data_size > max {
                return Err(Error::invalid(purpose));
            }
        }

        let mut vec = Vec::with_capacity(data_size.min(soft_max));
        Self::read_into_vec_le(read, &mut vec, data_size, soft_max, hard_max, purpose)?;
        Ok(vec)
    }

    /// Write this value to the writer, converting to little-endian format.
    fn write_le(self, write: &mut impl Write) -> UnitResult;

    /// Write this value to the writer, in **native-endian** format (no
    /// conversion).
    fn write_ne(self, write: &mut impl Write) -> UnitResult;

    /// Write all values of that slice to the writer, converting to
    /// little-endian format.
    fn write_slice_le(write: &mut impl Write, slice: &[Self]) -> UnitResult;

    /// Write all values of that slice to the writer, in **native-endian**
    /// format (no conversion).
    fn write_slice_ne(write: &mut impl Write, slice: &[Self]) -> UnitResult;

    /// Read as many values of type `Self` as specified with `data_size` into
    /// the provided vector.
    ///
    /// This method will not allocate more memory than `soft_max` at once.
    /// If `hard_max` is specified, it will never read any more than that.
    /// Returns `Error::Invalid` if reader does not contain the desired number
    /// of elements.
    #[inline]
    fn read_into_vec_le(
        read: &mut impl Read,
        data: &mut impl ResizableVec<Self>,
        data_size: usize,
        soft_max: usize,
        hard_max: Option<usize>,
        purpose: &'static str,
    ) -> UnitResult {
        if let Some(max) = hard_max {
            if data_size > max {
                return Err(Error::invalid(purpose));
            }
        }

        let soft_max = hard_max.unwrap_or(soft_max).min(soft_max);
        let end = data.len() + data_size;

        // do not allocate more than $chunks memory at once
        // (most of the time, this loop will run only once)
        while data.len() < end {
            let chunk_start = data.len();
            let chunk_end = (chunk_start + soft_max).min(data_size);

            data.resize(chunk_end, Self::default());
            Self::read_slice_le(read, &mut data.as_mut()[chunk_start..chunk_end])?;
            // safe because of `min(data_size)`
        }

        Ok(())
    }

    /// Write the length of the slice and then its contents, converting to
    /// little-endian format.
    #[inline]
    fn write_i32_sized_slice_le<W: Write>(write: &mut W, slice: &[Self]) -> UnitResult {
        i32::try_from(slice.len())?.write_le(write)?;
        Self::write_slice_le(write, slice)
    }

    /// Read the desired element count and then read that many items into a
    /// vector.
    ///
    /// This method will not allocate more memory than `soft_max` at once.
    /// If `hard_max` is specified, it will never read any more than that.
    /// Returns `Error::Invalid` if reader does not contain the desired number
    /// of elements.
    #[inline]
    fn read_i32_sized_vec_le(
        read: &mut impl Read,
        soft_max: usize,
        hard_max: Option<usize>,
        purpose: &'static str,
    ) -> Result<Vec<Self>> {
        let size = usize::try_from(i32::read_le(read)?)?;
        Self::read_vec_le(read, size, soft_max, hard_max, purpose)
    }

    /// Fill the slice with this value.
    #[inline]
    fn fill_slice(self, slice: &mut [Self])
    where
        Self: Copy,
    {
        // hopefully compiles down to a single memset call
        for value in slice {
            *value = self;
        }
    }
}

/// A unifying trait that is implemented for Vec and `SmallVec`,
/// focused on resizing capabilities.
pub trait ResizableVec<T>: AsMut<[T]> {
    fn resize(&mut self, new_len: usize, value: T);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Clone> ResizableVec<T> for Vec<T> {
    fn resize(&mut self, new_len: usize, value: T) {
        Self::resize(self, new_len, value);
    }

    fn len(&self) -> usize {
        Self::len(self)
    }
}

impl<T: Clone, A: Array<Item = T>> ResizableVec<T> for SmallVec<A> {
    fn resize(&mut self, new_len: usize, value: T) {
        Self::resize(self, new_len, value);
    }

    fn len(&self) -> usize {
        Self::len(self)
    }
}

macro_rules! implement_data_for_primitive {
    ($kind: ident) => {
        impl Data for $kind {
            #[inline]
            fn read_le(read: &mut impl Read) -> Result<Self> {
                Ok(read.read_from_little_endian()?)
            }

            #[inline]
            fn read_ne(read: &mut impl Read) -> Result<Self> {
                Ok(read.read_from_native_endian()?)
            }

            #[inline]
            fn write_le(self, write: &mut impl Write) -> Result<()> {
                write.write_as_little_endian(&self)?;
                Ok(())
            }

            #[inline]
            fn write_ne(self, write: &mut impl Write) -> Result<()> {
                write.write_as_native_endian(&self)?;
                Ok(())
            }

            #[inline]
            fn read_slice_le(read: &mut impl Read, slice: &mut [Self]) -> Result<()> {
                read.read_from_little_endian_into(slice)?;
                Ok(())
            }

            #[inline]
            fn read_slice_ne(read: &mut impl Read, slice: &mut [Self]) -> Result<()> {
                read.read_from_native_endian_into(slice)?;
                Ok(())
            }

            #[inline]
            fn write_slice_le(write: &mut impl Write, slice: &[Self]) -> Result<()> {
                write.write_as_little_endian(slice)?;
                Ok(())
            }

            #[inline]
            fn write_slice_ne(write: &mut impl Write, slice: &[Self]) -> Result<()> {
                write.write_as_native_endian(slice)?;
                Ok(())
            }
        }
    };
}

implement_data_for_primitive!(u8);
implement_data_for_primitive!(i8);
implement_data_for_primitive!(i16);
implement_data_for_primitive!(u16);
implement_data_for_primitive!(u32);
implement_data_for_primitive!(i32);
implement_data_for_primitive!(i64);
implement_data_for_primitive!(u64);
implement_data_for_primitive!(f32);
implement_data_for_primitive!(f64);

impl Data for f16 {
    #[inline]
    fn read_le(read: &mut impl Read) -> Result<Self> {
        u16::read_le(read).map(Self::from_bits)
    }

    #[inline]
    fn read_ne(read: &mut impl Read) -> Result<Self> {
        u16::read_ne(read).map(Self::from_bits)
    }

    #[inline]
    fn read_slice_le(read: &mut impl Read, slice: &mut [Self]) -> Result<()> {
        let bits_mut = slice.reinterpret_cast_mut();
        u16::read_slice_le(read, bits_mut)
    }

    #[inline]
    fn read_slice_ne(read: &mut impl Read, slice: &mut [Self]) -> Result<()> {
        let bits_mut = slice.reinterpret_cast_mut();
        u16::read_slice_ne(read, bits_mut)
    }

    #[inline]
    fn write_le(self, write: &mut impl Write) -> Result<()> {
        self.to_bits().write_le(write)
    }

    #[inline]
    fn write_ne(self, write: &mut impl Write) -> Result<()> {
        self.to_bits().write_ne(write)
    }

    #[inline]
    fn write_slice_le(write: &mut impl Write, slice: &[Self]) -> Result<()> {
        let bits = slice.reinterpret_cast();
        u16::write_slice_le(write, bits)
    }

    #[inline]
    fn write_slice_ne(write: &mut impl Write, slice: &[Self]) -> Result<()> {
        let bits = slice.reinterpret_cast();
        u16::write_slice_ne(write, bits)
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use std::io::Read;

    use crate::io::PeekRead;

    #[test]
    fn peek() {
        use lebe::prelude::*;
        let buffer: &[u8] = &[0, 1, 2, 3];
        let mut peek = PeekRead::new(buffer);

        assert_eq!(peek.peek_u8().as_ref().unwrap(), &0);
        assert_eq!(peek.peek_u8().as_ref().unwrap(), &0);
        assert_eq!(peek.peek_u8().as_ref().unwrap(), &0);
        assert_eq!(u8::read_from_little_endian(&mut peek).unwrap(), 0_u8);

        assert_eq!(peek.read(&mut [0, 0]).unwrap(), 2);

        assert_eq!(peek.peek_u8().as_ref().unwrap(), &3);
        assert_eq!(u8::read_from_little_endian(&mut peek).unwrap(), 3_u8);

        assert!(peek.peek_u8().is_err());
        assert!(peek.peek_u8().is_err());
        assert!(peek.peek_u8().is_err());
        assert!(peek.peek_u8().is_err());

        assert!(u8::read_from_little_endian(&mut peek).is_err());
    }
}
