# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Bagz file reader/writer and PyGrain-compatible data source for POSIX systems.

Bagz is a file format for storing a sequence of string records, typically
serialised protocol buffers. It supports fast index based look-up.
"""

import bisect
from collections.abc import Sequence
import itertools
import mmap
import os
import re
import shutil
import struct
from typing import Any, SupportsIndex

from etils import epath
from typing_extensions import Self
import zstandard as zstd


class BagFileReader(Sequence[bytes]):
  """Reader for single Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      decompress: bool | None = None,
  ) -> None:
    """Creates a BagFileReader.

    Args:
      filename: The name of the single Bagz file to read.
      separate_limits: Whether the limits are stored in a separate file.
      decompress: Whether to decompress the records. If None, uses the file
        extension to determine whether to decompress.
    """
    if decompress or (decompress is None and filename.endswith('.bagz')):
      self._process = lambda x: zstd.decompress(x) if x else x
    else:
      self._process = lambda x: x
    self._filename = filename
    fd = os.open(filename, os.O_RDONLY)
    try:
      self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
      file_size = self._records.size()
    except ValueError:
      self._records = b''
      file_size = 0
    finally:
      os.close(fd)
    if separate_limits:
      directory, name = os.path.split(filename)
      fd = os.open(os.path.join(directory, 'limits.' + name), os.O_RDONLY)
      try:
        self._limits = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        index_size = self._limits.size()
      except ValueError:
        self._limits = b''
        index_size = 0
      finally:
        os.close(fd)
      index_start = 0
    else:
      if 0 < file_size < 8:
        raise ValueError('Bagz file too small')
      self._limits = self._records
      if file_size:
        (index_start,) = struct.unpack('<Q', self._records[-8:])
      else:
        index_start = 0
      assert file_size >= index_start
      index_size = file_size - index_start
    assert index_size % 8 == 0
    self._num_records = index_size // 8
    self._limits_start = index_start

  def __len__(self) -> int:
    """Returns the number of records in the Bagz file."""
    return self._num_records

  def __getitem__(self, index: SupportsIndex) -> bytes:
    """Returns a record from the Bagz file."""
    i = index.__index__()
    if not 0 <= i < self._num_records:
      raise IndexError('bagz.BragReader index out of range')
    end = i * 8 + self._limits_start
    if i:
      rec_range = struct.unpack('<2q', self._limits[end - 8 : end + 8])
    else:
      rec_range = (0, *struct.unpack('<q', self._limits[end : end + 8]))
    return self._process(self._records[slice(*rec_range)])


class BagShardReader(Sequence[bytes]):
  """Reader for sharded Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      decompress: bool | None = None,
  ) -> None:
    """Creates a BagShardReader.

    Args:
      filename: The name of the sharded Bagz file to read.
      separate_limits: Whether the limits are stored in a separate file.
      decompress: Whether to decompress the records. If None, uses the file
        extension to determine whether to decompress.
    """
    matches = re.findall(r'@(\d+)', filename)
    assert len(matches) == 1
    num_files = int(matches[0])
    assert num_files < 100_000
    self._bags = tuple(
        BagFileReader(
            filename=re.sub(
                r'@(\d+)', f'-{idx:05d}-of-{num_files:05d}', filename
            ),
            separate_limits=separate_limits,
            decompress=decompress,
        )
        for idx in range(num_files)
    )
    self._accum = tuple(itertools.accumulate(map(len, self._bags)))

  def __len__(self) -> int:
    """Returns the number of records in the Bagz file."""
    return self._accum[-1]

  def __getitem__(self, index: int) -> bytes:
    if index < 0:
      index += self._accum[-1]
    if seqn := bisect.bisect_left(self._accum, index + 1):
      index -= self._accum[seqn - 1]
    return self._bags[seqn][index]


class BagReader(Sequence[bytes]):
  """Reader for Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      decompress: bool | None = None,
  ) -> None:
    """Creates a BagReader.

    Args:
      filename: The name of the Bagz file to read. Supports the @N shard syntax
        (where @0 corresponds to the single file case). If the shard syntax does
        not parse, then `filename` is treated as a single file.
      separate_limits: Whether the limits are stored in a separate file.
      decompress: Whether to decompress the records. If None, uses the file
        extension to determine whether to decompress.
    """
    if matches := re.findall(r'@(\d+)', filename):
      assert len(matches) == 1
      if int(matches[0]) != '0':
        reader_class = BagShardReader
      else:
        filename = filename.replace(matches[0], '')
        reader_class = BagFileReader
    else:
      reader_class = BagFileReader

    self._reader = reader_class(
        filename=filename,
        separate_limits=separate_limits,
        decompress=decompress,
    )

  def __len__(self) -> int:
    """Returns the number of records in the Bagz file."""
    return len(self._reader)

  def __getitem__(self, index: SupportsIndex) -> bytes:
    """Returns a record from the Bagz file."""
    return self._reader[index]


class BagWriter:
  """Writer for Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      compress: bool | None = None,
      compression_level: int = 0,
  ) -> None:
    """Creates a BagWriter.

    Args:
      filename: The name of the Bagz file to write.
      separate_limits: Whether to keep the limits in a separate file.
      compress: Whether to compress the records. If None, uses the file
        extension to determine whether to compress.
      compression_level: The compression level to use when compressing.
    """
    if compress or (compress is None and filename.endswith('.bagz')):
      self._process = zstd.ZstdCompressor(level=compression_level).compress
    else:
      self._process = lambda x: x
    self._separate_limits = separate_limits
    directory, name = os.path.split(filename)
    self._records = open(filename, 'wb')
    self._limits = open(os.path.join(directory, 'limits.' + name), 'wb+')

  def write(self, data: bytes) -> None:
    """Writes a record to the Bagz file."""
    if data:
      self._records.write(self._process(data))
    self._limits.write(struct.pack('<q', self._records.tell()))

  def flush(self) -> None:
    """Flushes the Bagz file."""
    self._records.flush()
    self._limits.flush()

  def __enter__(self) -> Self:
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    """Ensures the Bagz file is closed when exiting a context."""
    self.close()

  def close(self) -> None:
    """Concatenates the limits file to the end of the data file."""
    if self._separate_limits:
      self._records.close()
      self._limits.close()
    else:
      self._limits.seek(0)
      shutil.copyfileobj(self._limits, self._records)
      self._records.close()
      os.unlink(self._limits.name)
      self._limits.close()


class BagDataSource:
  """PyGrain-compatible data source for bagz files."""

  def __init__(self, path: epath.PathLike) -> None:
    """Creates a new BagDataSource object.

    Args:
      path: The path to the bag file.
    """
    self._path = os.fspath(path)
    self._reader = BagReader(self._path)
    self._num_records = len(self._reader)

  def __len__(self) -> int:
    return self._num_records

  def __getitem__(self, record_key: SupportsIndex) -> bytes:
    return self._reader[record_key]

  def __getstate__(self) -> dict[str, Any]:
    state = self.__dict__.copy()
    del state['_reader']
    return state

  def __setstate__(self, state) -> None:
    self.__dict__.update(state)
    self._reader = BagReader(self._path)

  def __repr__(self) -> str:
    return f'BagDataSource(path={self._path!r}'
