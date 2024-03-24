import zstandard
import itertools
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator, List, Optional, Sequence, Tuple, TypeVar, Union
import msgspec
import numpy as np
from tokenizer import Tokenizer
from cached_path import cached_path
from smashed.utils.io_utils import (
    MultiPath,
    decompress_stream,
    open_file_for_write,
    recursively_list_files,
    stream_file_for_read,
)
import logging
from tool import DataTool

log = logging.getLogger(__name__)


class MemmapTool:

    @staticmethod
    def fill_memmap(
        tokenizer_id: str,
        path_or_paths: Union[str, List[str]],
        memmap_path: str,
        dtype: np.dtype,
        safe_mode: bool = False,
        # 512M tokens * 2 bytes per token (uint16) = 1GB
        max_tokens: int = 512 * 1024 * 1024,
        sample_rate: float = 1.0,
        random_seed: int = 3920,
        repeat_sequence: int = 1,
        cache_dir: Optional[str] = None,
    ) -> int:
        """Write a memmap file from a file of documents."""

        # set the seed in case we need to sample
        np.random.seed(random_seed)

        # we need to make a new tokenizer here because it's not pickleable
        # tokenizer = Tokenizer.from_pretrained(tokenizer_id, truncate_to=None)
        file = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/tokenizers/{tokenizer_id}"
        # print(file)
        tokenizer = Tokenizer.from_file(file, truncate_to=None)

        # first memmap file will be created in the loop below
        memmap: Optional[MemmapFile] = None

        # we increment this every time we create a new memmap file
        file_index = 0

        # total number of tokens written
        total_tokens = 0

        # make sure path is a list
        path_or_paths = [path_or_paths] if isinstance(
            path_or_paths, str) else path_or_paths

        with ExitStack() as stack:
            it = itertools.chain.from_iterable(
                # repeat the sequence if necessary
                tokenize_file(tokenizer=tokenizer, path=path,
                              safe_mode=safe_mode, cache_dir=cache_dir)
                for _ in range(repeat_sequence)
                for path in path_or_paths
            )
            for line_no, token_ids in enumerate(it, start=1):
                # perform sampling if necessary
                if sample_rate < 1.0 and np.random.rand() > sample_rate:
                    continue

                # flush any 10k lines or so; improves stability
                flush = line_no % 10_000 == 0

                # increment the total number of tokens written
                total_tokens += len(token_ids)

                # if leftovers_to_write is not None it means that either memmap is None or it's full,
                # so we will need to create a new one later
                leftovers_to_write = memmap.write(
                    token_ids, flush=flush) if memmap is not None else token_ids

                if leftovers_to_write is not None:
                    # close the previous memmap (if one is open)
                    stack.pop_all().close()

                    # create a new memmap file; progressively name them with an index
                    curr_memmap_path = f"{memmap_path}_{file_index:05d}.npy"
                    memmap = stack.enter_context(MemmapFile(
                        path=curr_memmap_path, dtype=dtype, max_tokens=max_tokens))

                    # increment the file index and reset the tokens index
                    file_index += 1

                    # do the actual writing
                    memmap.write(leftovers_to_write)

            # close the last memmap
            stack.pop_all().close()

        return total_tokens


def tokenize_file(
        tokenizer: Tokenizer,
        path: str,
        safe_mode: bool = False,
        cache_dir: Optional[str] = None,
) -> Generator[List[int], None, None]:
    """Tokenize a file of documents using the provided tokenizer; file is expected to be a gzipped JSON lines
    file, each containing a field named `text`.
    """

    text_list = DataTool.extract_content_from_zstd(path, 'text')
    i = 1
    try:
        for text in text_list:
            yield tokenizer.encode(text, add_special_tokens=True)
            i += 1
    except Exception:
        log.exception(f"Error processing {path}:{i:,}")
        pass


class MemmapFile:
    """Context manager responsible for writing, resizing, and closing / uploading a memmap file."""

    DEFAULT_MAX_TOKENS = 512 * 1024 * 1024  # 500M tokens / 1GB

    def __init__(
            self,
            path: str,
            dtype: np.dtype,
            max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Create a new memmap file.

        Args:
            path (str): Location for the memmap file. If the path is not local, the memmap file will be
                written to a temporary file first and then uploaded to the destination.
            dtype (np.dtype): Data type for the memmap file; must be a valid numpy dtype.
            max_tokens (int, optional): Maximum number of tokens per file. Defaults to 500M tokens, which is 1GB.
        """
        self.path = MultiPath.parse(path)
        self.dtype = dtype
        self.max_tokens = max_tokens

        self._local_path: Optional[Path] = None
        self._written_tokens = 0
        self._memmap: Optional[np.memmap] = None

    def __len__(self) -> int:
        """Length of the memmap file in tokens that have been written."""
        return self._written_tokens

    def write(self, values: List[int], flush: bool = False) -> Optional[List[int]]:
        """Write a list of token IDs to the memmap file; if only a subset of the values can be written,
        return the rest.

        Args:
            values (List[int]): List of token IDs to write.
            flush (bool, optional): Whether to flush the memmap file after writing. Defaults to False.
        """

        if self._memmap is None:
            raise RuntimeError("MemmapFile is not open")

        if (len(values) + self._written_tokens) >= self.max_tokens:
            values = values[: self.max_tokens - self._written_tokens]
            rest = values[self.max_tokens - self._written_tokens:]
        else:
            rest = None

        self._memmap[self._written_tokens: self._written_tokens +
                     len(values)] = values
        self._written_tokens += len(values)

        if flush:
            self._memmap.flush()

        return rest

    def __enter__(self) -> "MemmapFile":
        """Context manager entry point. Creates the memmap file and returns self."""

        assert self._memmap is None, "MemmapFile is already open"

        if self.path.is_local:
            self._local_path = self.path.as_path
            # make sure the directory exists
            self._local_path.parent.mkdir(
                parents=True, exist_ok=True)  # type: ignore
        else:
            with NamedTemporaryFile(delete=False, prefix="olmo_memmap") as f:
                # if the destination for the memmap is not local, we need to write to a temporary file first
                self._local_path = Path(f.name)

        self._memmap = np.memmap(
            mode="w+", filename=self._local_path, dtype=self.dtype, shape=(self.max_tokens,))
        log.info(
            f"Created memmap file at {self._local_path} of size {self._memmap.nbytes:,} bytes")

        return self

    def __exit__(self, *_):
        """Context manager exit point. Closes the memmap file."""
        return self.close()

    def close(self):
        """Close the memmap file and optionally upload it to the destination (in the case of a remote path)."""
        assert self._local_path is not None, "MemmapFile is not open"
        assert self._memmap is not None, "MemmapFile is not open"

        try:
            # write the memmap to the destination
            self._memmap.flush()

            # we resize the memmap to the number of tokens actually written
            if self._written_tokens < self.max_tokens:
                del self._memmap
                os.rename(self._local_path, (temp_path :=
                          self._local_path.with_suffix(".tmp")))
                new_memmap = np.memmap(
                    mode="w+", filename=self._local_path, dtype=self.dtype, shape=(self._written_tokens,)
                )
                old_memmap = np.memmap(
                    mode="r", filename=temp_path, dtype=self.dtype, shape=(self.max_tokens,))
                new_memmap[:] = old_memmap[: self._written_tokens]
                new_memmap.flush()
                log.info(
                    f"Resized memmap file from {self._local_path} {old_memmap.nbytes:,} to {new_memmap.nbytes:,} bytes")
                os.remove(temp_path)

            if not self.path.is_local:
                with ExitStack() as stack:
                    f = stack.enter_context(
                        stream_file_for_read(self._local_path, "rb"))
                    g = stack.enter_context(
                        open_file_for_write(self.path, mode="wb"))
                    g.write(f.read())
                log.info(f"Written memmap file to {self.path.as_str}")
        finally:
            if not self.path.is_local:
                # delete the temporary file under any circumstances
                os.remove(self._local_path)

        self._local_path = self._memmap = None


class InputDocumentSpec(msgspec.Struct):
    # almost 5x faster than built-in json decoding in my tests;
    # can work with approximate spec (i.e., ignore missing fields)
    text: str
