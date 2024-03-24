import concurrent.futures
import functools
import json
import os
from concurrent.futures import Future
from typing import List, Optional, Tuple

import click
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from smashed.utils.io_utils import (
    decompress_stream,
    # open_file_for_write,
    recursively_list_files,
    stream_file_for_read,
)
# import tokenizer
# sys.path.append('../tokenizer')
from tokenizer import Tokenizer
from tool import DataTool, MemmapTool

import logging

logging.basicConfig(filename=f'{os.path.dirname(os.path.abspath(__file__))}/log'
                             f'/prepare_memmap_dataset.log',
                    level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


@click.command()
@click.argument(
    "src",
    nargs=-1,
    type=str,
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=str,
    help="Specify the output path.",
    prompt="Output directory",
)
@click.option(
    "--tokenizer",
    "tokenizer_id",
    type=str,
    help="Name of path of a pretrained tokenizer",
    default="allenai/eleuther-ai-gpt-neox-20b-pii-special",
)
@click.option("--dtype", "dtype_str", default="uint16")
@click.option("--validate/--no-validate", default=False)
@click.option("--sample-rate", type=click.FloatRange(min=0.0, max=1.0), default=1.0)
@click.option("--random-seed", type=int, default=3920)
@click.option("--repeat-sequence", type=click.IntRange(min=1), default=1)
@click.option("--paths-per-worker", type=click.IntRange(min=1), default=1)
@click.option(
    "--cache-dir",
    type=str,
    default=None,
    help="Cache directory for the tokenizer; use system default if not specified",
)
@click.option(
    "--max-tokens",
    default=512 * 1024 * 1024,
    type=int,
    help="Maximum number of tokens to store in a single memmap file (default: 512M tokens or 1GB)",
)
@click.option("--debug/--no-debug", default=False, help="Enable debug (single process mode)")
@click.option(
    "--safe-mode/--fast-mode", default=False, help="Safe mode caches locally and decompresses using gzip.open"
)
@click.option("-j", "--workers", "max_workers", type=int, default=1, help="Defaults to number of CPUs")
def main(
        src: Tuple[str, ...],
        output: str,
        tokenizer_id: str = "allenai-eleuther-ai-gpt-neox-20b-pii-special.json",
        dtype_str: str = "uint16",
        validate: bool = False,
        max_tokens: int = 512 * 1024 * 1024,
        safe_mode: bool = False,
        debug: bool = False,
        sample_rate: float = 1.0,
        random_seed: int = 3920,
        repeat_sequence: int = 1,
        paths_per_worker: int = 1,
        max_workers: int = 1,
        cache_dir: Optional[str] = None,
):
    print("=== CONFIGURATION ===")
    print(f"src:              {src}")
    print(f"output:           {output}")
    print(f"tokenizer_id:     {tokenizer_id}")
    print(f"dtype_str:        {dtype_str}")
    print(f"validate:         {validate}")
    print(f"max_tokens:       {max_tokens}")
    print(f"safe_mode:        {safe_mode}")
    print(f"debug:            {debug}")
    print(f"sample_rate:      {sample_rate}")
    print(f"random_seed:      {random_seed}")
    print(f"repeat_sequence:  {repeat_sequence}")
    print(f"paths_per_worker: {paths_per_worker}")
    print(f"max_workers:      {max_workers}")
    print(f"cache_dir:        {cache_dir}")
    print("=====================")
    dtype = np.dtype(dtype_str)

    # 1. make source and target
    exploded_src, exploded_dst = DataTool.make_source_and_target(
        src=src, output=output, random_seed=random_seed, suffix='npy'
    )

    for i, (src, dst) in enumerate(zip(exploded_src, exploded_dst), 1):
        print(f'i={i}, src={src}, dst={dst}')

    # 2. creating a partial here with all the arguments we need to pass to fill_memmap except for the paths
    # so that we don't make mistakes between debug and non-debug mode
    fill_memmap_fn = functools.partial(
        MemmapTool.fill_memmap,
        tokenizer_id=tokenizer_id,
        dtype=dtype,
        max_tokens=max_tokens,
        safe_mode=safe_mode,
        sample_rate=sample_rate,
        random_seed=random_seed,
        repeat_sequence=repeat_sequence,
        cache_dir=cache_dir,
    )

    total_tokens_written = 0

    # 3. Now tokenizer all documents again and populate the memmap array. We do this in parallel.
    workers_cnt = min(max_workers or os.cpu_count() or 1, len(exploded_src))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_cnt) as executor:
        futures: List[Future[int]] = []
        for src_path, dst_path in zip(exploded_src, exploded_dst):
            future = executor.submit(
                fill_memmap_fn, path_or_paths=src_path, memmap_path=dst_path)
            futures.append(future)
        with get_progress() as progress:
            for future in progress.track(
                    concurrent.futures.as_completed(futures),
                    description="Filling memmap arrays...",
                    total=len(futures),
            ):
                total_tokens_written += future.result()
    log.info(f"Done! File(s) written to {output}")
    log.info(f"Total tokens written: {total_tokens_written:,}")

    if validate:
        log.info("Validating...")
        tokenizer = Tokenizer.from_file(f"../tokenizers/{tokenizer_id}",
                                        truncate_to=None)

        def encode_fn(row):
            return tokenizer.encode(json.loads(row)["text"], add_special_tokens=True)  # noqa

        total_tokens = total_docs = 0
        for input_path in (path for prefix in src for path in recursively_list_files(prefix)):
            with stream_file_for_read(input_path, mode="rb") as f, decompress_stream(f, mode="rt") as g:
                for row in g:
                    total_docs += 1
                    total_tokens += len(encode_fn(row))

        for output_path in recursively_list_files(output):
            memmap = np.memmap(output_path, mode="r", dtype=dtype)
            total_tokens -= len(memmap)
            total_docs -= (memmap == tokenizer.eos_token_id).sum()
            assert (memmap < tokenizer.vocab_size).all(
            ), f"Invalid token ID in {output_path}"

        assert total_tokens == 0, f"Total tokens mismatch: {total_tokens} != 0"
        assert total_docs == 0, f"Total docs mismatch: {total_docs} != 0"

        log.info("All good!")


def get_progress() -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        "files",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


if __name__ == '__main__':
    main()
