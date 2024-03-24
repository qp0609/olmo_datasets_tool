import json
from typing import Tuple, List
import click
from tool.dataset_tool import DataTool
from contextlib import ExitStack
import concurrent.futures
from concurrent.futures import Future
import time
import os
# import logging
import zstandard as zstd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
)

# logging.basicConfig(filename=f'{os.path.dirname(os.path.abspath(__file__))}/log'
#                              f'/prepare_starcode.log',
#                              level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# log = logging.getLogger("logger")


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
@click.option("--random-seed", type=int, default=3920)
@click.option("-j", "--workers", "max_workers", type=int, default=1, help="Defaults to number of CPUs")
def main(
        src: Tuple[str, ...],
        output: str,
        max_workers: int = 1,
        random_seed: int = 3920,
):
    print("=== CONFIGURATION ===")
    print(f"src:                {src}")
    print(f"output:             {output}")
    print(f"random_seed:        {random_seed}")
    print(f"max_workers:        {max_workers}")
    print("=====================")

    # 1. make source and target
    exploded_src, exploded_dst = DataTool.make_source_and_target(
        src=src, output=output,
        random_seed=random_seed,
    )

    # 2. zstd compress
    workers_cnt = min(max_workers or os.cpu_count() or 1, len(exploded_src))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_cnt) as executor:
        futures: List[Future[int]] = []
        for src_path, dst_path in zip(exploded_src, exploded_dst):
            future = executor.submit(
                zstd_compressor, src_path=src_path, dst_path=dst_path)
            futures.append(future)
        # 3. print progress
        with get_progress() as progress:
            for future in progress.track(
                    concurrent.futures.as_completed(futures),
                    description="Compressing jsonl ...",
                    total=len(futures),
            ):
                future.result()
        pass


def zstd_compressor(src_path: str, dst_path: str):
    """ load jsonl and compress in zstd """
    start_time = time.time()
    with ExitStack() as stack:
        input_file = stack.enter_context(open(src_path, 'rb'))
        output_file = stack.enter_context(open(dst_path, 'wb'))
        cctx = zstd.ZstdCompressor(level=3)

        json_objects = []

        for i, line in enumerate(input_file, 1):
            try:
                json_data = json.loads(line.strip().decode('utf-8'))
                if 'content' in json_data:
                    json_objects.append({'text': json_data['content']})
                if len(json_objects) >= 100:
                    compressed_data = cctx.compress(
                        json.dumps(json_objects).encode('utf-8'))
                    output_file.write(compressed_data)
                    json_objects.clear()
            except Exception as e:
                print(f"Error processing {src_path}:{i:,} -> {e}")
                pass

        if json_objects:
            compressed_data = cctx.compress(
                json.dumps(json_objects).encode('utf-8'))
            output_file.write(compressed_data)

        end_time = time.time()
        print(f"Saved {src_path} to {dst_path}, time: {end_time - start_time}")


def get_progress() -> Progress:
    return Progress(
        "[progress.description]{task.description}",
        MofNCompleteColumn(),
        "files",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


if __name__ == "__main__":
    main()
