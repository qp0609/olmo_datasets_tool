import json
from typing import Tuple, Union, List
import click
from tool import dataset_tool
from contextlib import ExitStack
import time
import logging
import zstandard as zstd

logging.basicConfig(filename='tmp.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("logger")


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
@click.option("--paths_per_worker", type=click.IntRange(min=1), default=1)
@click.option("-j", "--workers", "max_workers", type=int, default=1, help="Defaults to number of CPUs")
def main(
        src: Tuple[str, ...],
        output: str,
        max_workers: int = 1,
        random_seed: int = 3920,
        paths_per_worker: int = 1,
):
    print("=== CONFIGURATION ===")
    print(f"src:                {src}")
    print(f"output:             {output}")
    print(f"random_seed:        {random_seed}")
    print(f"max_workers:        {max_workers}")
    print(f"paths_per_worker:   {paths_per_worker}")
    print("=====================")

    # 1. make source and target
    exploded_src, exploded_dst = dataset_tool.make_source_and_target(
        src=src, output=output, random_seed=random_seed, paths_per_worker=paths_per_worker
    )

    # 2. zstd compress
    for i, (src_path, dst_path) in enumerate(zip(exploded_src, exploded_dst), 1):
        start_time = time.time()
        with ExitStack() as stack:
            input_file = stack.enter_context(open(src_path, 'rb'))
            json_objects_str = ''
            try:
                for line in input_file:
                    json_data = json.loads(line.strip().decode('utf-8'))
                    if 'content' in json_data:
                        content = json_data['content']
                    if content:
                        json_objects_str += json.dumps({'text': content})
                        json_objects_str += '\n'
            except Exception as e:
                log.error(f"Error processing {src_path}:{i:,} -> {e}")
                pass

            with open(dst_path, 'wb') as output_file:
                cctx = zstd.ZstdCompressor(level=3)
                compressed_data = cctx.compress(json_objects_str.encode('utf-8'))
                output_file.write(compressed_data)

            end_time = time.time()
            i += 1
            log.info(f"file-{i} save {src_path} to {dst_path}, time: {end_time - start_time}")


if __name__ == "__main__":
    main()
