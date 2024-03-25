import numpy as np
import random
from typing import Tuple
import logging
from smashed.utils.io_utils import (
    recursively_list_files,
)
import zstandard as zstd
import json
from typing import Generator

log = logging.getLogger(__name__)


class DataTool:

    @staticmethod
    def make_source_and_target(src: Tuple[str, ...],
                               output: str,
                               suffix: str = 'zst',
                               random_seed: int = 3920,
                               ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """ Recursively list all files in the source directories
            Args:
                src (Tuple[str, ...]): The directory path list for input file.
                output (str): The directory path for output file.
                suffix (str): output file suffix
                random_seed (int): shuffle the source paths
        """
        np.random.seed(random_seed)
        random.seed(random_seed)

        exploded_src = list(
            set(path for prefix in src for path in recursively_list_files(prefix)))
        output_digits = np.ceil(np.log10(len(exploded_src) + 1)).astype(int)
        # shuffle the source paths
        random.shuffle(exploded_src)

        # determine the destination paths
        # exploded_dst = [f'{output.rstrip("/")}/{src.split("/")[-1].split(".")[0]}.{suffix}' for src in exploded_src]
        exploded_dst = [
            f'{output.rstrip("/")}/{i:0{output_digits}d}' for i in range(len(exploded_src))]

        return tuple(exploded_src), tuple(exploded_dst)

    @staticmethod
    def extract_content_from_zstd(src: str,
                                  key_name: str = 'text'
                                  ) -> Generator[str, None, None]:
        with open(src, 'rb') as f:
            file_content = f.read()
            json_list = zstd.decompress(file_content).splitlines()
            if not json_list:
                log.error(f'extract zstd error, cause=no json, file={src}')
                return

            for line in json_list:
                try:
                    json_data = json.loads(line)
                    if isinstance(json_data, dict) and key_name in json_data.keys():
                        yield str(json_data[key_name])
                    if isinstance(json_data, list):
                        for data in json_data:
                            if isinstance(data, dict) and key_name in data.keys():
                                yield str(data[key_name])
                except json.decoder.JSONDecodeError:
                    log.exception(
                        f'extract zstd error, cause=decoderError, file={src}')
                    pass
