"""
Resilient S3 cache that skips corrupt files instead of crashing.

dictionary_learning's S3RCache sets stop=True and kills the download worker
when torch.load fails on a downloaded file (e.g. truncated download via
port-forward). This deadlocks the training loop once the buffer drains.

This module provides a drop-in replacement that logs and skips bad files.
"""

import asyncio
import logging
import time
from multiprocessing import Process

from dictionary_learning.activault_s3_buffer import (
    S3RCache,
    compile,
    download_chunks,
    write_tensor,
)

import aiohttp

logger = logging.getLogger(__name__)


async def _resilient_async_download(
    buffer,
    file_index,
    s3_paths,
    stop,
    readable_tensors,
    writeable_tensors,
    ongoing_downloads,
    concurrency,
    bytes_per_file,
    chunk_size,
    shuffle,
    seed,
    return_ids,
):
    """Like _async_download but skips corrupt files instead of crashing."""
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        while file_index.value < len(s3_paths) and not stop.value:
            with ongoing_downloads.get_lock():
                ongoing_downloads.value += 1
            with file_index.get_lock():
                url = s3_paths[file_index.value]
                file_index.value += 1
            bytes_results = await download_chunks(
                session, url, bytes_per_file, chunk_size
            )
            if bytes_results is not None:
                try:
                    t = compile(bytes_results, shuffle, seed, return_ids)
                    write_tensor(
                        t,
                        buffer,
                        writeable_tensors,
                        readable_tensors,
                        ongoing_downloads,
                    )
                except Exception as e:
                    logger.error(f"Skipping corrupt file: {e}")
                    logger.error(f"Failed URL: {url}")
                    with ongoing_downloads.get_lock():
                        ongoing_downloads.value -= 1
            else:
                logger.error(f"Failed to download URL: {url}")
                with ongoing_downloads.get_lock():
                    ongoing_downloads.value -= 1


def _resilient_download_loop(*args):
    asyncio.run(_resilient_async_download(*args))


class ResilientS3RCache(S3RCache):
    """S3RCache that skips corrupt files instead of killing the worker."""

    def __iter__(self):
        self._reset()

        if self._running_processes:
            raise ValueError(
                "Cannot iterate over cache a second time while it is downloading"
            )

        if len(self._s3_paths) > self._initial_file_index:
            while len(self._running_processes) < self.n_workers:
                p = Process(
                    target=_resilient_download_loop,
                    args=(
                        self.buffer,
                        self._file_index,
                        self._s3_paths[self._initial_file_index :],
                        self._stop,
                        self.readable_tensors,
                        self.writeable_tensors,
                        self._ongoing_downloads,
                        self.concurrency,
                        self.metadata["bytes_per_file"],
                        self.chunk_size,
                        self.shuffle,
                        self.seed,
                        self.return_ids,
                    ),
                )
                p.start()
                self._running_processes.append(p)
                time.sleep(0.75)

        return self
