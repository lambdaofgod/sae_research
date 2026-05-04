"""
Resilient S3 cache that skips corrupt files instead of crashing.

dictionary_learning's S3RCache sets stop=True and kills the download worker
when torch.load fails on a downloaded file (e.g. truncated download via
port-forward). This deadlocks the training loop once the buffer drains.

This module provides a drop-in replacement that logs and skips bad files.

Also fixes the upstream ~_stop_downloading~ deadlock: parent's shutdown
is a cooperative-only loop ~while not all(p not alive): sleep(0.25)~,
which never exits when workers are parked in ~aiohttp~ I/O syscalls
and never see the stop flag. We add a hard timeout + ~terminate()~ +
~kill()~ fallback, plus ~daemon=True~ on the worker processes so
they die with the parent on normal interpreter exit.
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

# How long to wait for workers to honor the stop flag before terminating.
# Cooperative shutdown only succeeds when workers happen to be at a queue
# checkpoint; if they're parked in aiohttp I/O, they never see it.
_COOPERATIVE_SHUTDOWN_TIMEOUT_S = 5.0
# After terminate(), grace period for SIGTERM to take effect before kill().
_TERMINATE_GRACE_S = 5.0


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
                    daemon=True,
                )
                p.start()
                self._running_processes.append(p)
                time.sleep(0.75)

        return self

    def _stop_downloading(self):
        """Override parent's cooperative-only shutdown with a hard timeout.

        Parent ~S3RCache._stop_downloading~ spins
        ~while not all(not p.is_alive()): sleep(0.25)~ forever when
        workers are blocked in aiohttp reads — they never see ~stop~
        because they're parked in syscalls. We give cooperative shutdown
        a few seconds, then terminate stragglers.
        """
        self._file_index.value = len(self._s3_paths)
        self._stop.value = True

        deadline = time.monotonic() + _COOPERATIVE_SHUTDOWN_TIMEOUT_S
        while not all(not p.is_alive() for p in self._running_processes):
            if time.monotonic() > deadline:
                break
            if not self.readable_tensors.empty():
                self.readable_tensors.get()
            if not self.writeable_tensors.full():
                self.writeable_tensors.put(0)
            time.sleep(0.25)

        stragglers = [p for p in self._running_processes if p.is_alive()]
        if stragglers:
            logger.warning(
                "S3 cache shutdown: %d worker(s) ignored stop flag after "
                "%.1fs; terminating",
                len(stragglers),
                _COOPERATIVE_SHUTDOWN_TIMEOUT_S,
            )
            for p in stragglers:
                p.terminate()
            for p in stragglers:
                p.join(timeout=_TERMINATE_GRACE_S)
                if p.is_alive():
                    logger.error(
                        "S3 cache shutdown: worker pid=%s survived terminate(); kill()",
                        p.pid,
                    )
                    p.kill()
                    p.join()

        for p in self._running_processes:
            p.join()

        self._ongoing_downloads.value = 0
        self._running_processes = []
