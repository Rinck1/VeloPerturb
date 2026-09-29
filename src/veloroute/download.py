"""Bounded HTTPS range downloads with verified resume prefixes and atomic release.

Used when a long-lived SRA HTTP connection repeatedly stalls. The original
prefetch partial file is read-only and preserved. Published .sra appears only
after vdb-validate succeeds, so downstream workers cannot consume partial data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .artifacts import Run, save_json, sha256, utc_now
from .raw import run_command


def remote_info(url):
    if not url.startswith('https://'):
        raise ValueError('Only HTTPS sources are accepted')
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=45) as response:
        return int(response.headers['Content-Length']), response.headers.get('ETag')


def get_range(url, start, end, etag, *, timeout=90):
    headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}
    if etag:
        headers['If-Match'] = etag
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('Server did not honor exact HTTP byte range; no append is permitted')
                block = response.read(end-start+2)
                if len(block) != end-start+1:
                    raise IOError('Incomplete range response')
                return block
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def resume_prefix_matches(path, url, size, etag):
    prefix = Path(path)
    n = prefix.stat().st_size
    if not 0 < n < size:
        raise ValueError('Resume prefix must be a nonempty incomplete file')
    with prefix.open('rb') as stream:
        for start, end in [(0, min(n, 65536)-1), (max(0, n-65536), n-1)]:
            stream.seek(start)
            if stream.read(end-start+1) != get_range(url, start, end, etag):
                raise ValueError('Local partial is not a byte-identical prefix of this mirror')
    return n


def download_ranges(url, destination, output, *, resume_from=None, workers=4, chunk_mb=8, validator=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Destination exists; never overwrite a completed or active download')
    if not 1 <= workers <= 4 or not 1 <= chunk_mb <= 32:
        raise ValueError('Range concurrency/chunk budget exceeded')
    size, etag = remote_info(url)
    offset = resume_prefix_matches(resume_from, url, size, etag) if resume_from else 0
    config = {'url': url, 'destination': str(destination), 'size': size, 'etag': etag,
              'resume_from': str(resume_from) if resume_from else None, 'prefix_bytes': offset,
              'workers': workers, 'chunk_mb': chunk_mb, 'validator': str(validator) if validator else None}
    with Run(output, stage='range_download', kind='engineering', config=config,
             inputs=[resume_from] if resume_from else []) as run:
        stage = run.directory/'download.incomplete'
        with stage.open('xb') as stream:
            if resume_from:
                with Path(resume_from).open('rb') as source:
                    shutil.copyfileobj(source, stream, length=8 << 20)
            stream.truncate(size)
        ranges = [(start, min(size-1, start+(chunk_mb << 20)-1)) for start in range(offset, size, chunk_mb << 20)]
        completed = offset
        save_json(run.directory/'status.json', {'status': 'running', 'pid': os.getpid(), 'bytes_complete': completed, 'total_bytes': size})

        def fetch(span):
            start, end = span
            block = get_range(url, start, end, etag)
            with stage.open('r+b', buffering=0) as stream:
                stream.seek(start)
                stream.write(block)
            return len(block)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(fetch, span) for span in ranges]
            try:
                for future in as_completed(pending):
                    completed += future.result()
                    save_json(run.directory/'status.json', {'status': 'running', 'pid': os.getpid(),
                              'bytes_complete': completed, 'total_bytes': size, 'updated_utc': utc_now()})
            except BaseException as error:
                for future in pending:
                    future.cancel()
                save_json(run.directory/'status.json', {'status': 'failed', 'bytes_complete': completed,
                          'total_bytes': size, 'error': f'{type(error).__name__}: {error}', 'finished_utc': utc_now()})
                raise
        if completed != size or stage.stat().st_size != size:
            raise ValueError('Assembled byte count mismatch')
        if validator:
            run_command([str(Path(validator).resolve()), str(stage)], run.directory/'validate', timeout_seconds=600)
        checksum = sha256(stage)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError('Destination appeared during download; another owner is active')
        # Atomic hard-link publication is exclusive even if another process races
        # us. Retaining the staging hard link costs no extra data blocks.
        os.link(stage, destination)
        save_json(run.directory/'status.json', {'status': 'complete', 'bytes_complete': completed, 'total_bytes': size,
                  'sha256': checksum, 'destination': str(destination), 'finished_utc': utc_now()})
        (run.directory/'RESULTS.md').write_text('# Verified range download\n\n'+json.dumps(config, indent=2)
            +f'\n\nSHA256: {checksum}. Original prefetch partial preserved; released only after validation.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume-from')
    parser.add_argument('--validator')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.probe:
        size, etag = remote_info(args.url)
        start = time.monotonic()
        block = get_range(args.url, 0, (1 << 20)-1, etag)
        print(json.dumps({'size': size, 'etag': etag, 'probe_bytes': len(block), 'elapsed_seconds': time.monotonic()-start}))
    else:
        download_ranges(args.url, args.destination, args.output, resume_from=args.resume_from, validator=args.validator)


if __name__ == '__main__':
    main()
