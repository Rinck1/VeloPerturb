import json

import pytest

from veloroute import download


def test_range_resume_and_atomic_release(tmp_path, monkeypatch):
    data = bytes(range(251))*10000
    monkeypatch.setattr(download, 'remote_info', lambda url: (len(data), 'frozen_etag'))
    monkeypatch.setattr(download, 'get_range', lambda url, start, end, etag: data[start:end+1])
    prefix = tmp_path/'original.partial'
    prefix.write_bytes(data[:100000])
    destination = tmp_path/'done.sra'
    download.download_ranges('https://official.example/file', destination, tmp_path/'run',
                             resume_from=prefix, workers=2, chunk_mb=1)
    assert destination.read_bytes() == data
    assert prefix.read_bytes() == data[:100000]
    status = json.loads((tmp_path/'run/status.json').read_text())
    assert status['status'] == 'complete'


def test_bad_resume_prefix_is_not_appended(tmp_path, monkeypatch):
    monkeypatch.setattr(download, 'remote_info', lambda url: (100, 'frozen'))
    monkeypatch.setattr(download, 'get_range', lambda url, start, end, etag: b'A'*(end-start+1))
    prefix = tmp_path/'bad.partial'; prefix.write_bytes(b'WRONG')
    with pytest.raises(ValueError, match='byte-identical'):
        download.download_ranges('https://official.example/file', tmp_path/'out', tmp_path/'run', resume_from=prefix)
    assert not (tmp_path/'out').exists()
    assert prefix.read_bytes() == b'WRONG'


def test_failed_range_never_publishes_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(download, 'remote_info', lambda url: (100, 'frozen'))
    def fail(*args, **kwargs):
        raise IOError('interrupted range')
    monkeypatch.setattr(download, 'get_range', fail)
    with pytest.raises(IOError):
        download.download_ranges('https://official.example/file', tmp_path/'out', tmp_path/'run')
    assert not (tmp_path/'out').exists()
    assert json.loads((tmp_path/'run/status.json').read_text())['status'] == 'failed'


def test_existing_destination_is_never_overwritten(tmp_path):
    path = tmp_path/'out'; path.write_text('owned')
    with pytest.raises(FileExistsError):
        download.download_ranges('https://official.example/file', path, tmp_path/'run')
    assert path.read_text() == 'owned'
