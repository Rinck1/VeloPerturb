"""Regressions for the observed SRA multicall dispatch and partial-output failure."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
import w4_renge_data as data


def test_fasterq_command_preserves_multicall_symlink_name(tmp_path, monkeypatch):
    directory = tmp_path/'bin'; directory.mkdir()
    dispatcher = directory/'sratools.3.4.1'; dispatcher.touch()
    tool = directory/'fasterq-dump'; tool.symlink_to(dispatcher.name)
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    command = data.extraction_command(dict(sra_bin='bin', extract_threads=8),
                                      tmp_path/'run.sra', tmp_path/'fastq', tmp_path/'scratch')
    assert Path(command[0]).name == 'fasterq-dump'
    assert Path(command[0]).is_symlink()
    assert Path(command[0]).resolve() == dispatcher
    assert '--include-technical' in command and '--split-files' in command


def test_empty_failed_extraction_directories_are_reused(tmp_path):
    fastq, scratch = tmp_path/'fastq', tmp_path/'scratch'
    fastq.mkdir(); scratch.mkdir()
    data.prepare_extraction_directories(fastq, scratch)
    assert fastq.is_dir() and scratch.is_dir()
    assert list(fastq.iterdir()) == list(scratch.iterdir()) == []


@pytest.mark.parametrize('interrupted', ['fastq', 'scratch'])
def test_nonempty_partial_is_not_overwritten_or_deleted(tmp_path, interrupted):
    partial_directory = tmp_path/interrupted; partial_directory.mkdir()
    partial = partial_directory/'partial'; partial.write_bytes(b'keep original partial')
    with pytest.raises(ValueError, match='preserved'):
        data.prepare_extraction_directories(tmp_path/'fastq', tmp_path/'scratch')
    assert partial.read_bytes() == b'keep original partial'
    other = tmp_path/('scratch' if interrupted == 'fastq' else 'fastq')
    assert not other.exists()
