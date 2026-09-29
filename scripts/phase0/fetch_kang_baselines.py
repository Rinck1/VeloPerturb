"""Fetch reviewed official repository identities without requiring system git."""
import io
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

from veloroute.artifacts import Run,load_config,save_json,sha256

parser=argparse.ArgumentParser();parser.add_argument('--branch-fallback',action='store_true');args=parser.parse_args()
c=load_config('configs/veloroute_kang_20260914.yaml');root=Path(c['root'])/'external'
# CellOT was already reproduced by the user; retain existing sources, fetch no new copy.
repos={'CellFlow':'theislab/CellFlow','scDFM':'AI4Science-WestlakeU/scDFM'}
for name,repo in repos.items():
    out=root/name
    with Run(root/(name+('_acquisition_branch' if args.branch_fallback else '_acquisition')),stage='official_baseline_source_acquisition',kind='engineering',
             config=dict(repository=repo),inputs=[]) as run:
        commit=None
        if not args.branch_fallback:
            request=urllib.request.Request(f'https://api.github.com/repos/{repo}/commits/HEAD',headers={'User-Agent':'VeloRoute-reproducibility'})
            with urllib.request.urlopen(request,timeout=45) as f:commit=json.load(f)['sha']
        url=f'https://codeload.github.com/{repo}/tar.gz/{commit or "refs/heads/main"}'
        archive=run.directory/'source.tar.gz'
        with urllib.request.urlopen(url,timeout=90) as response,archive.open('xb') as f:
            shutil.copyfileobj(response,f,length=1<<20)
        out.mkdir(parents=True,exist_ok=False)
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers()
            for m in members:
                p=Path(m.name)
                if p.is_absolute() or '..' in p.parts or not (m.isdir() or m.isfile()):
                    raise ValueError('Unsafe/nonregular archive member; manual review required')
            for m in members:
                parts=Path(m.name).parts[1:]
                if not parts:continue
                dest=out.joinpath(*parts)
                if m.isdir():dest.mkdir(parents=True,exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(m) as source,dest.open('xb') as f:shutil.copyfileobj(source,f)
        result=dict(repository=repo,commit=commit,archive_sha256=sha256(archive),source=str(out),
                    version_binding='archive_content_SHA256' if commit is None else 'upstream_commit_SHA256_archive',
                    acquired_only_not_installed_or_run=True)
        save_json(run.directory/'summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Official baseline source\n\n'+json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
