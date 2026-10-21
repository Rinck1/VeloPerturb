"""Source-only CellFlow transfer pack over the frozen 32 RENGE L1 folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse

from w4_artifacts import Artifacts, sha256
from w4_shorttime import condition_embeddings, read_count_rows, read_metadata
from w4_streams import SourceTransform

ROOT = Path(__file__).resolve().parents[2]


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def checked(path, digest):
    if sha256(path) != digest:
        raise ValueError(f'Artifact changed: {path}')
    return Path(path)


def pooled_tokens(transform, s, u):
    """Pool raw counts before normalization; both modalities share all-gene S depth."""
    ps = sparse.csr_matrix(np.asarray(s.sum(axis=0)))
    pu = sparse.csr_matrix(np.asarray(u.sum(axis=0)))
    sn, un = transform.normalized(ps, pu)
    return dict(kin=transform.projected_rate(sn, un-transform.gamma*sn)[0],
        Ssum=(np.log1p(sn) @ transform.components.T)[0],
        directU=(np.log1p(un) @ transform.components.T)[0])


def normalization(groups):
    result = {}
    for kind in ('kin', 'Ssum', 'directU', 'cell_kin', 'cell_directU'):
        a = (np.vstack([g[kind] for g in groups]) if kind.startswith('cell_')
             else np.stack([g['population_'+kind] for g in groups]))
        result['mean_'+kind] = a.mean(0)
        result['scale_'+kind] = np.maximum(a.std(0), 1e-5)
    return result


def convert(parent, tokens, reference, embedding, label, day, *, training):
    result = dict(z=parent['z'], cell_ids=parent['cell_ids'],
        cell_kin=(parent['velocity']-reference['kin']).astype('float32'),
        cell_directU=(parent['direct_U']-reference['directU']).astype('float32'),
        embedding=np.asarray(embedding, dtype='float32'), label=np.asarray(label),
        time_hours=np.asarray(24.*(day-2)))
    for kind in ('kin', 'Ssum', 'directU'):
        result['population_'+kind] = (tokens[kind]-reference[kind]).astype('float32')
    if training:
        result['target'] = parent['target']
    else:
        result['score_indices'] = parent['score_indices']
    return result


def prepare(output, config_path):
    config_path = Path(config_path).resolve()
    c = json.loads(config_path.read_text())
    parent_root = Path(c['parent_root'])
    parent_manifest = parent_root/'study_manifest.json'
    parent = json.loads(parent_manifest.read_text())
    lockpath = ROOT/'outputs/w4_protocol_lock_v1/protocol_lock.json'
    lock = json.loads(lockpath.read_text())
    run = Artifacts(output, seed=c['seeds'], inputs=[config_path, parent_manifest, lockpath,
        Path(__file__), ROOT/c['protocol_path']])
    # Verify only source counts and the frozen condition embeddings here.
    for key in ('day2', 'day3', 'embeddings'):
        checked(**dict(path=lock['data_hashes'][key]['path'], digest=lock['data_hashes'][key]['sha256']))
    metadata = {day: read_metadata(lock['data_hashes'][f'day{day}']['path'])[0] for day in (2,3)}
    records = []
    for old in parent['folds']:
        name = old['name']; day = old['source_day']
        original_path = parent_root/'folds'/name/'manifest.json'
        if json.loads(original_path.read_text()) != old:
            raise ValueError('Parent study/fold mismatch')
        transform = SourceTransform.load(checked(old['transform_path'], old['transform_sha256']))
        groups = [read_npz(checked(g['path'], g['sha256'])) for g in old['training_groups']]
        source = read_npz(checked(old['source_path'], old['source_sha256']))
        labels = [g['label'] for g in old['training_groups']]
        if old['TF'] in labels:
            raise ValueError('Heldout condition present in training')
        all_groups = groups+[source]; all_labels = labels+[old['TF']]
        obs = metadata[day]
        all_ids = np.concatenate([g['cell_ids'].astype(str) for g in all_groups])
        rows = obs.index.astype(str).get_indexer(all_ids)
        if (rows < 0).any() or len(np.unique(rows)) != len(rows):
            raise ValueError('Source barcode alignment is not one-to-one')
        ordered = np.sort(rows)
        source_path = lock['data_hashes'][f'day{day}']['path']
        s, u = [read_count_rows(source_path, layer, ordered) for layer in ('spliced','unspliced')]
        offsets = np.cumsum([0]+[len(g['z']) for g in all_groups])
        positions = [np.searchsorted(ordered, rows[offsets[i]:offsets[i+1]]) for i in range(len(all_groups))]
        control_positions = [positions[i] for i,label in enumerate(labels) if label in ('AAVS1','CTRL')]
        if not control_positions:
            raise ValueError('No observed same-day training control for source reference')
        ci = np.concatenate(control_positions)
        reference = pooled_tokens(transform, s[ci], u[ci])
        embeddings = condition_embeddings(lock['data_hashes']['embeddings']['path'], all_labels)
        converted = []
        for i,(g,label,pi) in enumerate(zip(all_groups,all_labels,positions)):
            if not np.all(obs.iloc[ordered[pi]]['condition'].astype(str).to_numpy() == label):
                raise ValueError('Condition/barcode mismatch')
            np.testing.assert_allclose(transform.state(s[pi]), g['z'], rtol=0, atol=2e-5)
            tokens = pooled_tokens(transform,s[pi],u[pi])
            converted.append(convert(g,tokens,reference,embeddings[i],label,day,training=i<len(groups)))
        directory = run.directory/'folds'/name; directory.mkdir(parents=True)
        group_records = []
        for label,g in zip(labels,converted[:-1]):
            path = directory/f'train_{label}.npz'; np.savez_compressed(path,**g)
            group_records.append(dict(label=label,path=str(path.resolve()),sha256=sha256(path)))
        source_path_out = directory/'source.npz'; np.savez_compressed(source_path_out,**converted[-1])
        norm_path = directory/'normalization.npz'; np.savez_compressed(norm_path,**normalization(converted[:-1]))
        fm = dict(name=name,fold=name,TF=old['TF'],source_day=day,target_day=old['target_day'],
            training_groups=group_records, source_path=str(source_path_out.resolve()),source_sha256=sha256(source_path_out),
            normalization_path=str(norm_path.resolve()),normalization_sha256=sha256(norm_path),
            original_manifest=old,parent_manifest_sha256=sha256(original_path),
            heldout_future_read=False,mean_anchor=False,reference='same_day_observed_training_controls_raw_pool',
            source_count_path=source_path,source_count_sha256=lock['data_hashes'][f'day{day}']['sha256'])
        path = directory/'manifest.json'; path.write_text(json.dumps(fm,indent=2)+'\n')
        records.append(dict(name=name,path=str(path.resolve()),sha256=sha256(path)))
        run.json('status.json',dict(status='preparing',completed_folds=len(records),expected_folds=32))
        print(json.dumps(dict(status='fold_ready',fold=name,completed=len(records))),flush=True)
    if len(records)!=32:
        raise ValueError('Expected the complete fixed 32-fold RENGE grid')
    run.json('study_manifest.json',dict(status='ready',folds=records,parent_root=str(parent_root),
        config_path=str(config_path),config_sha256=sha256(config_path),heldout_future_read=False,
        source_only_transform=True,mean_anchor=False))
    run.json('status.json',dict(status='ready',folds=len(records)))


def load_fold(pack_directory, fold_name):
    root = Path(pack_directory)
    study = json.loads((root/'study_manifest.json').read_text())
    matches = [r for r in study['folds'] if r['name']==fold_name]
    if len(matches)!=1:
        raise ValueError('Unknown/duplicate fold')
    record = matches[0]
    fm = json.loads(checked(record['path'],record['sha256']).read_text())
    groups = [read_npz(checked(g['path'],g['sha256'])) for g in fm['training_groups']]
    source = read_npz(checked(fm['source_path'],fm['source_sha256']))
    if 'target' in source:
        raise ValueError('Source pack contains heldout target')
    norm = read_npz(checked(fm['normalization_path'],fm['normalization_sha256']))
    for g in groups+[source]:
        g['label'] = str(g['label']); g['time_hours'] = float(g['time_hours'])
    return fm,groups,source,norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True); parser.add_argument('--configuration',required=True)
    args = parser.parse_args(); prepare(args.output,args.configuration)
