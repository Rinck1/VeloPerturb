"""Same metric definitions as the supplied CellFlow report, explicit cohort binding.

These supplementary scores never replace the registered energy/mode endpoints.
Only call the file evaluator after the complete prediction grid is frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist,pdist
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from .artifacts import Run,load_config,object_hash,save_csv,save_json,sha256
from .kang_shared_metric_kernels import prdc,projected_peak_metrics,UPSTREAM_SHA256
from .latent import FrozenSplicingTransform,load_pack


SETTINGS_PATH=Path(__file__).resolve().parents[2]/'configs/veloroute_kang_shared_metrics_20260914.yaml'
METRICS=('mean_gene_r2','delta_gene_r2','gene_mean_rmse','mmd2_unbiased_eval_pca20',
         'variance_trace_ratio_eval_pca20','precision','recall','density','coverage')
PEAK_METRICS=('peak_f1','peak_recall','peak_precision','missing_peak_rate','spurious_peak_rate',
              'mode_mass_tv','rare_peak_recall','valley_mass_excess')
COMPARISON_BINDINGS=('split_id','source_ids_hash','target_ids_hash','training_ids_hash',
                    'gene_ids_hash','normalization','evaluation_space_id','sampling_protocol_id','cell_types')


def array_hash(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()


def numeric_mean(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def compare_bindings(a,b):
    missing=[k for k in COMPARISON_BINDINGS if k not in a or k not in b]
    different=[k for k in COMPARISON_BINDINGS if k in a and k in b and a[k]!=b[k]]
    return dict(directly_comparable=not missing and not different,missing_bindings=missing,
                mismatched_bindings=different)


def fit_evaluation_space(train_source,train_target,*,gene_ids,training_ids,settings):
    """No held-out values are arguments: changing them cannot change this fit."""
    training=np.concatenate((train_source,train_target))
    if training.ndim!=2 or training.shape!=(len(training_ids),len(gene_ids)) or not np.isfinite(training).all():
        raise ValueError('Invalid training-only evaluation matrix')
    if len(set(training_ids))!=len(training_ids):raise ValueError('Duplicate evaluation fit IDs')
    pca=PCA(n_components=settings['evaluation_components'],svd_solver='randomized',random_state=settings['evaluation_seed'])
    pca.fit(training)
    identity=dict(training_ids_hash=object_hash(sorted(training_ids)),gene_ids_hash=object_hash(list(gene_ids)),
        normalization=settings['normalization'],components_sha256=array_hash(pca.components_),mean_sha256=array_hash(pca.mean_))
    return pca,dict(**identity,evaluation_space_id=object_hash(identity))


def _gene_values(fold,role,side,latent,meta,transform):
    path=fold/(f'{role}_gene_source.npz' if side=='source' else f'{role}_target_genes.npz')
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files if k!='metadata_json'}
        gm=json.loads(str(archive['metadata_json']))
    wanted='source_gene_US' if side=='source' else 'target_gene_space'
    if (gm['role'],gm['side'],gm['transform_hash'])!=(role,wanted,meta['transform_hash']):
        raise ValueError('Gene-expression role/transform mismatch')
    if not np.array_equal(data['cell_ids'],latent['cell_ids']):raise ValueError('Gene/latent cell IDs differ')
    genes=np.asarray(transform.gene_ids)[transform.selected]
    if not np.array_equal(genes,data['gene_ids']):raise ValueError('Gene panel/order mismatch')
    if side=='source':
        if gm['source_hash']!=sha256(fold/f'{role}_source.npz'):raise ValueError('Source archive hash mismatch')
        counts=data['gene_us']
        if counts.shape!=(len(latent['z']),2*len(genes)) or (counts<0).any():raise ValueError('Invalid gene-level S/U')
        values=np.log1p(counts[:,len(genes):])
    else:
        if not np.array_equal(data['conditions'],latent['conditions']):raise ValueError('Target group mismatch')
        values=data['gene_logspliced']
    if values.shape!=(len(latent['z']),len(genes)) or not np.isfinite(values).all():raise ValueError('Invalid gene expressions')
    np.testing.assert_allclose((values-transform.mean)@transform.components.T,latent['z'],rtol=1e-4,atol=2e-5)
    return values


def load_evaluation_panel(fold,donor,settings):
    fold=Path(fold);transform=FrozenSplicingTransform.load(fold/'transform.npz');parts={}
    fingerprint=sha256(fold/'transform.npz')
    for role in ('train','validation'):
        for side in ('source','target'):
            p,m=load_pack(fold/f'{role}_{side}.npz',expected_side=side)
            if m['role']!=role or m['heldout_donor']!=donor or m['transform_hash']!=fingerprint:
                raise ValueError('Wrong fold/role for supplementary evaluation')
            if role=='validation' and m['role_semantics']!='outer_heldout':raise ValueError('Not an outer holdout')
            if any((g.split('|',1)[0]==donor)!=(role=='validation') for g in p['conditions']):
                raise ValueError('Donor identity leaks across fit/holdout')
            parts[role,side]=dict(pack=p,genes=_gene_values(fold,role,side,p,m,transform))
    train_ids=np.concatenate([parts['train',side]['pack']['cell_ids'] for side in ('source','target')]).tolist()
    test_ids=np.concatenate([parts['validation',side]['pack']['cell_ids'] for side in ('source','target')]).tolist()
    if set(train_ids)&set(test_ids):raise ValueError('Fit and held-out cells overlap')
    if object_hash(sorted(train_ids))!=transform.fit_ids_hash:raise ValueError('Evaluation fit scope changed')
    genes=np.asarray(transform.gene_ids)[transform.selected].tolist()
    pca,identity=fit_evaluation_space(parts['train','source']['genes'],parts['train','target']['genes'],
        gene_ids=genes,training_ids=train_ids,settings=settings)
    for part in parts.values():part['eval_z']=pca.transform(part['genes'])
    return dict(parts=parts,transform=transform,pca=pca,identity=identity,gene_ids=genes,training_ids=train_ids)


def reference_bandwidth(reference,settings):
    rng=np.random.default_rng(settings['evaluation_seed'])
    indices=rng.choice(len(reference),min(settings['mmd_bandwidth_reference_limit'],len(reference)),replace=False)
    distances=pdist(np.asarray(reference)[indices],metric='sqeuclidean')
    positive=distances[distances>0]
    return float(np.median(positive)) if len(positive) else None


def mmd2_unbiased(real,predicted,bandwidth_squared):
    real,predicted=[np.asarray(a,dtype=np.float64) for a in (real,predicted)]
    n,m=len(real),len(predicted)
    if min(n,m)<2 or bandwidth_squared is None or not np.isfinite(bandwidth_squared) or bandwidth_squared<=0:
        return None
    aa=np.exp(-cdist(real,real,'sqeuclidean')/(2*bandwidth_squared))
    bb=np.exp(-cdist(predicted,predicted,'sqeuclidean')/(2*bandwidth_squared))
    ab=np.exp(-cdist(real,predicted,'sqeuclidean')/(2*bandwidth_squared))
    # Do NOT clip negative finite-sample U-statistic estimates.
    return float((aa.sum()-np.trace(aa))/(n*(n-1))+(bb.sum()-np.trace(bb))/(m*(m-1))-2*ab.mean())


def sample_metrics(real_genes,predicted_genes,source_genes,real_z,predicted_z,bandwidth_squared,*,k=5):
    arrays=[np.asarray(a,dtype=np.float64) for a in (real_genes,predicted_genes,source_genes,real_z,predicted_z)]
    if any(a.ndim!=2 or not len(a) or not np.isfinite(a).all() for a in arrays):raise ValueError('Invalid score arrays')
    real,pred,source,y,z=arrays
    if not (real.shape==pred.shape==source.shape) or y.shape!=z.shape or len(y)!=len(real):
        raise ValueError('Balanced gene and distribution score shapes differ')
    my,mp,mx=[a.mean(0) for a in (real,pred,source)]
    row=dict(mean_gene_r2=float(r2_score(my,mp)),delta_gene_r2=float(r2_score(my-mx,mp-mx)),
        gene_mean_rmse=float(np.sqrt(np.mean((mp-my)**2))),mmd2_unbiased_eval_pca20=mmd2_unbiased(y,z,bandwidth_squared),
        variance_trace_ratio_eval_pca20=None,**{name:None for name in ('precision','recall','density','coverage')})
    if len(y)>1:
        denominator=float(y.var(0,ddof=1).sum())
        if denominator>0:row['variance_trace_ratio_eval_pca20']=float(z.var(0,ddof=1).sum()/denominator)
    if len(y)>k:row.update(prdc(y,z,k=k))
    return row


def score_panel_prediction(panel,prediction,*,donor,seed,arm,definitions,settings):
    source=panel['parts']['validation','source'];target=panel['parts']['validation','target']
    reference=panel['parts']['train','target'];pca=panel['pca']
    if not np.array_equal(prediction['source_ids'],source['pack']['cell_ids']) or not np.array_equal(prediction['conditions'],source['pack']['conditions']):
        raise ValueError('Prediction/source barcode order differs')
    # Sampled single endpoints, never flattened experts or averages over modes.
    decoded=panel['transform'].decode(prediction['z'])
    negative_fraction=float((decoded<0).mean())
    if settings['clip_negative_decoded_expression']:decoded=np.maximum(decoded,0)
    predicted_z=pca.transform(decoded);rows=[];peaks=[];audit=[]
    for label in sorted(set(source['pack']['conditions'])):
        ct=label.split('|',1)[1]
        si=np.flatnonzero(source['pack']['conditions']==label)
        ti=np.flatnonzero(target['pack']['conditions']==label)
        ri=np.flatnonzero([g.split('|',1)[1]==ct for g in reference['pack']['conditions']])
        n=min(len(si),len(ti),len(ri),settings['max_samples'])
        if not n:raise ValueError('Source cohort lacks target/reference for supplementary scoring')
        ref=reference['eval_z'][ri];bandwidth=reference_bandwidth(ref,settings)
        for sampling_seed in settings['sampling_seeds']:
            rng=np.random.default_rng([settings['evaluation_seed'],sampling_seed])
            yt=rng.choice(ti,n,replace=False);xs=rng.choice(si,n,replace=False);rr=rng.choice(ri,n,replace=False)
            row=dict(donor=donor,seed=seed,arm=arm,cell_type=ct,sampling_seed=sampling_seed,n=n,
                adequate_test_count=n>=settings['minimum_balanced_cells'],gene_count=len(panel['gene_ids']),
                evaluation_components=pca.n_components_,evaluation_space_id=panel['identity']['evaluation_space_id'],
                mmd_bandwidth_squared=bandwidth,negative_prediction_fraction_before_clipping=negative_fraction,
                **sample_metrics(target['genes'][yt],decoded[xs],source['genes'][xs],target['eval_z'][yt],predicted_z[xs],bandwidth,k=settings['prdc_k']))
            rows.append(row)
            audit.append(dict(donor=donor,seed=seed,arm=arm,cell_type=ct,sampling_seed=sampling_seed,n=n,
                source_ids_hash=object_hash(source['pack']['cell_ids'][xs].tolist()),
                real_ids_hash=object_hash(target['pack']['cell_ids'][yt].tolist()),
                reference_ids_hash=object_hash(reference['pack']['cell_ids'][rr].tolist())))
            if sampling_seed in settings['peak_sampling_seeds']:
                for axis in range(min(settings['peak_axes'],pca.n_components_)):
                    result=projected_peak_metrics(ref[:,axis],target['eval_z'][yt,axis],predicted_z[xs,axis],**settings['peaks'])
                    result.update(donor=donor,seed=seed,arm=arm,cell_type=ct,axis=axis+1,sampling_seed=sampling_seed,
                        training_mode_approved=bool(definitions[ct]['approved']))
                    result['candidate_multipeak']=bool(result['status']=='ok' and result.get('n_real_peaks',0)>=2)
                    result['approved_multipeak']=result['candidate_multipeak'] and result['training_mode_approved']
                    peaks.append(result)
    return rows,peaks,audit,decoded,predicted_z


def summarize_metrics(rows,expected_types,*,minimum_cells=50):
    """Technical subsamples, donors, and cell types must not be pooled as replicates."""
    keys=sorted({(r['arm'],r['seed'],r['donor'],r['cell_type']) for r in rows})
    per_donor=[]
    for arm,seed,donor,ct in keys:
        selected=[r for r in rows if (r['arm'],r['seed'],r['donor'],r['cell_type'])==(arm,seed,donor,ct)]
        per_donor.append(dict(arm=arm,seed=seed,donor=donor,cell_type=ct,n=min(r['n'] for r in selected),
            resampling_repeats=len(selected),**{m:numeric_mean(r.get(m) for r in selected) for m in METRICS}))
    per_type=[];macro=[]
    for arm,seed in sorted({(r['arm'],r['seed']) for r in per_donor}):
        for ct in expected_types:
            selected=[r for r in per_donor if (r['arm'],r['seed'],r['cell_type'])==(arm,seed,ct) and r['n']>=minimum_cells]
            per_type.append(dict(arm=arm,seed=seed,cell_type=ct,n_eligible_donors=len(selected),
                **{m:numeric_mean(r[m] for r in selected) for m in METRICS}))
        eligible=[r for r in per_type if (r['arm'],r['seed'])==(arm,seed) and r['n_eligible_donors']]
        complete=len(eligible)==len(expected_types)
        row=dict(arm=arm,seed=seed,complete_type_panel=complete,n_expected_types=len(expected_types),n_eligible_types=len(eligible),
            eligible_types='|'.join(r['cell_type'] for r in eligible),aggregation='resamples_then_donors_then_types_equal_weight')
        row.update({m:numeric_mean(r[m] for r in eligible) if complete and all(r[m] is not None for r in eligible) else None for m in METRICS})
        row.update({'available_types_'+m:numeric_mean(r[m] for r in eligible) for m in METRICS})
        macro.append(row)
    return per_donor,per_type,macro


def summarize_peaks(peaks,identities):
    summaries=[]
    for arm,seed in identities:
        for scope,flag in [('candidate_projection_diagnostic','candidate_multipeak'),('training_approved_type_projection_diagnostic','approved_multipeak')]:
            selected=[r for r in peaks if (r['arm'],r['seed'])==(arm,seed) and r[flag]]
            # Average axes/repeats inside a donor/type, then donors, then types.
            by_type={}
            for ct in sorted({r['cell_type'] for r in selected}):
                donor_means=[]
                for donor in sorted({r['donor'] for r in selected if r['cell_type']==ct}):
                    group=[r for r in selected if r['donor']==donor and r['cell_type']==ct]
                    donor_means.append({m:numeric_mean(r.get(m) for r in group) for m in PEAK_METRICS})
                by_type[ct]={m:numeric_mean(r[m] for r in donor_means) for m in PEAK_METRICS}
            summaries.append(dict(arm=arm,seed=seed,scope=scope,status='diagnostic_only_not_D4' if selected else 'not_evaluable_no_eligible_multipeak',
                n_eligible_projections=len(selected),n_eligible_types=len(by_type),n_eligible_donors=len({r['donor'] for r in selected}),
                **{m:numeric_mean(r[m] for r in by_type.values()) for m in PEAK_METRICS},
                **{'n_valid_'+m:sum(r.get(m) is not None for r in selected) for m in PEAK_METRICS}))
    return summaries


def _verify_frozen_records(config,frozen):
    identities=set()
    for r in frozen:
        identity=(r['donor'],r['seed'],r['arm'])
        if identity in identities:raise ValueError('Duplicated frozen prediction')
        identities.add(identity);path=Path(r['path'])
        manifest=json.loads((path/'prediction_manifest.json').read_text());provenance=json.loads((path/'provenance.json').read_text())
        digest=sha256(path/'predictions.npz')
        if provenance['status']!='complete' or manifest['future_target_read'] or manifest['protocol_hash']!=object_hash(config):
            raise ValueError('Unfrozen or non-source-only prediction')
        if identity!=(manifest['donor'],manifest['seed'],manifest['arm']):raise ValueError('Frozen record identity mismatch')
        if digest!=r['sha256'] or digest!=manifest['prediction_sha256'] or digest!=provenance['outputs']['predictions.npz']:
            raise ValueError('Changed prediction archive')
    arms={r['arm'] for r in frozen}
    expected={(d,s,a) for d in config['donors'] for s in config['experiments']['seeds'] for a in arms}
    if not arms or identities!=expected:raise ValueError('Incomplete donor/seed prediction grid')


def evaluate_shared_metrics(config,output,frozen,*,settings=None):
    settings=load_config(SETTINGS_PATH) if settings is None else settings
    _verify_frozen_records(config,frozen)  # before opening any held-out outcomes
    root=Path(config['root']);expected_types=[config['cell_types']['primary']]+config['cell_types']['secondary']
    inputs=[SETTINGS_PATH,*[Path(r['path'])/'prediction_manifest.json' for r in frozen]]
    for donor in config['donors']:
        fold=root/'folds'/donor
        inputs.extend([fold/'transform.npz',*[fold/f'{role}_{name}.npz' for role in ('train','validation')
                      for name in ('source','target','gene_source','target_genes')],root/'modes'/donor/'mode_definitions.json'])
    with Run(output,stage='Kang_CellFlow_definition_matched_supplementary_metrics',kind='development',
             config=settings,inputs=inputs,seed=settings['sampling_seeds']) as run:
        save_json(run.directory/'frozen_grid.json',frozen)
        all_rows=[];all_peaks=[];sampling_audit=[];manifests=[]
        for donor in config['donors']:
            panel=load_evaluation_panel(root/'folds'/donor,donor,settings)
            definitions_path=root/'modes'/donor/'mode_definitions.json'
            definitions=json.loads(definitions_path.read_text())
            modeprov=json.loads(definitions_path.with_name('provenance.json').read_text())
            if modeprov['status']!='complete' or sha256(definitions_path)!=modeprov['outputs']['mode_definitions.json']:
                raise ValueError('Mode definition not frozen')
            folder=run.directory/'folds'/donor;folder.mkdir(parents=True)
            pca=panel['pca'];source=panel['parts']['validation','source'];target=panel['parts']['validation','target']
            manifest=dict(**panel['identity'],donor=donor,split_id=object_hash(dict(method=config['split']['method'],donor=donor,
                fit_ids=panel['identity']['training_ids_hash'])),source_ids_hash=object_hash(source['pack']['cell_ids'].tolist()),
                target_ids_hash=object_hash(target['pack']['cell_ids'].tolist()),sampling_protocol_id=object_hash(settings),
                cell_types=expected_types,actual_fold_types=sorted({g.split('|',1)[1] for g in source['pack']['conditions']}),
                gene_count=len(panel['gene_ids']),evaluation_components=int(pca.n_components_),
                dataset_split='leave_one_donor_out_not_external_CellFlow_IID',decoded_negatives_clipped=settings['clip_negative_decoded_expression'],
                formal_D4=False,source_only_prediction_files=True)
            manifests.append(manifest);save_json(folder/'comparison_manifest.json',manifest)
            with (folder/'evaluation_basis.npz').open('xb') as f:
                np.savez_compressed(f,components=pca.components_,mean=pca.mean_,gene_ids=np.array(panel['gene_ids']),
                    training_ids=np.array(panel['training_ids']),source_ids=source['pack']['cell_ids'],target_ids=target['pack']['cell_ids'])
            for record in [r for r in frozen if r['donor']==donor]:
                with np.load(Path(record['path'])/'predictions.npz',allow_pickle=False) as f:prediction={k:f[k] for k in f.files}
                rows,peaks,audit,decoded,z=score_panel_prediction(panel,prediction,donor=donor,seed=record['seed'],arm=record['arm'],definitions=definitions,settings=settings)
                all_rows.extend(rows);all_peaks.extend(peaks);sampling_audit.extend(audit)
                # Portable measured-gene outputs for a common evaluator; not 4634-gene imputations.
                out=folder/f"{record['arm']}_seed{record['seed']}.npz"
                with out.open('xb') as f:
                    np.savez_compressed(f,gene_logspliced=decoded.astype('float32'),X_eval_pca=z.astype('float32'),
                        gene_ids=np.array(panel['gene_ids']),source_ids=prediction['source_ids'],conditions=prediction['conditions'],
                        comparison_manifest_json=np.array(json.dumps(manifest,sort_keys=True)))
        per_donor,per_type,macro=summarize_metrics(all_rows,expected_types,minimum_cells=settings['minimum_balanced_cells'])
        peak_summary=summarize_peaks(all_peaks,sorted({(r['arm'],r['seed']) for r in frozen}))
        save_csv(run.directory/'per_sampling_run.csv',all_rows)
        save_csv(run.directory/'per_donor_cell_type.csv',per_donor)
        save_csv(run.directory/'per_cell_type.csv',per_type)
        save_csv(run.directory/'primary_metrics.csv',macro)
        save_csv(run.directory/'sampling_audit.csv',sampling_audit)
        save_json(run.directory/'peak_diagnostics.json',all_peaks)
        save_csv(run.directory/'peak_summary.csv',peak_summary)
        external=Path(settings['external_baseline_manifest'])
        external_audit=dict(status='NOT_DIRECTLY_COMPARABLE_TO_EXISTING_CellFlow_TABLE',same_metric_formulas=True,
            differences=['donor_holdout_vs_cell_IID','fold_selected_genes_vs_4634_genes','spliced_library_vs_published_total_RNA',
                         'different_training_fitted_PCA20_bases','different_QC_and_cell_ID_cohorts'],
            required_bindings=list(COMPARISON_BINDINGS),current_manifests=manifests)
        if external.is_file():external_audit.update(external_manifest=json.loads(external.read_text()),external_manifest_sha256=sha256(external))
        save_json(run.directory/'comparability_audit.json',external_audit)
        result=dict(status='SUPPLEMENTARY_METRICS_COMPLETE',predictions=len(frozen),metrics=list(METRICS),peak_metrics=list(PEAK_METRICS),
            same_formula_not_same_dataset_protocol=True,existing_CellFlow_table_directly_comparable=False,
            original_primary_endpoints_unchanged=True,source_metric_code_sha256=UPSTREAM_SHA256,
            peak_empty_policy='null_not_zero',aggregation=settings['aggregation'])
        save_json(run.directory/'summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Kang supplementary distribution and peak metrics\n\n'+json.dumps(result,indent=2)
            +'\n\nDefinitions match the supplied CellFlow report. They do not make different splits, genes, normalizations or PCA bases comparable. '
             'Technical resamples are averaged first, donors second, types third; missing six-type panels are explicitly NA. '
             'Peak metrics are one-dimensional diagnostics, not proven fate branching. Existing registered primary endpoints remain separate.\n')
    return result
