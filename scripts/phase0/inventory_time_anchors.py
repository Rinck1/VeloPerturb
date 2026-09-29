"""Inventory metadata/files only, never expression or confirmation labels."""
import argparse
from pathlib import Path
import shutil
from veloroute.artifacts import Run, read_csv, save_csv, save_json
from veloroute.raw import plan_raw

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
manifest = Path('outputs/veloroute_phase0/renge_structural_20260912/run_manifest.csv')
qc = manifest.parent/'data_qc.csv'
data_root = Path('/data/yuchang/veloroute_timeanchors_20260914')
published = Path('/data/dataset/perturbation_v1/renge_2023')
selected = [r for r in read_csv(manifest) if int(r['day']) in (2, 3)]
with Run(args.output, stage='time_anchor_metadata_inventory', kind='engineering',
         config=dict(confirmation='SEALED', response_values_read=False,
                     allowed_gpus=[0, 1, 2, 3], reserved_gpus=[4, 5, 6, 7]),
         inputs=[manifest, qc]) as run:
    rows = []
    for row in selected:
        accession = row['run_accession']
        day = int(row['day'])
        released = [published/f'day{day}'/f for f in ('features.tsv.gz', 'barcodes.tsv.gz', 'matrix.mtx.gz')]
        raw_candidates = [root/'sra'/accession/f'{accession}.sra'
                          for root in (Path('data/renge/raw'), data_root/'raw')]
        rows.append(dict(day=day, accession=accession, library_type=row['library_type'],
                         ena_nontechnical_fastq_bytes=int(row['fastq_bytes']),
                         sra_bytes_known=bool(row['sra_bytes']),
                         published_matrix_files_exist=all(p.is_file() for p in released),
                         sra_located=any(p.is_file() for p in raw_candidates),
                         processed_su_located=(Path('data/renge/processed_release_v1')/f'day{day}'/f'day{day}.h5ad').is_file()))
    save_csv(run.directory/'metrics.csv', rows)
    for day in (2, 3):
        save_json(run.directory/f'day{day}_tasks.json', plan_raw(read_csv(manifest), day,
                  'tools/sratoolkit.3.4.1-ubuntu64/bin', data_root/'raw'))
    summary = dict(confirmation='SEALED', raw_runs_required=len(rows),
                   raw_runs_located=sum(r['sra_located'] for r in rows),
                   day2_published_cells=4710, day3_published_cells=6091,
                   cells_source='frozen_structural_audit_not_recounted',
                   ena_fastq_bytes=sum(r['ena_nontechnical_fastq_bytes'] for r in rows),
                   ena_bytes_exclude_technical_reads_and_are_not_SRA_or_scratch_sizes=True,
                   workspace_free_bytes=shutil.disk_usage('.').free,
                   dedicated_root=str(data_root), dedicated_free_bytes=shutil.disk_usage(data_root).free,
                   next_action='day3_raw_CPU_acquisition_then_layout_USA_QC_train_only_temporal_protocol',
                   scientifically_independent_confirmation=False)
    save_json(run.directory/'summary.json', summary)
    (run.directory/'RESULTS.md').write_text(
        '# day2/day3 时间锚资产审计\n\n'
        '发布矩阵已落地：day2 4,710 细胞、day3 6,091 细胞（冻结结构审计数字）。'
        '当前指定目录未发现两天的成品 S/U。每一天是两条 mRNA 和两条 gRNA 文库，不能按文件大小混用。\n\n'
        f'ENA 非 technical FASTQ 合计 {summary["ena_fastq_bytes"] / 1e9:.2f} GB；'
        '这不是 NCBI SRA、完整 FASTQ 或 scratch 的容量。新原始文件放在专属数据盘目录，保留全部既有文件。\n\n'
        '先处理 day3（原始数据获取不占 GPU），每条 run 下载/校验/提取/压缩；'
        '布局和 S/U QC 过关后，另冻结训练折内的时间锚比较协议，再运行模型。'
        '新增时间点不是新的独立供体或独立确认集；不读取最终确认条件响应。\n')
