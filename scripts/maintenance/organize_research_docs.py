"""One-time, auditable document moves; preserve experimental artifacts and protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r'(!?\[[^\]\n]*\]\()([^\)\n]+)(\))')
CURRENT = {'w13_time_results.md','w13_time_model.md','w13_project_position_and_next_steps.md'}
MAINLINE = {'w5_joint_kinetics_results.md','w6_local_router_results.md','w7_router_horizon_results.md',
            'w7_router_scaling_results.md','w8_router_longtrain_results.md','w9_large_model_audit.md',
            'w10_recipe_bridge_results.md'}
DATA = {'w4_external_data_audit.md','w4_renge_data_audit.md','w5_scifate2_data_audit.md'}
BOUNDARIES = {'w3_results.md','w4_results.md','w5_scntseq_results.md',
              'w5_scntseq_repair_results.md','w5_scntseq_U_attribution_results.md'}
RETRACTED = {'w1_results_20261009.md','w2_results_20261010.md','w2_gfg_v2_results.md'}
AUDITS = {'velocity_code_review_20261014.md'}
# Preserve externally shared links and links embedded in immutable run reports.
PUBLIC_ALIASES = {'docs/research_archive_20261017.md','docs/w13_time_results.md',
                  'docs/w13_time_model.md','docs/w13_project_position_and_next_steps.md',
                  'docs/w10_recipe_bridge_results.md','docs/w9_large_model_audit.md'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_target(parent, url):
    url = url.strip()
    if not url or url.startswith(('#','/','http:','https:','mailto:','data:')):
        return None,None
    if url.startswith('<') and '>' in url:
        split = url.index('>');path = url[1:split];tail = url[split+1:]
    else:
        parts = url.split(' ',1);path = parts[0];tail = ' '+parts[1] if len(parts)>1 else ''
    base,sep,fragment = path.partition('#')
    target = (parent/unquote(base)).resolve()
    if not target.is_relative_to(ROOT):return None,None
    return target.relative_to(ROOT).as_posix(),('#'+fragment if sep else '')+tail


def pinned_protocols():
    result = set()
    for top in ('scripts','tests','src','configs'):
        for path in (ROOT/top).rglob('*'):
            if path.is_file() and path.suffix in {'.py','.json','.yaml','.yml'}:
                result.update(re.findall(r'PLANS/[A-Za-z0-9_.-]+\.md',path.read_text()))
    return {name for name in result if (ROOT/name).is_file()}


def design():
    routes = {};kinds = {};pinned = pinned_protocols()
    for path in sorted((ROOT/'docs').glob('*.md')):
        name = path.name
        if name in CURRENT:folder,status = 'current','current'
        elif name in MAINLINE:folder,status = 'evidence/mainline','supporting_milestone'
        elif name in DATA:folder,status = 'evidence/data_audits','data_provenance'
        elif name in BOUNDARIES:folder,status = 'evidence/boundaries','valid_boundary_or_mixed_result'
        elif name in RETRACTED:folder,status = 'history/superseded_claims','historical_with_retracted_inferences'
        elif name in AUDITS:folder,status = 'corrections/audits','correction_audit'
        elif name=='research_archive_20261017.md':folder,status = 'history/record_exports','publication_inventory'
        elif name=='veloroute_experiment_ledger_20260914.md':folder,status = 'history/experiment_ledgers','historical_ledger'
        elif name.startswith('veloroute_'):folder,status = 'history/legacy_architecture','superseded_architecture'
        else:folder,status = 'history/explorations','historical_exploration'
        old = path.relative_to(ROOT).as_posix();routes[old]=f'docs/{folder}/{name}';kinds[old]=status
    for path in sorted((ROOT/'PLANS').glob('*.md')):
        old = path.relative_to(ROOT).as_posix()
        if old in pinned:continue
        if path.name=='paper_outline_20261017.md':folder = 'superseded_paper'
        elif path.name.startswith(('w2','w3','veloroute_')):folder = 'early_exploration'
        else:folder = 'completed_milestones'
        routes[old]=f'PLANS/history/{folder}/{path.name}';kinds[old]='superseded_or_completed_plan'
    routes['HANDOFF.md']='docs/history/handoffs/HANDOFF_before_organization_20261017.md'
    routes['README.md']='docs/history/handoffs/README_before_organization_20261017.md'
    kinds['HANDOFF.md']=kinds['README.md']='byte_exact_front_page_snapshot'
    return routes,kinds,pinned


def banner(status,parent):
    state = os.path.relpath(ROOT/'docs/current/STATUS.md',parent).replace(os.sep,'/')
    corrections = os.path.relpath(ROOT/'docs/corrections/CLAIMS.md',parent).replace(os.sep,'/')
    labels = {
        'current':'当前主线材料；研究结论以最新状态页为准。',
        'supporting_milestone':'有效前期证据；其中“当前/下一步”仅指当时阶段，不是最新执行队列。',
        'data_provenance':'有效数据来源与处理审计；后续任务状态按最新状态页更新。',
        'valid_boundary_or_mixed_result':'保留的阴性、边界或混合结果，不因未获阳性而归为错误。',
        'historical_with_retracted_inferences':'历史报告，含已撤回或被审计限制的推断；旧数值不自动支持旧总括结论。',
        'correction_audit':'纠错依据；审计有效，其中的旧阶段状态/待办不代表当前进度。',
        'publication_inventory':'历史发布清单；其文件路径和哈希对应发布当时，不代表当前文档布局。',
        'historical_ledger':'历史实验总账，不作为当前结论或启动指令。',
        'superseded_architecture':'历史架构/执行方案，已非当前主线；并非其中每个实验都无效。',
        'historical_exploration':'历史探索或备选路线；保留证据，不作为当前执行计划。',
        'superseded_or_completed_plan':'历史或已完成阶段计划；不得据此自动恢复旧实验。',
    }
    return f'> 归档状态：{labels[status]} [当前状态]({state}) · [撤回与更正清单]({corrections})\n'


def rewrite(text,old,new,routes):
    old_parent=(ROOT/old).parent;new_parent=(ROOT/new).parent
    def replace(match):
        target,tail=local_target(old_parent,match.group(2))
        if target is None:return match.group(0)
        destination=target if target in {'HANDOFF.md','README.md'} else routes.get(target,target)
        relative=os.path.relpath(ROOT/destination,new_parent).replace(os.sep,'/')
        return match.group(1)+relative+tail+match.group(3)
    text=LINK.sub(replace,text)
    # Root-relative literal paths in mutable narrative text, not immutable artifacts.
    for source,dest in sorted(routes.items(),key=lambda pair:-len(pair[0])):
        if '/' in source:text=text.replace(source,dest)
    return text


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    manifest_path=ROOT/'docs/organization_manifest_20261017.json'
    if manifest_path.exists():raise ValueError('Organization is already recorded; do not rerun blindly')
    routes,kinds,pinned=design()
    aliases=set(PUBLIC_ALIASES)
    frozen=list((ROOT/'outputs').rglob('*.md'))+[ROOT/name for name in pinned]
    for path in frozen:
        for match in LINK.finditer(path.read_text()):
            target,_=local_target(path.parent,match.group(2))
            if target in routes and target not in {'HANDOFF.md','README.md'}:aliases.add(target)
    protected={name:sha(ROOT/name) for name in sorted(pinned)}
    protected['docs/research_archive_20261017.json']=sha(ROOT/'docs/research_archive_20261017.json')
    for path in (ROOT/'scripts/phase0').glob('w*.py'):
        protected[path.relative_to(ROOT).as_posix()]=sha(path)
    original={old:sha(ROOT/old) for old in routes}
    print(json.dumps(dict(moves=len(routes),pinned_protocols=sorted(pinned),compatibility_aliases=sorted(aliases)),indent=2))
    if not args.apply:return
    for old,new in routes.items():
        destination=ROOT/new
        if destination.exists():raise ValueError(f'Destination exists: {new}')
        destination.parent.mkdir(parents=True,exist_ok=True)
        (ROOT/old).rename(destination)
    for old,new in routes.items():
        if kinds[old]=='byte_exact_front_page_snapshot':continue
        path=ROOT/new;text=rewrite(path.read_text(),old,new,routes)
        lines=text.splitlines(keepends=True)
        if lines and lines[0].startswith('#'):
            text=lines[0]+'\n'+banner(kinds[old],path.parent)+'\n'+''.join(lines[1:]).lstrip('\n')
        else:text=banner(kinds[old],path.parent)+'\n'+text
        path.write_text(text)
    # Update other mutable Markdown references, but not pinned protocol bytes or snapshots.
    moved=set(routes.values())
    for top in ('docs','PLANS'):
        for path in (ROOT/top).rglob('*.md'):
            rel=path.relative_to(ROOT).as_posix()
            if rel in moved or rel in protected:continue
            before=path.read_text();after=rewrite(before,rel,rel,routes)
            if after!=before:path.write_text(after)
    for old in sorted(aliases):
        if old not in routes:continue
        path=ROOT/old;destination=ROOT/routes[old]
        path.write_text('# 已整理：兼容入口\n\n'
            '本路径仅保留给旧链接；完整文档已分类存放。请勿把旧报告当作当前执行计划。\n\n'
            f'[打开分类后的原文]({os.path.relpath(destination,path.parent).replace(os.sep,"/")})\n\n'
            '[当前项目入口]('+os.path.relpath(ROOT/'docs/current/STATUS.md',path.parent).replace(os.sep,'/')+')\n')
    for name,digest in protected.items():
        if sha(ROOT/name)!=digest:raise ArithmeticError(f'Protected content changed: {name}')
    records=[dict(original_path=old,path=new,status=kinds[old],original_sha256=original[old],
                  organized_sha256=sha(ROOT/new),compatibility_alias=old in aliases) for old,new in routes.items()]
    manifest=dict(status='DOCUMENT_ORGANIZATION_COMPLETE',records=records,protected_sha256=protected,
        note='Only document placement/navigation/status changed; outputs and scientific training code preserved')
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print('DOCUMENT_MOVES_COMPLETE')


if __name__=='__main__':main()
