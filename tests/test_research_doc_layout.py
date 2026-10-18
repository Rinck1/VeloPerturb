import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/maintenance'))
from check_research_docs import ROOT,navigation_errors


def layout():return json.loads((ROOT/'docs/organization_manifest_20261017.json').read_text())


def test_current_pages_and_all_category_readmes_have_resolvable_local_links():
    assert navigation_errors()==[]


def test_negative_results_are_separate_from_retracted_inferences():
    records={r['original_path']:r for r in layout()['records']}
    assert records['docs/w4_results.md']['path'].startswith('docs/evidence/boundaries/')
    assert records['docs/w2_gfg_v2_results.md']['path'].startswith('docs/history/superseded_claims/')
    assert (ROOT/'docs/corrections/CLAIMS.md').is_file()
    assert records['docs/w13_time_results.md']['path'].startswith('docs/current/')


def test_old_front_pages_are_retained_byte_exact():
    for row in layout()['records']:
        if row['status']=='byte_exact_front_page_snapshot':
            assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['original_sha256']


def test_compatibility_aliases_point_to_existing_canonical_documents():
    for row in layout()['records']:
        assert (ROOT/row['path']).is_file()
        if row['compatibility_alias']:
            assert '兼容入口' in (ROOT/row['original_path']).read_text()


def test_frozen_protocols_stay_at_their_original_paths_and_bytes():
    for path,expected in layout()['protected_sha256'].items():
        if path.startswith('PLANS/'):
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
