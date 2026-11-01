"""Description: Unit tests for non-destructive historical source inventory."""
import json
from pathlib import Path
from pit_audit import audit, classify_fields


def test_classify_fields():
    """Description: Check industry, capitalization and disclosure metadata detection. Parameters: none. Returns: None."""
    result = classify_fields(['sw_industry', 'total_mv', 'announcement_date', 'close'])
    assert result['industry'] == ['sw_industry']
    assert result['capitalization'] == ['total_mv']
    assert result['pit_timing'] == ['announcement_date']


def test_audit_never_certifies_pit(tmp_path):
    """Description: Verify source discovery never claims PIT certification. Parameters: tmp_path (Path) pytest temporary directory. Returns: None."""
    (tmp_path / 'industry_history.csv').write_text('datetime,instrument,industry\n2020-01-01,SH600000,Bank\n', encoding='utf-8')
    output = tmp_path / 'report.json'
    result = audit(tmp_path, output)
    assert result['pit_certified'] is False
    assert result['candidate_filename_files'] == 1
    assert json.loads(output.read_text(encoding='utf-8'))['pit_certified'] is False
