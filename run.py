#!/usr/bin/env python3
"""Read archived evidence or print the entry point for a GPU rerun."""
import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence'
STEPS = [
    ('base', '27_full_model_aot.py', None),
    ('layer', '25B_a100_or_blackwell_static_group_ladder_final.py', 'prepared_core_bridge_bundle.zip'),
    ('tiling', '30_full_model_tiling_ab.py', 'full_model_tiling_ab_bundle.zip'),
    ('gate_rows2', '34_full_model_rows2_ab.py', 'full_model_rows2_ab_bundle.zip'),
    ('down_rows2', '36_full_model_down_rows2_ab.py', 'full_model_down_rows2_ab_bundle.zip'),
    ('gate_rows4', '39_full_model_gate_up_rows4_ab.py', 'full_model_gate_up_rows4_ab_bundle.zip'),
    ('down_rows4', '41_full_model_down_rows4_ab.py', 'full_model_down_rows4_ab_bundle.zip'),
    ('frozen', '42_final_full_model_gap.py', 'final_full_model_gap_bundle.zip'),
    ('dynamic_build', '44_dynamic_full_model_comparison.py', 'dynamic_full_model_comparison_bundle.zip'),
    ('four_way', '46_four_way_full_model_comparison.py', 'four_way_full_model_comparison_bundle.zip'),
]

def read(archive, name='report.json'):
    with zipfile.ZipFile(E / archive) as z:
        return json.loads(z.read(name))

def verify():
    for p in E.glob('*.zip'):
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            if bad: raise RuntimeError(f'{p.name}: corrupt member {bad}')
    values = read('four_way_full_model_comparison_bundle.zip')['mean_of_prompt_medians_ms']
    assert abs(values['dynamic'] / values['tuned'] - 5.1908) < 0.002
    frozen = read('final_full_model_gap_bundle.zip')['by_length']
    assert abs(frozen['256']['iree_speedup'] - 3.18401) < 0.0001
    for step, _, archive in STEPS[2:7]:
        row = read(archive)['summary']['code_s256']['median_ms']
        assert row['A'] > row['B'], step
    print('Evidence archives readable; headline ratios and five A/B directions checked.')

def summarize():
    verify()
    print('\nFull model, mean of prompt medians (ms):')
    for s, row in read('final_full_model_gap_bundle.zip')['by_length'].items():
        print(' S'+s, ', '.join(f'{k}={v:.2f}' for k,v in row['mean_of_prompt_medians_ms'].items()))
    print('Four-way S256:', ', '.join(f'{k}={v:.2f}' for k,v in read('four_way_full_model_comparison_bundle.zip')['mean_of_prompt_medians_ms'].items()))
    print('Tuning A/B on code_s256 (independent sessions):')
    for name, _, archive in STEPS[2:7]:
        v = read(archive)['summary']['code_s256']['median_ms']
        print(f' {name}: {v["A"]:.2f} -> {v["B"]:.2f} ms')

def gpu(name):
    step = next((x for x in STEPS if x[0] == name), None)
    if not step: raise SystemExit('Unknown phase; see `list`.')
    print('Source:', ROOT / 'scripts' / step[1])
    if step[2]: print('Recorded evidence:', E / step[2])
    if name == 'base':
        print('A100 example: python scripts/27_full_model_aot.py --root /content/drive/MyDrive/moe_iree_olmoe --lengths 64,256 --logits all')
    else:
        print('Historical script: inspect its header and pinned RUN/root paths, restore the full source-run artifacts and checkpoint, then execute in the pinned environment.')
        print('See REPRODUCING.md for phase dependencies and hardware changes. No GPU job launched by this command.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['list','verify','summarize','gpu'])
    p.add_argument('phase', nargs='?')
    a=p.parse_args()
    if a.action=='list':
        for name, script, archive in STEPS: print(f'{name:14} {script}  {archive or "builds base source run"}')
    elif a.action=='verify': verify()
    elif a.action=='summarize': summarize()
    elif not a.phase: p.error('gpu requires a phase (see list)')
    else: gpu(a.phase)

if __name__=='__main__': main()
