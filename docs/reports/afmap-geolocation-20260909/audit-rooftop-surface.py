"""Summarize frozen local surface evidence; never infer a roof height or write events."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def audit(root):
    inventory = json.loads((root / 'expanded/inventory.json').read_text())
    cloud = np.genfromtxt(root / 'cloud.csv', delimiter=',', names=True)
    stats = {}
    for radius in (5, 10, 25, 50):
        selected = cloud[(abs(cloud['X'] - 726607) < radius)
                         & (abs(cloud['Y'] - 3854922) < radius)]
        stats[str(radius * 2) + 'm_square'] = {
            'count': len(selected),
            'mixed_surface_z_quantiles': np.percentile(
                selected['Z'], [0, 10, 50, 90, 100]).tolist() if len(selected) else [],
        }
    coverage = {}
    for path in sorted(root.glob('*.png')):
        if len(path.stem) != 36:
            continue
        with Image.open(path) as image:
            alpha = np.asarray(image.getchannel('A'))
            h, w = alpha.shape
            coverage[path.stem] = {
                'valid_fraction_240m_window': float(np.mean(alpha > 0)),
                'valid_fraction_central_50m_window': float(np.mean(
                    alpha[int(h*(.5-25/240)):int(h*(.5+25/240)),
                          int(w*(.5-25/240)):int(w*(.5+25/240))] > 0)),
            }
    paths = [root / 'cloud.csv', root / 'flight.log', root / 'event.jpg',
             root / 'orthophoto-crop.tif', root / 'source-metadata.jsonl',
             root / 'source-photos/metadata.json', root / 'expanded/inventory.json',
             root / 'vertical-log-excerpts.json', root / 'matching.json']
    return {
        'scope': 'frozen local evidence; no service or database mutation',
        'expanded_records': len(inventory),
        'read_errors': [x for x in inventory if 'error' in x],
        'nonzero_preview_coverage_records': sum(x.get('valid_fraction', 0) > 0 for x in inventory),
        'preview_coverage_above_one_percent': sum(x.get('valid_fraction', 0) > .01 for x in inventory),
        'point_count_240m_window': len(cloud),
        'sample_center_utm49n': [726607, 3854922],
        'sampling_note': 'Rounded previous projected point; not a verified target location. All Z samples mix surfaces.',
        'mixed_surface_samples': stats,
        'alpha_coverage': coverage,
        'decision': {
            'same_roof_confirmed': False,
            'relative_roof_height_m': None,
            'vertical_datum_end_to_end_verified': False,
            'height_recalculation_performed': False,
            'accuracy_verified': False,
            'reason': 'No verified same-roof polygon and adjacent-ground pair; no accepted roof surface.',
        },
        'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_directory)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in (
        'expanded_records', 'nonzero_preview_coverage_records',
        'preview_coverage_above_one_percent', 'point_count_240m_window', 'decision')}, ensure_ascii=False))
