"""
corpus_eda.py — Exploratory analysis of the Georgian MusicXML corpus.

Walks a folder of MusicXML files, parses each with music21, and reports:
  * how many parts (voices) each piece has  -> part-count distribution
  * note / rest counts and note-type (duration) distribution across the corpus
  * simple per-piece stats (parts, notes, rests)

Outputs a console summary plus three CSVs:
    part_count_distribution.csv   (n_parts, n_pieces)
    duration_distribution.csv     (note_type, count, percent)   -- e.g. quarter, eighth, half
    per_piece.csv                 (file, n_parts, n_notes, n_rests)

Usage:
    pip install music21
    python scripts/corpus_eda.py --xml-root "E:\\Thesis\\Xml-s\\xalxuri simRerebi (nawilobriv)"
    python scripts/corpus_eda.py --xml-root <folder> --out-dir S:\\omr\\eval_out\\eda --limit 200

Notes:
  * music21 parsing is slow (~0.1-2 s per file), so a few thousand files can take
    many minutes.  Use --limit for a quick sample first.
  * Files that fail to parse are counted and skipped, not fatal.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

try:
    import music21 as m21
except ImportError:
    print('ERROR: music21 is required.  Install with:  pip install music21',
          file=sys.stderr)
    raise SystemExit(2)


XML_EXTS = {'.xml', '.musicxml', '.mxl'}


def _note_type_label(el) -> str:
    """A readable duration label, e.g. 'quarter', 'eighth.' (dotted), 'complex'."""
    d = el.duration
    base = d.type  # 'whole','half','quarter','eighth','16th',... or 'complex'/'zero'
    if base in ('complex', 'zero', None, ''):
        return base or 'unknown'
    dots = '.' * int(getattr(d, 'dots', 0) or 0)
    return f'{base}{dots}'


def analyze_file(path: Path):
    """Return (n_parts, n_notes, n_rests, duration_counter) or None on parse failure."""
    try:
        score = m21.converter.parse(str(path))
    except Exception:
        return None

    try:
        n_parts = len(score.parts)
    except Exception:
        n_parts = 0

    n_notes = n_rests = 0
    dur_counter: Counter = Counter()

    for el in score.recurse().notesAndRests:
        if isinstance(el, m21.chord.Chord):
            # each chord member is a sounding note
            n_notes += len(el.pitches)
            dur_counter[_note_type_label(el)] += len(el.pitches)
        elif isinstance(el, m21.note.Note):
            n_notes += 1
            dur_counter[_note_type_label(el)] += 1
        elif isinstance(el, m21.note.Rest):
            n_rests += 1
            dur_counter[f'rest:{_note_type_label(el)}'] += 1

    return n_parts, n_notes, n_rests, dur_counter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog='corpus_eda',
        description='Part-count and duration distribution over a MusicXML corpus.')
    ap.add_argument('--xml-root', required=True,
                    help='Folder to search recursively for MusicXML files.')
    ap.add_argument('--out-dir', default='.',
                    help='Where to write the CSV reports (default: current dir).')
    ap.add_argument('--limit', type=int, default=0,
                    help='Process only the first N files (0 = all).')
    args = ap.parse_args(argv)

    root = Path(args.xml_root)
    if not root.is_dir():
        print(f'ERROR: not a folder: {root}', file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in XML_EXTS)
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f'No MusicXML files found under {root}', file=sys.stderr)
        return 1
    print(f'Found {len(files)} MusicXML file(s) under {root}')

    part_counts: Counter = Counter()        # n_parts -> n_pieces
    dur_total: Counter = Counter()
    per_piece = []
    n_ok = n_fail = 0
    tot_notes = tot_rests = 0

    for i, fp in enumerate(files, 1):
        res = analyze_file(fp)
        if res is None:
            n_fail += 1
        else:
            n_parts, n_notes, n_rests, dur_counter = res
            n_ok += 1
            part_counts[n_parts] += 1
            dur_total.update(dur_counter)
            tot_notes += n_notes
            tot_rests += n_rests
            per_piece.append((fp.name, n_parts, n_notes, n_rests))
        if i % 50 == 0 or i == len(files):
            print(f'  ...{i}/{len(files)} parsed  (ok={n_ok}, fail={n_fail})',
                  file=sys.stderr)

    # ── Console summary ──
    print('\n=== Corpus summary ===')
    print(f'Pieces parsed : {n_ok}   (failed: {n_fail})')
    print(f'Total notes   : {tot_notes}')
    print(f'Total rests   : {tot_rests}')
    if n_ok:
        print(f'Avg notes/piece: {tot_notes / n_ok:.1f}')

    print('\nPart-count distribution (n_parts: n_pieces):')
    for k in sorted(part_counts):
        pct = 100.0 * part_counts[k] / max(1, n_ok)
        print(f'  {k:>2d} parts : {part_counts[k]:>5d}  ({pct:4.1f}%)')

    print('\nTop note-type durations:')
    note_durs = {k: v for k, v in dur_total.items() if not k.startswith('rest:')}
    total_dur = sum(note_durs.values()) or 1
    for label, cnt in sorted(note_durs.items(), key=lambda x: -x[1])[:12]:
        print(f'  {label:<12s}: {cnt:>7d}  ({100.0*cnt/total_dur:4.1f}%)')

    # ── CSV outputs ──
    with (out_dir / 'part_count_distribution.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['n_parts', 'n_pieces'])
        for k in sorted(part_counts):
            w.writerow([k, part_counts[k]])

    with (out_dir / 'duration_distribution.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['label', 'count', 'percent_of_notes'])
        for label, cnt in sorted(dur_total.items(), key=lambda x: -x[1]):
            w.writerow([label, cnt, f'{100.0*cnt/total_dur:.2f}'])

    with (out_dir / 'per_piece.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['file', 'n_parts', 'n_notes', 'n_rests'])
        for row in per_piece:
            w.writerow(row)

    print(f'\nCSVs written to {out_dir.resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
