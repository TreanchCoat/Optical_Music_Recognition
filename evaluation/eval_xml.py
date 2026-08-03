"""
eval_xml.py — Compare pipeline-output MusicXML(s) against a ground-truth
MusicXML (e.g. Finale-exported) and report note-level evaluation metrics.

Design notes
------------
* MEASURELESS comparison.  The pipeline numbers measures only when it
  detects a mid-stave time signature, so its measure indexing will not
  agree with Finale's.  We therefore flatten each part to a multiset
  of notes/rests and compare those — measure boundaries are ignored on
  both sides.
* MANY-TO-ONE file pairing.  GT XMLs are usually one-XML-per-piece
  covering multiple pages, while the pipeline emits one XML per page.
  All page-XMLs of a piece are concatenated (per part, in filename
  order) before being compared against the single GT XML for that
  piece.

What it measures
----------------
Three strictness levels, computed simultaneously:

    pitch_dur   — note matches if BOTH MIDI pitch and quarter-note
                  duration are identical (strictest)
    pitch_only  — note matches if MIDI pitch matches, regardless of
                  duration (isolates pitch detection from rhythm)
    rhythm_only — note matches if duration matches, regardless of pitch
                  (isolates rhythm detection from pitch)

Each level reports precision / recall / F1.  Rests are included as
events with kind='rest' and their own duration; in pitch-only mode all
rests collapse to a single 'rest' bucket so rest duration doesn't
contaminate pitch metrics.

Matching strategy
-----------------
* Pred XMLs in a piece directory are sorted by filename and concatenated
  per part index (1st part of page1 + 1st part of page2 + ...).
* Parts are paired by index (1st pred part ↔ 1st GT part, etc.).
* Within a part, notes are matched as a multiset — a predicted note
  pairs with any unused matching GT note.  Chord notes are flattened
  (each chord member becomes its own event).

CLI modes
---------

Single piece — explicit pred files and GT file::

    python evaluation/eval_xml.py single \\
        --pred-dir output/piece_X_pages/ \\
        --gt       finale_exports/piece_X.xml

Single piece — single pred file (page == piece)::

    python evaluation/eval_xml.py single \\
        --pred  output/page.xml \\
        --gt    finale_exports/foo.xml

Batch — pred root contains one subfolder per piece, each holding all
page XMLs for that piece.  GT root is searched recursively for an XML
whose stem matches the pred subfolder name::

    python evaluation/eval_xml.py batch \\
        --pred-root output/batch/ \\
        --gt-root   finale_exports/ \\
        --out       eval_results.csv

Layout expected for batch mode::

    output/batch/
        piece_001/
            page1.xml
            page2.xml
            ...
        piece_002/
            page1.xml
            ...

    finale_exports/                    (can be flat or arbitrarily nested)
        piece_001.xml
        subdir/piece_002.xml

Four metrics, increasing strictness
-----------------------------------
The first three are MULTISET (bag-of-notes) — they ignore order.  The
fourth uses sequence alignment to catch order errors that the multiset
metrics miss::

    pitch_dur   multiset    pitch+duration tuples must match
    pitch_only  multiset    only the pitch must match
    rhythm_only multiset    only the duration must match
    sequence    NW-aligned  pitch+duration tuples must match AT THE
                            SAME POSITION in the per-part sequence

The sequence metric is the strictest in two ways: order matters, and
it inherits the strictness of pitch+duration matching.  A piece that
detected every note correctly but put them in the wrong order would
score F1≈1.0 on pitch_dur (multiset) but well under 1.0 on sequence.
The gap between the multiset and sequence numbers tells you how much
of the error is detection-level vs ordering-level — useful for
isolating which subsystem to improve.

Limitations (documented; not silently ignored)
----------------------------------------------
* Parts pair by index, not by instrument name.  If the pipeline's part
  ordering disagrees with the GT, all parts will mismatch — fix
  ``num_parts`` first via OMR_NUM_PARTS env var.
* Voice info is ignored at the note level (pipeline emits voice=1 only).
* Enharmonics match by MIDI number (C# == Db).  Standard for OMR eval.

Dependencies::

    pip install music21
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import music21 as m21
except ImportError as e:
    print('ERROR: music21 is required.  Install with:  pip install music21',
          file=sys.stderr)
    raise SystemExit(2) from e


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NoteEvent:
    """One playable item: note or rest, with MIDI pitch and quarter-note duration."""
    kind: str            # 'note' | 'rest'
    midi: Optional[int]  # MIDI pitch 0..127, or None for rests
    dur: float           # quarter-note duration, rounded to 4 decimal places


@dataclass
class MatchStats:
    """TP/FP/FN counters at one strictness level."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: 'MatchStats') -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalResult:
    """Per-piece evaluation result with multi-level stats."""
    piece_id: str
    gt_path: str
    pred_paths: List[str]
    pitch_dur:   MatchStats = field(default_factory=MatchStats)   # multiset, pitch+dur
    pitch_only:  MatchStats = field(default_factory=MatchStats)   # multiset, pitch
    rhythm_only: MatchStats = field(default_factory=MatchStats)   # multiset, rhythm
    sequence:    MatchStats = field(default_factory=MatchStats)   # NW-aligned, pitch+dur
    n_parts_pred: int = 0
    n_parts_gt:   int = 0
    n_events_pred: int = 0
    n_events_gt:   int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Parsing — flat per-part event extraction (measure-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_events_per_part(score: 'm21.stream.Score') -> Dict[int, List[NoteEvent]]:
    """
    Walk a parsed music21 score and return::

        { part_idx: [NoteEvent, NoteEvent, ...], ... }

    Measure boundaries are flattened — events are simply collected in
    musical (time) order across all measures of each part.  Chords are
    expanded (each chord member becomes its own NoteEvent).  Rests are
    kept as ``kind='rest'``.
    """
    by_part: Dict[int, List[NoteEvent]] = {}
    for part_idx, part in enumerate(score.parts):
        events: List[NoteEvent] = []
        # ``recurse().notesAndRests`` walks every note/rest in every
        # measure of the part, in time order.
        for el in part.recurse().notesAndRests:
            dur = round(float(el.duration.quarterLength), 4)
            if isinstance(el, m21.chord.Chord):
                for p in el.pitches:
                    events.append(NoteEvent('note', int(p.midi), dur))
            elif isinstance(el, m21.note.Note):
                events.append(NoteEvent('note', int(el.pitch.midi), dur))
            elif isinstance(el, m21.note.Rest):
                events.append(NoteEvent('rest', None, dur))
            # Other element types (e.g. Unpitched percussion) silently
            # skipped — uncommon in our pipeline output.
        by_part[part_idx] = events
    return by_part


def _parse_xml(path: Path) -> 'm21.stream.Score':
    """Parse a MusicXML file with music21.  Raises if the file is invalid."""
    try:
        return m21.converter.parse(str(path))
    except Exception as e:
        raise RuntimeError(f'Could not parse {path}: {e}') from e


def _concat_pred_events(pred_paths: List[Path]) -> Dict[int, List[NoteEvent]]:
    """
    Parse and concatenate notes from many pred page-XMLs into one per-part
    event list.  Pages are processed in sorted-filename order to keep the
    concatenation deterministic.
    """
    pred_by_part: Dict[int, List[NoteEvent]] = defaultdict(list)
    for pp in sorted(pred_paths):
        try:
            s = _parse_xml(pp)
        except Exception as e:
            print(f'  WARN: skipping {pp.name} ({e})', file=sys.stderr)
            continue
        for pi, evts in _extract_events_per_part(s).items():
            pred_by_part[pi].extend(evts)
    return dict(pred_by_part)


# ─────────────────────────────────────────────────────────────────────────────
# Multiset matching
# ─────────────────────────────────────────────────────────────────────────────

def _match_multiset(pred: List[NoteEvent], gt: List[NoteEvent],
                    key_fn) -> MatchStats:
    """Multiset F1 via Counter intersection."""
    pred_keys = Counter(key_fn(e) for e in pred)
    gt_keys   = Counter(key_fn(e) for e in gt)
    inter = pred_keys & gt_keys
    tp = sum(inter.values())
    fp = sum(pred_keys.values()) - tp
    fn = sum(gt_keys.values()) - tp
    return MatchStats(tp=tp, fp=fp, fn=fn)


def _key_pitch_dur(e: NoteEvent):
    return (e.kind, e.midi, e.dur)


def _key_pitch_only(e: NoteEvent):
    # Rests collapse to a single bucket so rest duration doesn't affect
    # the pitch metric.
    return ('rest', None) if e.kind == 'rest' else ('note', e.midi)


def _key_rhythm_only(e: NoteEvent):
    return (e.kind, e.dur)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence alignment (Needleman-Wunsch) — order-aware matching
# ─────────────────────────────────────────────────────────────────────────────

# Scoring scheme: standard +1 for a match, -1 for a mismatch, -1 per gap.
# These three values together encode "matches are good, everything else
# is bad and roughly equally bad" — the simplest defensible choice.
# Tuning the relative magnitudes would let you say things like "a
# substitution is worse than a gap" but the literature doesn't have a
# clear consensus on what those weights should be for OMR, so we stick
# to symmetric.
_NW_MATCH    =  1
_NW_MISMATCH = -1
_NW_GAP      = -1


def _align_sequence(pred: List[NoteEvent],
                    gt: List[NoteEvent],
                    key_fn) -> MatchStats:
    """
    Needleman-Wunsch alignment of two NoteEvent sequences by ``key_fn``.

    Walks the optimal alignment and counts::

        TP = positions where pred[i] and gt[j] are aligned AND key matches
        FP = positions where pred has a real event that didn't recover any
             matching GT event (substitutions + insertions in pred)
        FN = positions where GT has a real event that pred missed
             (substitutions + deletions from pred)

    Returns a MatchStats; precision/recall/F1 come from the usual formulas.

    Complexity: O(len(pred) × len(gt)) time and memory.  For per-part
    sequences of a few thousand events, this is fast.  No need for
    banded / sparse variants at this scale.
    """
    n, m = len(pred), len(gt)
    if n == 0 and m == 0:
        return MatchStats(tp=0, fp=0, fn=0)
    if n == 0:
        return MatchStats(tp=0, fp=0, fn=m)
    if m == 0:
        return MatchStats(tp=0, fp=n, fn=0)

    # DP[i][j] = best alignment score for pred[:i] vs gt[:j].
    DP = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        DP[i][0] = i * _NW_GAP
    for j in range(1, m + 1):
        DP[0][j] = j * _NW_GAP

    pred_keys = [key_fn(e) for e in pred]
    gt_keys   = [key_fn(e) for e in gt]

    for i in range(1, n + 1):
        row_im1 = DP[i - 1]
        row_i   = DP[i]
        pk = pred_keys[i - 1]
        for j in range(1, m + 1):
            diag = row_im1[j - 1] + (_NW_MATCH if pk == gt_keys[j - 1]
                                     else _NW_MISMATCH)
            up   = row_im1[j] + _NW_GAP
            left = row_i[j - 1] + _NW_GAP
            row_i[j] = diag if diag >= up and diag >= left else (
                up if up >= left else left)

    # Traceback to count TP/FP/FN.
    tp = fp = fn = 0
    i, j = n, m
    while i > 0 and j > 0:
        diag_score = (_NW_MATCH if pred_keys[i - 1] == gt_keys[j - 1]
                      else _NW_MISMATCH)
        if DP[i][j] == DP[i - 1][j - 1] + diag_score:
            if diag_score == _NW_MATCH:
                tp += 1
            else:
                # Substitution: both sides "lose" a real event.
                fp += 1
                fn += 1
            i -= 1
            j -= 1
        elif DP[i][j] == DP[i - 1][j] + _NW_GAP:
            fp += 1     # pred has an extra event with no GT counterpart
            i -= 1
        else:
            fn += 1     # gt has a real event with no pred counterpart
            j -= 1
    while i > 0:
        fp += 1
        i -= 1
    while j > 0:
        fn += 1
        j -= 1
    return MatchStats(tp=tp, fp=fp, fn=fn)


# ─────────────────────────────────────────────────────────────────────────────
# Per-piece evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_piece(piece_id: str,
                   pred_paths: List[Path],
                   gt_path: Path,
                   verbose: bool = False,
                   aggregate_parts: bool = False) -> EvalResult:
    """
    Compare concatenated pred pages against one GT XML for a piece.

    aggregate_parts
        When True, ignore part assignment entirely — flatten every part
        into a single multiset on both sides before matching.  Use this
        when the pipeline's part count doesn't match the GT's (e.g.
        the pipeline collapses three-voice music into one part, so a
        per-part comparison systematically over-counts FP/FN even when
        the actual notes are correct).
    """
    pred_by_part = _concat_pred_events(pred_paths)
    gt_score = _parse_xml(gt_path)
    gt_by_part = _extract_events_per_part(gt_score)

    result = EvalResult(
        piece_id=piece_id,
        gt_path=str(gt_path),
        pred_paths=[str(p) for p in pred_paths],
    )
    result.n_parts_pred = len(pred_by_part)
    result.n_parts_gt   = len(gt_by_part)
    result.n_events_pred = sum(len(v) for v in pred_by_part.values())
    result.n_events_gt   = sum(len(v) for v in gt_by_part.values())

    if aggregate_parts:
        # Flatten everything into a single multiset per side.  Part
        # assignment is ignored — this measures pure note detection.
        pred_flat: List[NoteEvent] = []
        for evts in pred_by_part.values():
            pred_flat.extend(evts)
        gt_flat: List[NoteEvent] = []
        for evts in gt_by_part.values():
            gt_flat.extend(evts)
        result.pitch_dur.add(  _match_multiset(pred_flat, gt_flat, _key_pitch_dur))
        result.pitch_only.add( _match_multiset(pred_flat, gt_flat, _key_pitch_only))
        result.rhythm_only.add(_match_multiset(pred_flat, gt_flat, _key_rhythm_only))
        # Sequence metric over the flattened streams.  Less meaningful
        # than the per-part version since aggregation interleaves voices
        # arbitrarily, but still surfaces gross ordering differences.
        result.sequence.add(   _align_sequence(pred_flat, gt_flat, _key_pitch_dur))
        if verbose:
            _print_piece_summary(piece_id, pred_paths, result,
                                 mode='aggregated parts')
        return result

    # Pair parts by index.  Iterate the union so missing-on-one-side
    # parts count as full FP or FN.
    all_part_indices = set(pred_by_part) | set(gt_by_part)
    for pi in sorted(all_part_indices):
        p_notes = pred_by_part.get(pi, [])
        g_notes = gt_by_part.get(pi, [])
        result.pitch_dur.add(  _match_multiset(p_notes, g_notes, _key_pitch_dur))
        result.pitch_only.add( _match_multiset(p_notes, g_notes, _key_pitch_only))
        result.rhythm_only.add(_match_multiset(p_notes, g_notes, _key_rhythm_only))
        # Sequence alignment within each part — captures order errors
        # the multiset metrics miss.  Per-part so the metric measures
        # ordering WITHIN a voice, not across voices.
        result.sequence.add(   _align_sequence(p_notes, g_notes, _key_pitch_dur))

    if verbose:
        _print_piece_summary(piece_id, pred_paths, result, mode='per-part')

    return result


def _print_piece_summary(piece_id: str, pred_paths: List[Path],
                         result: EvalResult, mode: str) -> None:
    print(f'  {piece_id}: pred {len(pred_paths)} page(s), '
          f'parts {result.n_parts_pred}/{result.n_parts_gt}, '
          f'events {result.n_events_pred}/{result.n_events_gt}  '
          f'[{mode}]')
    for label, st in (('pitch+dur', result.pitch_dur),
                      ('pitch',     result.pitch_only),
                      ('rhythm',    result.rhythm_only),
                      ('sequence',  result.sequence)):
        print(f'    {label:<10s} TP={st.tp:>5d} FP={st.fp:>5d} '
              f'FN={st.fn:>5d}  P={st.precision():.3f} '
              f'R={st.recall():.3f}  F1={st.f1():.3f}')


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(label: str, st: MatchStats) -> None:
    print(f'  {label:<14s} TP={st.tp:>7d}  FP={st.fp:>7d}  FN={st.fn:>7d}    '
          f'P={st.precision():.3f}  R={st.recall():.3f}  F1={st.f1():.3f}')


def report_corpus(results: List[EvalResult]) -> None:
    if not results:
        print('No results.')
        return
    print(f'\n=== Corpus summary over {len(results)} piece(s) ===')
    total_pd = MatchStats()
    total_p  = MatchStats()
    total_r  = MatchStats()
    total_s  = MatchStats()
    for r in results:
        total_pd.add(r.pitch_dur)
        total_p.add(r.pitch_only)
        total_r.add(r.rhythm_only)
        total_s.add(r.sequence)
    print('Aggregate (sum of TP/FP/FN across all pieces):')
    _print_summary('pitch + dur',  total_pd)
    _print_summary('pitch only',   total_p)
    _print_summary('rhythm only',  total_r)
    _print_summary('sequence',     total_s)


def write_csv(results: List[EvalResult], out_path: Path) -> None:
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'piece_id', 'gt_path', 'n_pred_files',
            'n_parts_pred', 'n_parts_gt',
            'n_events_pred', 'n_events_gt',
            'pd_tp', 'pd_fp', 'pd_fn', 'pd_P', 'pd_R', 'pd_F1',
            'p_tp', 'p_fp', 'p_fn', 'p_P', 'p_R', 'p_F1',
            'r_tp', 'r_fp', 'r_fn', 'r_P', 'r_R', 'r_F1',
            's_tp', 's_fp', 's_fn', 's_P', 's_R', 's_F1',
        ])
        for r in results:
            w.writerow([
                r.piece_id, r.gt_path, len(r.pred_paths),
                r.n_parts_pred, r.n_parts_gt,
                r.n_events_pred, r.n_events_gt,
                r.pitch_dur.tp, r.pitch_dur.fp, r.pitch_dur.fn,
                f'{r.pitch_dur.precision():.4f}', f'{r.pitch_dur.recall():.4f}',
                f'{r.pitch_dur.f1():.4f}',
                r.pitch_only.tp, r.pitch_only.fp, r.pitch_only.fn,
                f'{r.pitch_only.precision():.4f}', f'{r.pitch_only.recall():.4f}',
                f'{r.pitch_only.f1():.4f}',
                r.rhythm_only.tp, r.rhythm_only.fp, r.rhythm_only.fn,
                f'{r.rhythm_only.precision():.4f}', f'{r.rhythm_only.recall():.4f}',
                f'{r.rhythm_only.f1():.4f}',
                r.sequence.tp, r.sequence.fp, r.sequence.fn,
                f'{r.sequence.precision():.4f}', f'{r.sequence.recall():.4f}',
                f'{r.sequence.f1():.4f}',
            ])
    print(f'Per-piece results -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Batch pairing
# ─────────────────────────────────────────────────────────────────────────────

def _find_gt_for_piece(piece_name: str, gt_root: Path) -> Optional[Path]:
    """
    Locate the GT XML whose stem matches ``piece_name``.

    Searches:
      1. ``gt_root / f'{piece_name}.xml'``         (flat layout)
      2. ``gt_root / '**/' / f'{piece_name}.xml'`` (nested)

    Returns the first match.  Returns None if no match.  Prints a
    warning if multiple candidates exist (caller should pick a more
    specific layout or rename to disambiguate).
    """
    flat = gt_root / f'{piece_name}.xml'
    if flat.is_file():
        return flat
    candidates = list(gt_root.rglob(f'{piece_name}.xml'))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f'  WARN: multiple GT XMLs match piece "{piece_name}":',
              file=sys.stderr)
        for c in candidates:
            print(f'         {c}', file=sys.stderr)
        print(f'         using first ({candidates[0]})', file=sys.stderr)
    return candidates[0]


def _discover_pieces(pred_root: Path,
                     gt_root: Path) -> List[Tuple[str, List[Path], Path]]:
    """
    For each immediate subfolder of ``pred_root``, treat the folder as
    one piece.  Collect all *.xml files inside that folder (recursive)
    as the piece's predicted pages.  Look up the matching GT XML in
    ``gt_root`` by folder name.

    Returns a list of (piece_id, [pred_paths], gt_path).
    """
    pieces: List[Tuple[str, List[Path], Path]] = []
    n_skipped = 0
    for sub in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        pred_paths = sorted(sub.rglob('*.xml'))
        if not pred_paths:
            continue
        piece_name = sub.name
        gt = _find_gt_for_piece(piece_name, gt_root)
        if gt is None:
            print(f'  WARN: no GT XML found for piece "{piece_name}" '
                  f'(searched {gt_root} for {piece_name}.xml)',
                  file=sys.stderr)
            n_skipped += 1
            continue
        pieces.append((piece_name, pred_paths, gt))
    if n_skipped:
        print(f'  Skipped {n_skipped} piece(s) with no matching GT.',
              file=sys.stderr)
    return pieces


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_single(args) -> int:
    """Single-piece mode: --pred-dir OR --pred, plus --gt."""
    if not args.gt:
        print('--gt is required in single mode', file=sys.stderr)
        return 2
    gt_path = Path(args.gt).resolve()
    if not gt_path.is_file():
        print(f'Not a file: {gt_path}', file=sys.stderr); return 2

    if args.pred_dir:
        pred_dir = Path(args.pred_dir).resolve()
        if not pred_dir.is_dir():
            print(f'Not a dir: {pred_dir}', file=sys.stderr); return 2
        pred_paths = sorted(pred_dir.rglob('*.xml'))
        if not pred_paths:
            print(f'No .xml files in {pred_dir}', file=sys.stderr); return 1
        piece_id = pred_dir.name
    elif args.pred:
        p = Path(args.pred).resolve()
        if not p.is_file():
            print(f'Not a file: {p}', file=sys.stderr); return 2
        pred_paths = [p]
        piece_id = p.stem
    else:
        print('Specify --pred-dir or --pred', file=sys.stderr); return 2

    result = evaluate_piece(piece_id, pred_paths, gt_path, verbose=True,
                            aggregate_parts=args.aggregate_parts)
    report_corpus([result])
    if args.out:
        write_csv([result], Path(args.out).resolve())
    return 0


def _cmd_batch(args) -> int:
    """Batch mode: pred-root has one subdir per piece, gt-root has matching XMLs."""
    pred_root = Path(args.pred_root).resolve()
    gt_root   = Path(args.gt_root).resolve()
    if not pred_root.is_dir():
        print(f'Not a dir: {pred_root}', file=sys.stderr); return 2
    if not gt_root.is_dir():
        print(f'Not a dir: {gt_root}', file=sys.stderr); return 2

    pieces = _discover_pieces(pred_root, gt_root)
    if not pieces:
        print('No (pred, gt) piece pairs found.', file=sys.stderr); return 1
    print(f'Evaluating {len(pieces)} piece(s)...')
    results: List[EvalResult] = []
    for i, (piece_id, preds, gt) in enumerate(pieces, 1):
        try:
            r = evaluate_piece(piece_id, preds, gt, verbose=args.verbose,
                               aggregate_parts=args.aggregate_parts)
            results.append(r)
        except Exception as e:
            print(f'  [{i}/{len(pieces)}] {piece_id}: skipped ({e})',
                  file=sys.stderr)
        if i % 100 == 0:
            print(f'  ...{i}/{len(pieces)} processed', file=sys.stderr)
    report_corpus(results)
    if args.out:
        write_csv(results, Path(args.out).resolve())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='eval_xml',
        description='Note-level F1 comparison of OMR-output XML vs ground-truth XML.',
    )
    sub = ap.add_subparsers(dest='cmd', required=True)

    sp_single = sub.add_parser('single',
                               help='Evaluate a single piece (one GT XML).')
    sp_single_group = sp_single.add_mutually_exclusive_group(required=True)
    sp_single_group.add_argument('--pred-dir',
                                 help='Folder of predicted page XMLs for one piece.')
    sp_single_group.add_argument('--pred',
                                 help='Single predicted XML file (page == piece).')
    sp_single.add_argument('--gt', required=True,
                           help='Ground-truth XML for the piece.')
    sp_single.add_argument('--out', default=None,
                           help='Optional CSV path for results.')
    sp_single.add_argument('--aggregate-parts', action='store_true',
                           help='Flatten all parts on both sides before '
                                'matching.  Use when the pipeline collapses '
                                'multi-voice music into one part — gives a '
                                'pure note-detection F1 without penalising '
                                'incorrect part assignment.')

    sp_batch = sub.add_parser('batch',
                              help='Evaluate a corpus (one subdir per piece).')
    sp_batch.add_argument('--pred-root', required=True,
                          help='Root containing one subdir per piece.')
    sp_batch.add_argument('--gt-root', required=True,
                          help='Root containing GT XMLs (flat or nested).')
    sp_batch.add_argument('--out', default=None,
                          help='Optional CSV path for per-piece results.')
    sp_batch.add_argument('--aggregate-parts', action='store_true',
                          help='Flatten all parts on both sides before '
                               'matching.  Use when the pipeline collapses '
                               'multi-voice music into one part — gives a '
                               'pure note-detection F1 without penalising '
                               'incorrect part assignment.')
    sp_batch.add_argument('--verbose', '-v', action='store_true',
                          help='Print per-piece stats.')

    args = ap.parse_args(argv)
    if args.cmd == 'single':
        return _cmd_single(args)
    return _cmd_batch(args)


if __name__ == '__main__':
    sys.exit(main())
