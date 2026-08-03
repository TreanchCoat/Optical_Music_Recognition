"""
build_musicxml_v3.py — Voice-split variant of build_musicxml_v2.

What's different from v2
------------------------
In v2, simultaneous noteheads (within ±CHORD_X_TOLERANCE on the x-axis)
are treated as a CHORD — emitted as a single <voice> with <chord/> tags
on the second-and-subsequent notes.

In v3, those same simultaneous noteheads are treated as INDEPENDENT
VOICES on the same staff — emitted as N separate <note> elements with
<voice>1</voice>, <voice>2</voice>, …, and <backup> elements between
them so each voice's time cursor stays synchronised at the cluster's
onset.

Everything else is identical: clef detection, key signature, time
signature parsing, rest emission, fermata / ornament attachment, slur
assignment, repeat barlines, coordinate embedding.  Only the
chord-emission block inside the per-measure builder changes.

When to use which builder
-------------------------
    v2  — instrumental music, piano / guitar, jazz lead sheets — any
          score where simultaneous notes are actual chords meant to be
          played together by one performer.
    v3  — vocal polyphony, especially Georgian three-voice folk songs,
          hymns / chorales notated on a single staff per choir, or any
          score where simultaneous notes represent INDEPENDENT MELODIC
          LINES sung by different voices.

How to swap into the pipeline
-----------------------------
In ``scripts/pipeline.py`` change::

    import build_musicxml_v2

to::

    import build_musicxml_v3 as build_musicxml_v2

The function names match (``build_score_xml_v3`` mirrors
``build_score_xml_v2``), so a simple alias swap is enough.  Or update
the direct calls to ``build_musicxml_v2.build_score_xml_v2`` to use
``build_musicxml_v3.build_score_xml_v3`` explicitly.

Dependency note
---------------
This module re-uses v2's helpers (group_chords, calculate_pitch,
determine_duration, all the constants and detection functions) — it
does NOT copy them.  Keep ``build_musicxml_v2.py`` alongside this file.
If you ever delete v2 in favour of v3, this module would need its
imports inlined (about 50 lines of constants + helpers).

Voice-split details
-------------------
Within an x-cluster of N noteheads sorted top→bottom (lowest cy first):

    voice 1 = topmost
    voice 2 = middle
    voice 3 = bottom
    voice N = bottommost

For each cluster the emission pattern is::

    <note voice=1>...</note>      ← topmost
    <backup duration=D/>          ← rewind time cursor to cluster onset
    <note voice=2>...</note>      ← middle
    <backup duration=D/>          ← rewind again
    <note voice=3>...</note>      ← bottom
    [no trailing backup — next cluster picks up here in voice 3's timeline]

For HOMOPHONIC textures (all voices share the same rhythm — typical of
Georgian three-voice singing) this works cleanly because every voice's
duration matches and the backups balance out.  When durations differ
within a cluster (rare in this repertoire), the time math may drift
slightly — see the note in ``_emit_chord_as_voices`` below.

Cluster sizing assumption
-------------------------
The voice split assumes that cluster sizes are roughly consistent
through the piece — i.e. if the music is three-voice, every cluster
should have three noteheads.  When a cluster is missing one or more
voices (likely a detection error), the corresponding voice's time
cursor advances out of step with the others.  The MusicXML output will
still render but the missing voice will appear to "skip" through that
cluster.  In practice MuseScore handles this by displaying the gap as
silence in that voice.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from lxml import etree as _etree

# XMLBuilder + low-level layout helper come from the xml_builder module,
# same place v2 gets them.
from xml_builder import XMLBuilder, pixels_to_tenths

# Re-use every helper and constant from v2 verbatim.  Only the chord-
# emission block changes; everything else (clef, key, time-sig, rest,
# fermata, slur, repeat-bar, coordinate embed) is identical.
from build_musicxml_v2 import (
    # Public constants
    CHORD_X_TOLERANCE,
    # Symbolic dicts used during emission
    CLEF_TO_SIGN_LINE,
    ACCIDENTAL_DETECT,
    REST_CLASSES,
    FERMATA_ABOVE_CLASSES,
    FERMATA_BELOW_CLASSES,
    ORNAMENT_DETECT,
    # Detection helpers
    find_main_clef,
    has_octave_marker,
    detect_key_signature,
    detect_time_signature,
    detect_all_time_signatures,
    detect_repeats,
    find_accidental_at,
    find_flag_at,
    find_dots_at,
    AUGMENTATION_DOT_CLASS,
    # Grouping + assignment helpers
    group_chords,
    assign_slurs,
    # Pitch / duration computation
    calculate_pitch,
    determine_duration,
    # Top-level helpers
    auto_divisions,
    assign_detection_ids,
    build_coordinate_records,
    apply_melisma_overrides,
    # Types
    MelismaOverrides,
)


# ─────────────────────────────────────────────────────────────────────────────
# Backup element helper
# ─────────────────────────────────────────────────────────────────────────────

def _add_backup(measure, duration: int):
    """
    Insert a ``<backup>`` element that rewinds the measure's time
    cursor by ``duration`` divisions.  Used between voice emissions
    in the same x-cluster so each voice starts at the cluster's onset.
    """
    backup = _etree.SubElement(measure, 'backup')
    dur_el = _etree.SubElement(backup, 'duration')
    dur_el.text = str(int(max(1, duration)))
    return backup


# ─────────────────────────────────────────────────────────────────────────────
# Core entry point (mirrors build_score_xml_v2)
# ─────────────────────────────────────────────────────────────────────────────

def build_score_xml_v3(processed_score,
                       page_detections,
                       output_path: str,
                       instrument_name: str = 'Voice',
                       midi_program: int = 53,   # GM 53 = "Voice Oohs"
                       divisions: Optional[int] = None,
                       add_bracket: bool = True,
                       include_layout: bool = True,
                       embed_coordinates: bool = True,
                       chord_tolerance: int = CHORD_X_TOLERANCE,
                       melisma_overrides: Optional[MelismaOverrides] = None,
                       id_map: Optional[Dict[int, str]] = None,
                       ) -> XMLBuilder:
    """
    Voice-split MusicXML assembly.  Same call signature as
    ``build_musicxml_v2.build_score_xml_v2``; differs only in that
    same-onset noteheads are split into independent voices rather than
    grouped as a chord.

    Defaults that differ from v2:
      * ``instrument_name='Voice'`` and ``midi_program=53`` (GM "Voice
        Oohs") rather than 'Clarinet' / 72.  Override either if you're
        using v3 for a mixed-instrument vocal-style score.
    """
    builder = XMLBuilder()
    builder.set_defaults()

    if divisions is None:
        divisions = auto_divisions(page_detections)

    if id_map is None:
        id_map = assign_detection_ids(page_detections)
    id_to_note: Dict[str, object] = {}

    num_parts = processed_score.num_parts

    # ── Part list ──
    if add_bracket and num_parts > 1:
        builder.add_part_group(number=1, group_type='start', symbol='bracket')

    for i in range(num_parts):
        part_id = f'P{i+1}'
        name = f'{instrument_name} {i+1}' if num_parts > 1 else instrument_name
        builder.add_part(part_id,
                         instrument_name=name,
                         midi_channel=i + 1,
                         midi_program=midi_program)

    if add_bracket and num_parts > 1:
        builder.add_part_group(number=1, group_type='stop')

    # Pre-scan: find a page-level time signature so later staves
    # without a detected time signature still get correct measure-rest
    # durations.  Same logic as v2.
    page_time_sig: Optional[Tuple[int, int]] = None
    for sd in page_detections.all_staves:
        ts = detect_time_signature(sd.detections, after_x=0)
        if ts is not None:
            page_time_sig = ts
            break

    # ── Walk each part / staff ──
    for part_idx in range(num_parts):
        part_id = f'P{part_idx + 1}'
        staves = processed_score.parts[part_idx]
        staff_dets_list = page_detections.parts[part_idx]

        for staff_num, (pstaff, sdet) in enumerate(zip(staves,
                                                       staff_dets_list)):
            measure_num = staff_num + 1
            _build_measure_v3(
                builder, part_id, measure_num,
                pstaff, sdet.detections,
                divisions=divisions,
                chord_tolerance=chord_tolerance,
                is_first_measure=(staff_num == 0),
                include_layout=include_layout,
                id_map=id_map,
                id_to_note=id_to_note,
                page_time_sig=page_time_sig,
            )

    # ── Apply melisma overrides ──
    apply_melisma_overrides(builder, id_to_note, melisma_overrides)

    # ── Embed coordinate metadata ──
    if embed_coordinates:
        # Tag with v3 so a later reader can tell which builder produced
        # the file (matters for the voice-vs-chord interpretation when
        # editing).
        builder.add_miscellaneous_field('omr-version', '3.0')
        builder.add_miscellaneous_field(
            'omr-source-image',
            os.path.basename(page_detections.image_path),
        )
        builder.add_miscellaneous_field('omr-image-width',
                                        str(page_detections.img_w))
        builder.add_miscellaneous_field('omr-image-height',
                                        str(page_detections.img_h))
        builder.add_miscellaneous_field('omr-divisions', str(divisions))
        builder.add_miscellaneous_field('omr-chord-policy', 'voices')

        coords = build_coordinate_records(page_detections, id_map)
        builder.add_miscellaneous_field(
            'omr-coordinates',
            json.dumps(coords, separators=(',', ':')),
        )

    builder.save(output_path)
    return builder


# ─────────────────────────────────────────────────────────────────────────────
# Measure assembly — voice-split variant
# ─────────────────────────────────────────────────────────────────────────────

def _build_measure_v3(builder: XMLBuilder, part_id: str, measure_num: int,
                      pstaff, dets, *,
                      divisions: int,
                      chord_tolerance: int,
                      is_first_measure: bool,
                      include_layout: bool,
                      id_map: Dict[int, str],
                      id_to_note: Dict[str, object],
                      page_time_sig: Optional[Tuple[int, int]] = None):
    """
    Build one measure.  Identical to v2's ``_build_measure_v2`` EXCEPT
    that x-clusters of noteheads are emitted as separate voices rather
    than chord-grouped.

    The diff against v2 is concentrated in the section labelled
    "VOICE SPLIT" below.  Everything before that point (clef, key
    signature, time signature, rest handling, time-signature changes)
    is verbatim from v2.
    """
    # ── Clef ──
    clef_det = find_main_clef(dets)
    clef_type = clef_det.class_name if clef_det else 'clefG'
    clef_sign, clef_line = CLEF_TO_SIGN_LINE.get(clef_type, ('G', 2))
    clef_octave_change = -1 if has_octave_marker(dets, clef_det) else None

    # ── Key signature ──
    after_x = clef_det.cx if clef_det else 0
    key_fifths, key_mode = detect_key_signature(dets, after_x)

    # ── Time signature(s) ──
    all_sigs = detect_all_time_signatures(dets, after_x=after_x)
    local_sig = all_sigs[0][1] if all_sigs else None
    time_sig = local_sig or page_time_sig
    extra_time_sigs = all_sigs[1:]
    write_time = is_first_measure or (
        local_sig is not None and local_sig != page_time_sig
    )
    time_beats = time_sig[0] if (time_sig and write_time) else None
    time_beat_type = time_sig[1] if (time_sig and write_time) else None

    # ── Repeat barlines ──
    forward_repeat, backward_repeat = detect_repeats(
        dets,
        staff_left_x  = pstaff.left_x,
        staff_right_x = pstaff.right_x,
    )

    # ── Create the measure element ──
    if is_first_measure:
        measure = builder.add_measure(
            part_id, measure_num,
            divisions=divisions,
            key_fifths=key_fifths, key_mode=key_mode,
            time_beats=time_beats, time_beat_type=time_beat_type,
            clef_sign=clef_sign, clef_line=clef_line,
            clef_octave_change=clef_octave_change,
        )
    else:
        measure = builder.add_measure(part_id, measure_num)
        builder.add_print(measure, new_system=True)

    if forward_repeat:
        builder.add_repeat(measure, direction='forward')

    # ── Pull symbol groups (identical to v2) ──
    noteheads = sorted([d for d in dets if d.class_name.startswith('notehead')],
                       key=lambda d: d.cx)
    rests = sorted([d for d in dets if d.class_name in REST_CLASSES],
                   key=lambda d: d.cx)
    accidentals = [d for d in dets if d.class_name in ACCIDENTAL_DETECT]
    slur_dets = [d for d in dets if d.class_name == 'slur']
    fermatas = [d for d in dets
                if d.class_name in FERMATA_ABOVE_CLASSES
                or d.class_name in FERMATA_BELOW_CLASSES]
    ornaments = [d for d in dets if d.class_name in ORNAMENT_DETECT]

    # If nothing detected → whole-measure rest.
    if not noteheads and not rests:
        builder.add_rest(measure,
                         duration=divisions * 4,
                         measure_rest=True,
                         voice=1)
        if backward_repeat:
            builder.add_repeat(measure, direction='backward')
        return

    # ── Group noteheads into x-clusters (same as v2's chord_groups) ──
    # Reuse v2's grouping function — we only differ in HOW we emit each
    # cluster, not in which noteheads belong together.
    chord_groups = group_chords(noteheads, x_tolerance=chord_tolerance)
    slur_starts, slur_stops = assign_slurs(chord_groups, slur_dets)

    # ── Unified event timeline (identical to v2) ──
    events = []
    for i, g in enumerate(chord_groups):
        anchor = min(g, key=lambda d: d.cx)
        events.append(('chord', g, anchor.cx))
    for r in rests:
        events.append(('rest', r, r.cx))
    for cx, sig in extra_time_sigs:
        events.append(('timesig', sig, cx))
    events.sort(key=lambda e: e[2])

    # ── Layout reference (first note's full_cx for default_x offsets) ──
    first_x = None
    if include_layout and events:
        if events[0][0] == 'chord':
            first_x = min(n.full_cx for n in events[0][1])
        elif events[0][0] == 'rest':
            first_x = events[0][1].full_cx
        else:
            for ev in events:
                if ev[0] in ('chord', 'rest'):
                    first_x = (min(n.full_cx for n in ev[1])
                               if ev[0] == 'chord' else ev[1].full_cx)
                    break

    # Top-line y for layout's default_y reference.
    top_line_y = pstaff.line_positions[0] if pstaff.line_positions else 0

    note_anchors = []        # (note_el, anchor_cx, anchor_cy) for ornament attach
    chord_idx_counter = 0

    for kind, payload, _evt_cx in events:
        if kind == 'timesig':
            # Identical to v2: emit an inline <attributes><time> so the
            # recalculator can split into a fresh measure here.
            beats, beat_type = payload
            attrs = _etree.SubElement(measure, 'attributes')
            t_el = _etree.SubElement(attrs, 'time')
            _etree.SubElement(t_el, 'beats').text = str(beats)
            _etree.SubElement(t_el, 'beat-type').text = str(beat_type)
            continue

        if kind == 'rest':
            rest_det = payload
            xml_type, q_fraction = REST_CLASSES[rest_det.class_name]
            measure_rest_flag = (xml_type == 'whole')
            rest_dots = 0
            if not measure_rest_flag:
                rest_dots = find_dots_at(
                    rest_det,
                    [d for d in dets if d.class_name == AUGMENTATION_DOT_CLASS])
                q_fraction = q_fraction * (1.5 if rest_dots == 1
                                           else 1.75 if rest_dots >= 2 else 1.0)
                rest_dots = min(rest_dots, 2)
            dur = max(1, int(round(q_fraction * divisions)))
            r_default_x = r_default_y = None
            if include_layout and first_x is not None:
                r_default_x = pixels_to_tenths(rest_det.full_cx - first_x) + 15
                r_default_y = pixels_to_tenths(
                    top_line_y - rest_det.full_cy)
            rest_el = builder.add_rest(
                measure,
                duration=dur,
                duration_type=xml_type,
                dot=rest_dots,
                measure_rest=measure_rest_flag,
                voice=1,
                default_x=r_default_x,
                default_y=r_default_y,
                element_id=id_map.get(id(rest_det)),
            )
            rid = id_map.get(id(rest_det))
            if rid is not None:
                id_to_note[rid] = rest_el
            note_anchors.append((rest_el,
                                 float(rest_det.full_cx),
                                 float(rest_det.full_cy)))
            continue

        # ═════════════════════════════════════════════════════════════════
        #                          VOICE SPLIT
        # ═════════════════════════════════════════════════════════════════
        # kind == 'chord' — split into voices rather than chord-group.
        #
        # group is already sorted top-to-bottom (smallest full_cy first)
        # by group_chords, so voice numbers map directly:
        #   voice = k + 1
        # where k is the index in the sorted group.
        # ═════════════════════════════════════════════════════════════════
        group = payload
        gi = chord_idx_counter
        chord_idx_counter += 1

        # Topmost note drives the duration for the whole cluster — the
        # voices are assumed to share the same rhythm.  For non-uniform
        # durations within a cluster, see the docstring's "Voice-split
        # details" section.
        ref_note = group[0]
        dur_type, dur_value, dur_dots = determine_duration(ref_note, dets, divisions)

        anchor_full_cx = min(n.full_cx for n in group)
        n_voices = len(group)

        first_note_in_cluster = None
        for k, nh in enumerate(group):
            voice_num = k + 1

            step, octave = calculate_pitch(
                nh.full_cy, pstaff.line_positions,
                pstaff.line_spacing, clef_type,
            )
            acc_det = find_accidental_at(nh, accidentals)
            acc_name = acc_alter = None
            if acc_det is not None:
                acc_name, acc_alter = ACCIDENTAL_DETECT[acc_det.class_name]

            default_x = default_y = None
            if include_layout and first_x is not None:
                default_x = pixels_to_tenths(anchor_full_cx - first_x) + 15
                default_y = pixels_to_tenths(top_line_y - nh.full_cy)

            nh_id = id_map.get(id(nh))
            note_el = builder.add_note(
                measure,
                step=step, octave=octave,
                alter=acc_alter,
                accidental=acc_name,
                duration=dur_value,
                duration_type=dur_type,
                dot=dur_dots,
                voice=voice_num,                # << v3 change: per-voice numbering
                default_x=default_x,
                default_y=default_y,
                chord=False,                    # << v3 change: never use <chord/>
                # Slurs attach only to the topmost voice — they were
                # geometrically anchored to the cluster's left edge so
                # there's nothing meaningful to attach to lower voices.
                slur_start=(slur_starts[gi] if k == 0 else False),
                slur_stop=(slur_stops[gi] if k == 0 else False),
                element_id=nh_id,
            )
            if nh_id is not None:
                id_to_note[nh_id] = note_el
            if k == 0:
                first_note_in_cluster = note_el

            # Backup so the next voice in this cluster starts at the
            # same time as the topmost voice.  Skip on the last note —
            # we want time to advance after the final voice so the
            # next cluster picks up at the right onset.
            if k < n_voices - 1:
                _add_backup(measure, dur_value)

        if first_note_in_cluster is not None:
            note_anchors.append((
                first_note_in_cluster,
                float(anchor_full_cx),
                float(group[0].full_cy),
            ))

    # ── Post-pass: attach fermatas / ornaments to nearest anchor ──
    # Same as v2.  Anchors are the topmost (voice 1) notes; symbols
    # appearing above the staff naturally end up attached to voice 1
    # which is musically correct (fermatas span all voices but MusicXML
    # only needs them once).
    def _nearest(symbol):
        if not note_anchors:
            return None
        best = min(note_anchors,
                   key=lambda a: ((a[1] - symbol.full_cx) ** 2
                                  + (a[2] - symbol.full_cy) ** 2))
        return best[0]

    for f in fermatas:
        anchor = _nearest(f)
        if anchor is not None:
            kind = ('above' if f.class_name in FERMATA_ABOVE_CLASSES
                    else 'below')
            builder.add_fermata(anchor, type=kind)

    for o in ornaments:
        anchor = _nearest(o)
        if anchor is not None:
            ornament_kind = ORNAMENT_DETECT.get(o.class_name)
            if ornament_kind is not None:
                builder.add_ornament(anchor, ornament_kind)

    if backward_repeat:
        builder.add_repeat(measure, direction='backward')
