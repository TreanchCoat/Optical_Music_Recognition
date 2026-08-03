#!/usr/bin/env python3
r"""
separate_staves.py - post-process a MusicXML file so its staves render as
independent staves (no group bracket / no connected barlines) instead of being
joined together in MuseScore, Finale, etc.

By default it sets every <part-group>'s <group-barline> to "no", so the barlines
break between staves. Optionally it can also change or remove the bracket.

USAGE
    python separate_staves.py score.xml                 # -> score_separated.xml
    python separate_staves.py score.xml -o out.xml
    python separate_staves.py score.xml --inplace       # overwrite in place
    python separate_staves.py *.xml                      # batch (many files)

    python separate_staves.py score.xml --group remove   # also drop the bracket
    python separate_staves.py score.xml --group brace     # bracket -> curly brace
    python separate_staves.py score.xml --barlines connected   # keep joined barlines

OPTIONS
    --barlines separate|connected   default: separate  (group-barline no / yes)
    --group    keep|remove|bracket|brace|line|square    default: keep
    -o, --output PATH               output file (single input only)
    --inplace                       overwrite the input file(s)

Notes
  * Works on standard MusicXML (score-partwise). It edits the <part-list> only,
    so the notes/measures are left byte-for-byte untouched.
  * If a file has no <part-group>, nothing changes and it tells you so.
"""
import argparse
import os
import re


def transform(text, barlines, group):
    changes = []

    # 1) barline connection between grouped staves
    if barlines in ("separate", "connected"):
        val = "no" if barlines == "separate" else "yes"
        text, n = re.subn(
            r'(<group-barline>)\s*(?:yes|no)\s*(</group-barline>)',
            r"\g<1>%s\g<2>" % val, text)
        if n:
            changes.append("group-barline -> %s  (%d)" % (val, n))

    # 2) the bracket itself
    if group == "remove":
        text, a = re.subn(
            r'[ \t]*<part-group\b[^>]*type="start"[^>]*>.*?</part-group>\s*',
            "", text, flags=re.S)
        text, b = re.subn(
            r'[ \t]*<part-group\b[^>]*type="stop"[^>]*/>\s*', "", text)
        if a or b:
            changes.append("removed %d part-group(s)" % max(a, b))
    elif group in ("bracket", "brace", "line", "square"):
        text, n = re.subn(
            r'(<group-symbol>)\s*\w+\s*(</group-symbol>)',
            r"\g<1>%s\g<2>" % group, text)
        if n:
            changes.append("group-symbol -> %s  (%d)" % (group, n))
    # group == "keep" -> leave the bracket as-is

    return text, changes


def main():
    ap = argparse.ArgumentParser(
        description="Make MusicXML staves render as independent staves.")
    ap.add_argument("inputs", nargs="+", help="MusicXML file(s)")
    ap.add_argument("-o", "--output", help="output path (single input only)")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite the input file(s)")
    ap.add_argument("--barlines", choices=["separate", "connected"],
                    default="separate",
                    help="break (separate) or keep (connected) barlines between staves")
    ap.add_argument("--group",
                    choices=["keep", "remove", "bracket", "brace", "line", "square"],
                    default="keep", help="what to do with the group bracket")
    args = ap.parse_args()

    if args.output and len(args.inputs) > 1:
        ap.error("-o/--output can only be used with a single input file")

    for path in args.inputs:
        if not os.path.isfile(path):
            print("!! not found:", path)
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        out, changes = transform(text, args.barlines, args.group)

        if args.inplace:
            dest = path
        elif args.output:
            dest = args.output
        else:
            base, ext = os.path.splitext(path)
            dest = base + "_separated" + (ext or ".xml")

        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(out)

        print("%s  ->  %s" % (os.path.basename(path), os.path.basename(dest)))
        for c in changes:
            print("    ", c)
        if not changes:
            print("     (no <part-group>/<group-barline> found - nothing to change)")


if __name__ == "__main__":
    main()
