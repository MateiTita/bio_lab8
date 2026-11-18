#!/usr/bin/env python3
"""DNA pattern generator: C+G% and information content (kappa IC) over a sliding window.

Each sliding window of a sequence yields a point (C+G%, IC). The cloud of points
is the sequence's *pattern*; its mean point is the *center of weight*.

Algorithms follow P. A. Gagniuc, "Algorithms in Bioinformatics: Theory and
Implementation" (Wiley, 2021) / PromKappa.

Usage:
    python3 dna_patterns.py                       # built-in test sequence S
    python3 dna_patterns.py genes.fasta -w 30     # sequences from FASTA file(s)
    python3 dna_patterns.py ACGTTGCA... -w 20     # raw sequence(s) as parameters
Pure standard library; writes an HTML report (default: dna_patterns.html).
"""

import argparse
import html
import json
import os
import sys

TEST_SEQUENCE = "CGGACTGATCTATCT" + "A" * 27 + "CGTAGCATCTATCGATCTATCTAGCGATCTA"
EXPECTED_CG = 29.27
EXPECTED_IC = 27.53


# ---------------------------------------------------------------- core algorithms

def cg_content(seq):
    """C+G percentage of a sequence."""
    if not seq:
        return 0.0
    return (seq.count("C") + seq.count("G")) / len(seq) * 100.0


def info_content(seq):
    """Kappa index of coincidence (self-sequence alignment), in percent.

    For every shift u = 1..N-1 the sequence is aligned against itself shifted by u;
    the share of matching positions over the N-u overlap is averaged over all shifts.
    """
    n = len(seq)
    if n < 2:
        return 0.0
    total = 0.0
    for u in range(1, n):
        matches = sum(1 for i in range(n - u) if seq[i] == seq[i + u])
        total += matches / (n - u) * 100.0
    return total / (n - 1)


def sliding_windows(seq, window):
    """Yield (start_position, window_string) for every full window (step 1)."""
    for i in range(len(seq) - window + 1):
        yield i, seq[i:i + window]


def build_pattern(seq, window):
    """Return the pattern: a list of dicts {pos, cg, ic}, one per sliding window."""
    return [{"pos": i, "cg": cg_content(w), "ic": info_content(w)}
            for i, w in sliding_windows(seq, window)]


def center_of_weight(pattern):
    """Center of weight (centroid) of a pattern in (C+G%, IC) space."""
    if not pattern:
        return None
    k = len(pattern)
    return {"cg": sum(p["cg"] for p in pattern) / k,
            "ic": sum(p["ic"] for p in pattern) / k}


# ---------------------------------------------------------------- input handling

VALID = set("ACGT")


def clean(seq):
    """Upper-case, drop whitespace/digits; keep only A/C/G/T (N and others removed)."""
    return "".join(ch for ch in seq.upper() if ch in VALID)


def read_fasta(path):
    records, name, chunks = [], None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, clean("".join(chunks))))
                name, chunks = line[1:].strip() or f"seq{len(records) + 1}", []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, clean("".join(chunks))))
    elif chunks:  # plain file without header
        records.append((os.path.basename(path), clean("".join(chunks))))
    return records


def load_inputs(params):
    records = []
    for p in params:
        if os.path.isfile(p):
            records.extend(read_fasta(p))
        else:
            records.append((f"param{len(records) + 1}", clean(p)))
    return records


# ---------------------------------------------------------------- SVG report

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLORS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

W, H = 760, 380
ML, MR, MT, MB = 56, 24, 20, 48


def _ticks(lo, hi, n=5):
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def _axes(x_lo, x_hi, y_lo, y_hi, x_label, y_label, x_fmt="{:.0f}", y_fmt="{:.0f}"):
    sx = lambda v: ML + (v - x_lo) / (x_hi - x_lo or 1) * (W - ML - MR)
    sy = lambda v: H - MB - (v - y_lo) / (y_hi - y_lo or 1) * (H - MT - MB)
    out = []
    for t in _ticks(y_lo, y_hi):
        y = sy(t)
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{y_fmt.format(t)}</text>')
    for t in _ticks(x_lo, x_hi):
        x = sx(t)
        out.append(f'<text class="tick" x="{x:.1f}" y="{H - MB + 18}" text-anchor="middle">{x_fmt.format(t)}</text>')
    out.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{H - MB}" y2="{H - MB}"/>')
    out.append(f'<text class="label" x="{(ML + W - MR) / 2}" y="{H - 8}" text-anchor="middle">{x_label}</text>')
    out.append(f'<text class="label" transform="translate(14,{(MT + H - MB) / 2}) rotate(-90)" '
               f'text-anchor="middle">{y_label}</text>')
    return sx, sy, out


def _range(vals, pad=0.08):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        lo, hi = lo - 5, hi + 5
    d = (hi - lo) * pad
    return max(0.0, lo - d), min(100.0, hi + d)


def _tip(**kv):
    return html.escape(json.dumps(kv), quote=True)


def svg_signals(name, pattern, color_idx):
    """Chart 1: C+G% and IC along the sequence (both in %, one shared axis)."""
    xs = [p["pos"] for p in pattern]
    ys = [p["cg"] for p in pattern] + [p["ic"] for p in pattern]
    y_lo, y_hi = _range(ys)
    sx, sy, parts = _axes(min(xs), max(xs) or 1, y_lo, y_hi, "Window start position (bp)", "Percent (%)")
    series = [("C+G %", "cg", "s1"), ("IC", "ic", "s2")]
    for label, key, cls in series:
        pts = " ".join(f"{sx(p['pos']):.1f},{sy(p[key]):.1f}" for p in pattern)
        parts.append(f'<polyline class="line {cls}" points="{pts}"/>')
        last = pattern[-1]
        parts.append(f'<text class="direct" x="{sx(last["pos"]) - 4:.1f}" y="{sy(last[key]) - 8:.1f}" '
                     f'text-anchor="end">{label}</text>')
    # hover columns
    step = (W - ML - MR) / max(1, len(pattern) - 1)
    for p in pattern:
        x = sx(p["pos"])
        parts.append(f'<rect class="hit" x="{x - step / 2:.1f}" y="{MT}" width="{step:.1f}" height="{H - MT - MB}" '
                     f'data-tip="{_tip(Sequence=name, Position=p["pos"], **{"C+G %": round(p["cg"], 2), "IC": round(p["ic"], 2)})}" '
                     f'data-x="{x:.1f}"/>')
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{MT}" y2="{H - MB}"/>')
    return _svg(parts, f"C+G% and IC along {name}")


def svg_pattern(results):
    """Chart 2: the pattern cloud(s) in (C+G%, IC) space, with centers of weight."""
    allp = [p for r in results for p in r["pattern"]]
    x_lo, x_hi = _range([p["cg"] for p in allp])
    y_lo, y_hi = _range([p["ic"] for p in allp])
    sx, sy, parts = _axes(x_lo, x_hi, y_lo, y_hi, "C+G (%)", "Information content, kappa IC (%)")
    for r in results:
        c = f"c{r['idx']}"
        for p in r["pattern"]:
            parts.append(f'<circle class="dot {c}" cx="{sx(p["cg"]):.1f}" cy="{sy(p["ic"]):.1f}" r="4" '
                         f'data-tip="{_tip(Sequence=r["name"], Window=p["pos"], **{"C+G %": round(p["cg"], 2), "IC": round(p["ic"], 2)})}"/>')
    for r in results:
        cw = r["center"]
        parts.append(f'<g class="center c{r["idx"]}" data-tip="{_tip(Sequence=r["name"], Center="of weight", **{"C+G %": round(cw["cg"], 2), "IC": round(cw["ic"], 2)})}">'
                     f'<circle cx="{sx(cw["cg"]):.1f}" cy="{sy(cw["ic"]):.1f}" r="9"/>'
                     f'<path d="M{sx(cw["cg"]) - 5:.1f},{sy(cw["ic"]):.1f}h10M{sx(cw["cg"]):.1f},{sy(cw["ic"]) - 5:.1f}v10"/></g>')
    return _svg(parts, "Pattern: C+G% vs IC per window")


def svg_centers(results):
    """Chart 3: only the center of weight of each pattern."""
    xs = [r["center"]["cg"] for r in results]
    ys = [r["center"]["ic"] for r in results]
    x_lo, x_hi = _range(xs, 0.25)
    y_lo, y_hi = _range(ys, 0.25)
    sx, sy, parts = _axes(x_lo, x_hi, y_lo, y_hi, "C+G (%) — center of weight",
                          "IC (%) — center of weight", "{:.1f}", "{:.1f}")
    placed = []
    for r in results:
        cw = r["center"]
        x, y = sx(cw["cg"]), sy(cw["ic"])
        parts.append(f'<circle class="dot big c{r["idx"]}" cx="{x:.1f}" cy="{y:.1f}" r="7" '
                     f'data-tip="{_tip(Sequence=r["name"], **{"C+G %": round(cw["cg"], 2), "IC": round(cw["ic"], 2), "Windows": len(r["pattern"])})}"/>')
        placed.append([x, y + 4, r["name"]])
    # nudge labels vertically so that no two overlap
    placed.sort(key=lambda t: t[1])
    for i in range(1, len(placed)):
        if placed[i][1] - placed[i - 1][1] < 18:
            placed[i][1] = placed[i - 1][1] + 18
    for x, ly, name in placed:
        anchor = "end" if x > W * 0.6 else "start"
        dx = -12 if anchor == "end" else 12
        parts.append(f'<text class="direct" x="{x + dx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">'
                     f'{html.escape(name[:40])}</text>')
    return _svg(parts, "Centers of weight of all patterns")


def _svg(parts, title):
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
            + "".join(parts) + "</svg>")


def render_html(results, window, check):
    color_css = "\n".join(
        f"  .c{i} {{ --c: var(--series-{i % 8 + 1}); }}" for i in range(len(results)))
    light = "\n".join(f"  --series-{i + 1}: {c};" for i, c in enumerate(COLORS))
    dark = "\n".join(f"  --series-{i + 1}: {c};" for i, c in enumerate(COLORS_DARK))

    legend = "".join(
        f'<span class="key"><i class="sw c{r["idx"]}"></i>{html.escape(r["name"])}</span>' for r in results)

    rows = "".join(
        f'<tr><td><i class="sw c{r["idx"]}"></i>{html.escape(r["name"])}</td><td>{len(r["seq"])}</td>'
        f'<td>{len(r["pattern"])}</td><td>{r["whole_cg"]:.2f}</td><td>{r["whole_ic"]:.2f}</td>'
        f'<td>{r["center"]["cg"]:.2f}</td><td>{r["center"]["ic"]:.2f}</td></tr>' for r in results)

    check_html = ""
    if check:
        check_html = (
            '<section><h2>Verification on test sequence S</h2>'
            f'<p class="seq">{check["seq"]}</p>'
            '<div class="table-wrap"><table><tr><th>Measure</th><th>Expected (assignment)</th><th>Whole S</th><th>Mean over windows</th></tr>'
            f'<tr><td>C+G %</td><td>≈ {EXPECTED_CG}</td><td>{check["cg"]:.2f}</td><td>{check["cg_mean"]:.2f}</td></tr>'
            f'<tr><td>IC (kappa)</td><td>≈ {EXPECTED_IC}</td><td>{check["ic"]:.2f}</td><td>{check["ic_mean"]:.2f}</td></tr>'
            '</table></div></section>')

    signal_blocks = "".join(
        f'<figure><figcaption>{html.escape(r["name"])} — C+G % and IC per window</figcaption>'
        f'{svg_signals(r["name"], r["pattern"], r["idx"])}</figure>' for r in results)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DNA Pattern Report</title>
<style>
:root {{
  color-scheme: light;
  --surface: #fcfcfb; --text: #0b0b0b; --text-2: #52514e; --grid: #e6e5e0; --axis: #b9b8b1;
{light}
  --s1: var(--series-1); --s2: var(--series-2);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --surface: #1a1a19; --text: #f0efec; --text-2: #c3c2b7; --grid: #2e2e2c; --axis: #55544f;
{dark}
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface: #1a1a19; --text: #f0efec; --text-2: #c3c2b7; --grid: #2e2e2c; --axis: #55544f;
{dark}
}}
{color_css}
body {{ margin: 0; padding: 24px 16px 48px; background: var(--surface); color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 820px; margin: 0 auto; min-width: 0; }}
section, figure {{ max-width: 100%; min-width: 0; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 32px 0 8px; }}
p, figcaption, .note {{ color: var(--text-2); }}
figure {{ margin: 12px 0 24px; }} figcaption {{ font-size: 14px; margin-bottom: 4px; }}
svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }} .axis {{ stroke: var(--axis); stroke-width: 1; }}
.tick {{ fill: var(--text-2); font-size: 12px; font-variant-numeric: tabular-nums; }}
.label {{ fill: var(--text-2); font-size: 13px; }}
.direct {{ fill: var(--text); font-size: 13px; font-weight: 600; paint-order: stroke;
  stroke: var(--surface); stroke-width: 4px; }}
.line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.line.s1 {{ stroke: var(--s1); }} .line.s2 {{ stroke: var(--s2); }}
.dot {{ fill: var(--c); fill-opacity: .75; stroke: var(--surface); stroke-width: 2; }}
.dot.big {{ fill-opacity: 1; }}
.center circle {{ fill: var(--surface); stroke: var(--c); stroke-width: 3; }}
.center path {{ stroke: var(--text); stroke-width: 2; }}
.hit {{ fill: transparent; }}
.crosshair {{ stroke: var(--text-2); stroke-width: 1; stroke-dasharray: 3 3; visibility: hidden; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 14px; }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--c); margin-right: 6px; }}
.sw.s1 {{ background: var(--s1); }} .sw.s2 {{ background: var(--s2); }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--text-2); font-weight: 600; }}
.seq {{ font-family: ui-monospace, monospace; font-size: 13px; word-break: break-all; }}
#tip {{ position: fixed; pointer-events: none; background: var(--surface); color: var(--text);
  border: 1px solid var(--axis); border-radius: 6px; padding: 6px 10px; font-size: 13px;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); display: none; z-index: 10; }}
#tip b {{ font-weight: 600; }}
</style></head>
<body><main>
<h1>DNA pattern report</h1>
<p>Sliding window = {window} bp, step 1. Each window gives one point (C+G %, IC). The cloud of points is the
sequence's pattern, and its centroid is the center of weight.</p>
{check_html}
<section><h2>Summary</h2><div class="table-wrap"><table>
<tr><th>Sequence</th><th>Length</th><th>Windows</th><th>C+G % (whole)</th><th>IC (whole)</th>
<th>Center C+G %</th><th>Center IC</th></tr>{rows}</table></div></section>
<section><h2>1 · Signals along each sequence</h2>
<div class="legend"><span class="key"><i class="sw s1"></i>C+G %</span><span class="key"><i class="sw s2"></i>IC</span></div>
{signal_blocks}</section>
<section><h2>2 · Pattern (C+G % vs IC), with center of weight ⊕</h2>
<div class="legend">{legend}</div>
<figure>{svg_pattern(results)}</figure></section>
<section><h2>3 · Centers of weight</h2>
<div class="legend">{legend}</div>
<figure>{svg_centers(results)}</figure></section>
</main>
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {{
  const svg = el.ownerSVGElement;
  const cross = svg && svg.querySelector('.crosshair');
  el.addEventListener('mousemove', e => {{
    const d = JSON.parse(el.dataset.tip);
    tip.innerHTML = Object.entries(d).map(([k, v]) => `${{k}}: <b>${{v}}</b>`).join('<br>');
    tip.style.display = 'block';
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + 'px'; tip.style.top = (e.clientY + 14) + 'px';
    if (cross && el.dataset.x) {{
      cross.setAttribute('x1', el.dataset.x); cross.setAttribute('x2', el.dataset.x);
      cross.style.visibility = 'visible';
    }}
  }});
  el.addEventListener('mouseleave', () => {{
    tip.style.display = 'none';
    if (cross) cross.style.visibility = 'hidden';
  }});
}});
</script>
</body></html>
"""


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="FASTA file(s) and/or raw DNA sequences")
    ap.add_argument("-w", "--window", type=int, default=30, help="sliding window length (default 30)")
    ap.add_argument("-o", "--output", default="dna_patterns.html", help="HTML report path")
    ap.add_argument("--no-test", action="store_true", help="do not include the test sequence S")
    args = ap.parse_args(argv)

    records = [] if args.no_test else [("S (test)", TEST_SEQUENCE)]
    records += load_inputs(args.inputs)
    if not records:
        ap.error("no sequences given")

    results = []
    for name, seq in records:
        if len(seq) < args.window:
            print(f"skip {name}: length {len(seq)} < window {args.window}", file=sys.stderr)
            continue
        pattern = build_pattern(seq, args.window)
        results.append({"idx": len(results), "name": name, "seq": seq, "pattern": pattern,
                        "center": center_of_weight(pattern),
                        "whole_cg": cg_content(seq), "whole_ic": info_content(seq)})
    if not results:
        sys.exit("no sequence is at least as long as the window")

    check = None
    if not args.no_test:
        r = results[0]
        check = {"seq": r["seq"], "cg": r["whole_cg"], "ic": r["whole_ic"],
                 "cg_mean": r["center"]["cg"], "ic_mean": r["center"]["ic"]}
        print(f"Test sequence S ({len(r['seq'])} bp), window {args.window}:")
        print(f"  C+G %  whole = {check['cg']:.2f}   window mean = {check['cg_mean']:.2f}   (assignment: ~{EXPECTED_CG})")
        print(f"  IC     whole = {check['ic']:.2f}   window mean = {check['ic_mean']:.2f}   (assignment: ~{EXPECTED_IC})")

    print(f"\n{'sequence':<30}{'len':>6}{'windows':>9}{'center C+G%':>13}{'center IC':>11}")
    for r in results:
        print(f"{r['name'][:30]:<30}{len(r['seq']):>6}{len(r['pattern']):>9}"
              f"{r['center']['cg']:>13.2f}{r['center']['ic']:>11.2f}")

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(render_html(results, args.window, check))
    print(f"\nReport written to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
