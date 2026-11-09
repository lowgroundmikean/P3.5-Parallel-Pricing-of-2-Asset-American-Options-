#!/usr/bin/env python3
"""
Parse the RESULT lines emitted by psor2d and build the tables and
speedup plots for AMS 530 Project 3, Problem 3.5.

    python3 collect.py psor35_<jobid>.out

Writes tables_<jobid>.txt next to the input, and, when matplotlib is
available, fig1_speedup.png, fig2_efficiency.png and fig3_karpflatt.png.
"""

import os
import re
import sys
from collections import OrderedDict


def parse(path):
    """Pull every RESULT line into a dict of typed fields."""
    rows = []
    pat = re.compile(r"(\w+)=(\([^)]*\)|[^\s]+)")
    with open(path) as fh:
        for line in fh:
            if not line.startswith("RESULT"):
                continue
            d = {}
            for key, val in pat.findall(line[len("RESULT"):]):
                if val.startswith("("):
                    d[key] = val
                else:
                    try:
                        d[key] = int(val)
                    except ValueError:
                        try:
                            d[key] = float(val)
                        except ValueError:
                            d[key] = val
            rows.append(d)
    return rows


def fmt_header(cols, widths):
    line = "  ".join(c.rjust(w) for c, w in zip(cols, widths))
    return line + "\n" + "  ".join("-" * w for w in widths)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: collect.py <psor35_JOBID.out>")

    src = sys.argv[1]
    rows = parse(src)
    if not rows:
        sys.exit("no RESULT lines found in %s" % src)

    mjob = re.search(r"_(\d+)\.", os.path.basename(src))
    jobid = mjob.group(1) if mjob else "out"
    outdir = os.path.dirname(os.path.abspath(src))
    out = []

    euro = [r for r in rows if r["style"] == "european"]
    amer = [r for r in rows if r["style"] == "american"]

    # ---------------- Table 1: European validation ----------------
    out.append("TABLE 1  Validation against the closed-form European price")
    out.append("The payoff max(K1-S1,0)+max(K2-S2,0) is separable, so the exact")
    out.append("European value is the sum of two Black-Scholes puts.")
    out.append("")
    cols = ["M", "nt", "numeric(K,K)", "exact(K,K)", "abs err", "rel err", "max err"]
    w = [6, 5, 14, 14, 11, 10, 11]
    out.append(fmt_header(cols, w))
    for r in sorted(euro, key=lambda r: r["M"]):
        ae = abs(r["probe"] - r["probe_euro_exact"])
        re_ = ae / r["probe_euro_exact"] if r["probe_euro_exact"] else 0.0
        out.append("  ".join(x.rjust(ww) for x, ww in zip(
            ["%d" % r["M"], "%d" % r["nt"], "%.8f" % r["probe"],
             "%.8f" % r["probe_euro_exact"], "%.3e" % ae, "%.3e" % re_,
             "%.3e" % r["maxerr_vs_euro"]], w)))
    out.append("")

    # ---------------- Table 2: independence of P ----------------
    by_m = OrderedDict()
    for r in amer:
        by_m.setdefault(r["M"], []).append(r)

    cons = sorted([r for r in amer if r["M"] == 256], key=lambda r: r["P"])
    if cons:
        out.append("TABLE 2  The American answer does not depend on P  (M = 256)")
        out.append("Red-black ordering is global, so the iterates are independent")
        out.append("of the decomposition. Identical columns are the correctness check.")
        out.append("")
        cols = ["P", "grid", "SOR iters", "||V||_2", "V(K,K)"]
        w = [5, 8, 11, 18, 16]
        out.append(fmt_header(cols, w))
        seen = set()
        for r in cons:
            if r["P"] in seen:
                continue
            seen.add(r["P"])
            out.append("  ".join(x.rjust(ww) for x, ww in zip(
                ["%d" % r["P"], r["dims"], "%d" % r["iters"],
                 "%.10e" % r["l2"], "%.10f" % r["probe"]], w)))
        out.append("")

    # ---------------- Table 3: early-exercise premium ----------------
    out.append("TABLE 3  Early-exercise premium at the money")
    out.append("")
    cols = ["M", "American", "European(exact)", "premium", "% of Euro"]
    w = [6, 14, 16, 12, 11]
    out.append(fmt_header(cols, w))
    for m in sorted(by_m):
        r = sorted(by_m[m], key=lambda r: r["P"])[0]
        pct = 100.0 * r["premium"] / r["probe_euro_exact"] if r["probe_euro_exact"] else 0.0
        out.append("  ".join(x.rjust(ww) for x, ww in zip(
            ["%d" % m, "%.8f" % r["probe"], "%.8f" % r["probe_euro_exact"],
             "%.6f" % r["premium"], "%.3f%%" % pct], w)))
    out.append("")

    # ---------------- Tables 4+: performance ----------------
    series = OrderedDict()
    for m in sorted(by_m):
        best = {}
        for r in by_m[m]:
            if r["P"] not in best or r["time"] < best[r["P"]]["time"]:
                best[r["P"]] = r
        series[m] = [best[p] for p in sorted(best)]

    out.append("TABLE 4  Parallel performance, American option, projected SOR")
    out.append("S = T(1)/T(P),  E = S/P,  Karp-Flatt e = (1/S - 1/P)/(1 - 1/P)")
    out.append("")
    cols = ["M", "P", "grid", "time (s)", "comm (s)", "comm %",
            "speedup", "effic", "Karp-Flatt"]
    w = [6, 5, 8, 11, 10, 8, 9, 8, 11]
    out.append(fmt_header(cols, w))
    for m, rs in series.items():
        t1 = next((r["time"] for r in rs if r["P"] == 1), None)
        for r in rs:
            s = t1 / r["time"] if t1 else float("nan")
            e = s / r["P"] if t1 else float("nan")
            if t1 and r["P"] > 1:
                kf = (1.0 / s - 1.0 / r["P"]) / (1.0 - 1.0 / r["P"])
                kfs = "%.4f" % kf
            else:
                kfs = "-"
            out.append("  ".join(x.rjust(ww) for x, ww in zip(
                ["%d" % m, "%d" % r["P"], r["dims"], "%.4f" % r["time"],
                 "%.4f" % r["comm"], "%.2f%%" % (100.0 * r["comm"] / r["time"]),
                 "%.3f" % s, "%.3f" % e, kfs], w)))
        out.append("")

    text = "\n".join(out)
    dest = os.path.join(outdir, "tables_%s.txt" % jobid)
    with open(dest, "w") as fh:
        fh.write(text + "\n")
    print(text)
    print("\nwrote %s" % dest)

    # ---------------- figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; tables written, figures skipped")
        return

    def figure(fname, ycalc, ylabel, title, ideal=None):
        fig, ax = plt.subplots(figsize=(7, 5))
        for m, rs in series.items():
            t1 = next((r["time"] for r in rs if r["P"] == 1), None)
            if not t1:
                continue
            xs = [r["P"] for r in rs]
            ys = [ycalc(r, t1) for r in rs]
            ax.plot(xs, ys, "o-", label="N = %d" % m, linewidth=1.6,
                    markersize=5)
        if ideal is not None:
            ps = sorted({r["P"] for rs in series.values() for r in rs})
            ax.plot(ps, [ideal(p) for p in ps], "k--", linewidth=1.0,
                    label="ideal")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("processes P")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("wrote %s" % path)

    figure("fig1_speedup.png", lambda r, t1: t1 / r["time"], "speedup  S(P)",
           "Projected SOR, 2-asset American put: speedup",
           ideal=lambda p: p)
    figure("fig2_efficiency.png", lambda r, t1: t1 / r["time"] / r["P"],
           "efficiency  E(P)",
           "Projected SOR: parallel efficiency", ideal=lambda p: 1.0)
    figure("fig3_commfrac.png",
           lambda r, t1: 100.0 * r["comm"] / r["time"],
           "communication share (%)",
           "Halo exchange and convergence reductions as a share of runtime")


if __name__ == "__main__":
    main()
