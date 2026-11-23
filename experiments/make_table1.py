"""Print Table II of the paper from the recomputed results and check every cell.

Usage
  python experiments/make_table1.py
Reads results/table1_reviewed/table1_reviewed.json and
results/table1_tuned/table1_tuned.json; compares with reference/paper_table1.json
(percent cells to the printed 0.05, object counts exactly, AUC/AP to 0.0005).
Writes results/table1.md. Exit code 0 only if every cell matches; the last line
is 'TABLE II REPRODUCED' or 'TABLE II MISMATCH (k cells)'.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
REF = json.load(open(os.path.join(HERE, "..", "reference", "paper_table1.json"), encoding="utf-8"))
bad = []
lines = []


def check(what, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok:
        bad.append(f"{what}: got {got}, paper {want}")
    return "" if ok else "  <-- paper " + str(want)


def out(s=""):
    print(s)
    lines.append(s)


def main():
    pr = os.path.join(RES, "table1_reviewed", "table1_reviewed.json")
    pt = os.path.join(RES, "table1_tuned", "table1_tuned.json")
    for p in (pr, pt):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing: run the table1 scripts first")
    R, T = json.load(open(pr)), json.load(open(pt))
    out("Table II. Retrieval recall (%), ties at their expected rank.")
    out("")
    out("| upper block: reviewed settings | FB top-1 | FB top-5 | PRESIOUS top-1 | PRESIOUS top-5 |")
    out("|---|---|---|---|---|")
    for arm in REF["reviewed"]["fantastic_breaks"]:
        f, p = R["fantastic_breaks"]["table"][arm], R["presious"]["table"][arm]
        wf, wp = REF["reviewed"]["fantastic_breaks"][arm], REF["reviewed"]["presious"][arm]
        notes = (check(f"FB {arm} top1", round(f["top1"], 1), wf[0], 0.05)
                 + check(f"FB {arm} top5", round(f["top5"], 1), wf[1], 0.05)
                 + check(f"PR {arm} top1", round(p["top1"], 1), wp[0], 0.05)
                 + check(f"PR {arm} top5", round(p["top5"], 1), wp[1], 0.05))
        out(f"| {arm} | {f['top1']:.1f} | {f['top5']:.1f} | {p['top1']:.1f} | {p['top5']:.1f} |{notes}")
    out("")
    out("| lower block: tuned alike | test top-1 | test top-5 | held-out top-1 | PRESIOUS top-1 |")
    out("|---|---|---|---|---|")
    for arm, w in REF["tuned"]["test_half"].items():
        t = T["test_half"]["table"][arm]
        h = T["heldout"]["table"][arm]
        p = T["presious"]["table"][arm]
        notes = (check(f"test {arm} top1", round(t["top1"], 1), w[0], 0.05)
                 + check(f"test {arm} top5", round(t["top5"], 1), w[1], 0.05)
                 + check(f"heldout {arm}", round(h["top1"], 1), REF["tuned"]["heldout_top1"][arm], 0.05)
                 + check(f"presious {arm}", round(p["top1"], 1), REF["tuned"]["presious_top1"][arm], 0.05))
        out(f"| {arm} | {t['top1']:.1f} | {t['top5']:.1f} | {h['top1']:.1f} | {p['top1']:.1f} |{notes}")
    out("")
    out("Object-level sign tests (wins:losses, objects with equal counts dropped)")
    for arm, (w, l) in REF["reviewed"]["objects_fb_datc_vs"].items():
        c = R["fantastic_breaks"]["datc_vs"][arm]
        n = check(f"FB objects DaTC vs {arm} wins", c["o_win"], w, 0) + check(f"... losses", c["o_lose"], l, 0)
        pv = "< 0.0001" if c["o_p"] < 1e-4 else f"= {c['o_p']:.4f}"
        out(f"- FB pool, DaTC vs {arm}: {c['o_win']}:{c['o_lose']}, p {pv}{n}")
    for arm, (w, l) in REF["tuned"]["objects_test_half_kt_vs_datc"].items():
        c = T["test_half"]["datc_vs"][arm]
        n = check(f"test objects {arm} vs DaTC wins", c["o_lose"], w, 0) + check("... losses", c["o_win"], l, 0)
        out(f"- test half, {arm} vs DaTC: {c['o_lose']}:{c['o_win']}, p = {c['o_p']:.2f}{n}")
    for arm, (w, l) in REF["tuned"]["objects_heldout_datc_vs"].items():
        c = T["heldout"]["datc_vs"][arm]
        n = check(f"held-out objects DaTC vs {arm} wins", c["o_win"], w, 0) + check("... losses", c["o_lose"], l, 0)
        out(f"- held-out, DaTC vs {arm}: {c['o_win']}:{c['o_lose']}, p = {c['o_p']:.2f}{n}")
    out("")
    out("Pair detection on the Fantastic Breaks pool")
    for arm, w in REF["reviewed"]["detection_fb"].items():
        d = R["fantastic_breaks"]["detection"][arm]
        n = check(f"AUC {arm}", round(d["auc"], 3), w["auc"], 0.0005) + check(f"AP {arm}", round(d["ap"], 3), w["ap"], 0.0005)
        out(f"- {arm}: ROC AUC {d['auc']:.3f}, average precision {d['ap']:.3f}{n}")
    out("")
    out("TABLE II REPRODUCED" if not bad else f"TABLE II MISMATCH ({len(bad)} cells)")
    for b in bad:
        print("  " + b)
    with open(os.path.join(RES, "table1.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
