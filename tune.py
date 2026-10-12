"""Fase 5 (+ pratinjau Fase 6): pipeline semalam yang bisa dihentikan dan dilanjutkan.

Tahap (dilewati jika sudah selesai):
  S1 seedbag_dev  : seed 43/44 untuk model dev Fase 4 (CatBoost, XGBoost, LogReg, MLP)
  S2 tune_lgb     : Optuna LightGBM pada B3 (fold 0-2), study tersimpan di SQLite
  S3 tune_cat     : Optuna CatBoost
  S4 tune_xgb     : Optuna XGBoost
  S5 final_train  : parameter terbaik, learning rate kecil, 5 fold x 3 seed
  S6 ensemble     : blend dengan OOF 3 seed, nested CV, Caruana -> kandidat submission + laporan

Contoh:
  python tune.py --stop-at 06:00                 # jalankan semua tahap, berhenti rapi sebelum 06.00
  python tune.py                                 # lanjutkan (tahap selesai dilewati, study dilanjutkan)
  python tune.py --smoke                         # uji coba mini (data kecil, 2 trial), tidak menyentuh hasil asli
Status: experiments/p5/status.json, log: experiments/p5/pipeline.log
"""
import argparse
import ctypes
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import run
from src import selection as sel
from src.config import N_FOLDS, ROOT, SEED
from src.io_utils import load_apps
from src.models import run_model_cv
from src.train import Timer, ks_score, log_experiment, make_folds, write_submission
from src.tuning import FINAL_LR, fit_fold, full_params, make_objective, params_hash

SEEDS = [42, 43, 44]
SEARCH_FOLDS = [0, 1, 2]


class Ctx:
    """Konteks run: data, deadline, path (mode smoke memakai folder terpisah)."""

    def __init__(self, args):
        self.smoke = args.smoke
        self.dir = ROOT / "experiments" / ("p5_smoke" if args.smoke else "p5")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.oof_dir = self.dir / "oof" if args.smoke else ROOT / "oof"
        self.preds_dir = self.dir / "preds" if args.smoke else ROOT / "preds"
        self.oof_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        self.storage = f"sqlite:///{(self.dir / 'optuna.db').as_posix()}"
        self.deadline = parse_deadline(args.stop_at)
        self.args = args
        self.status_path = self.dir / "status.json"
        self.status = json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        self._load_data()

    def _load_data(self):
        # Hemat memori (RAM laptop 7,7 GB; run 12 Okt dihentikan karena memori kritis): matriks penuh
        # langsung dipangkas ke kolom terpilih lalu dibuang sebelum matriks berikutnya dimuat.
        apps = load_apps()
        m = apps["is_test"].eq(0).values
        groups, cols, _ = run.resolve(dict(base="B3"))
        keep = set(sel.load_featset(cols))
        X = run.build_matrix(groups, apps)
        X = X[[c for c in X.columns if c in keep]]
        gc.collect()
        self.y = apps.loc[m, "target"].astype(int).values
        self.fold = make_folds(apps[m])["fold"].values
        self.test_ids = apps.loc[~m, "application_id"].values
        self.X_tr, self.X_te = X[m].reset_index(drop=True), X[~m].reset_index(drop=True)
        del X
        sel_b2 = set(sel.load_featset("sel_b2"))
        Xb2 = run.build_matrix(run.parse_feature_set(run.base_row("E71c")["feature_set"])[0], apps)
        Xb2 = Xb2[[c for c in Xb2.columns if c in sel_b2]]
        gc.collect()
        self.Xb2_tr, self.Xb2_te = Xb2[m].reset_index(drop=True), Xb2[~m].reset_index(drop=True)
        del Xb2, apps
        gc.collect()
        if self.smoke:  # 6.000 baris train, 800 baris test, fold tetap dari folds.csv
            idx = np.random.RandomState(SEED).choice(len(self.y), 6000, replace=False)
            idx.sort()
            self.y, self.fold = self.y[idx], self.fold[idx]
            self.X_tr, self.Xb2_tr = self.X_tr.iloc[idx], self.Xb2_tr.iloc[idx]
            self.X_te, self.Xb2_te = self.X_te.iloc[:800], self.Xb2_te.iloc[:800]
            self.test_ids = self.test_ids[:800]
            for a in ("X_tr", "Xb2_tr", "X_te", "Xb2_te"):
                setattr(self, a, getattr(self, a).reset_index(drop=True))

    def log(self, msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.dir / "pipeline.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_status(self, **kw):
        self.status.update(kw, updated=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
        self.status_path.write_text(json.dumps(self.status, indent=2, default=str))

    def seconds_left(self):
        return float("inf") if self.deadline is None else (self.deadline - datetime.now()).total_seconds()

    def can_start(self, est_seconds):
        """Jangan mulai pekerjaan yang diperkirakan melewati deadline (margin 2 menit)."""
        return self.seconds_left() > est_seconds + 120


def parse_deadline(s):
    if not s:
        return None
    h, m = map(int, s.split(":"))
    now = datetime.now()
    d = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return d if d > now else d + timedelta(days=1)


def keep_awake(on=True):
    """Cegah Windows sleep karena idle selama proses berjalan (tidak mengubah pengaturan power)."""
    try:
        flags = 0x80000000 | 0x00000001 if on else 0x80000000  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


# ------------------------------------------------------------------------- S1 seedbag dev
DEV_MODELS = ["E80", "E81", "E82", "E83"]  # CatBoost, XGBoost, LogReg, MLP (Fase 4)


def stage_seedbag(ctx):
    est = {"E80": 660, "E81": 150, "E82": 45, "E83": 150}
    for e in DEV_MODELS:
        cfg = run.EXPERIMENTS[e]
        for s in SEEDS[1:]:
            path = ctx.oof_dir / f"{e}_s{s}.npy"
            if path.exists():
                continue
            if not ctx.can_start(est[e] * (0.2 if ctx.smoke else 1)):
                ctx.log(f"S1: berhenti sebelum {e} seed {s} (deadline)")
                return False
            with Timer() as t:
                oof, pred, aucs, _, _ = run_model_cv(cfg["model"], cfg.get("params", {}), ctx.X_tr,
                                                     ctx.y, ctx.fold, ctx.X_te, s, log=lambda m: None)
            np.save(path, oof)
            np.save(ctx.preds_dir / f"{e}_s{s}.npy", pred)
            ctx.log(f"S1: {e} seed {s} AUC mean-fold {np.mean(aucs):.5f} ({t.minutes:.1f} mnt)")
    return True


# -------------------------------------------------------------------------- S2-S4 tuning
def stage_tune(ctx, model):
    a = ctx.args
    target = {"lgb": a.trials_lgb, "cat": a.trials_cat, "xgb": a.trials_xgb}[model]
    budget = {"lgb": a.budget_lgb, "cat": a.budget_cat, "xgb": a.budget_xgb}[model] * 60
    if ctx.smoke:
        target, budget = 2, 600
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=f"tune_{model}", storage=ctx.storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    done = lambda: len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
    start, durations, fails = time.time(), [], 0
    ctx.log(f"S-{model}: mulai, sudah {done()} trial, target {target}, anggaran {budget / 60:.0f} mnt")
    objective = make_objective(model, ctx.X_tr, ctx.y, ctx.fold, SEARCH_FOLDS, ctx.log)
    while done() < target:
        est = np.median(durations) if durations else {"lgb": 60, "cat": 300, "xgb": 120}[model]
        if time.time() - start + est > budget:
            ctx.log(f"S-{model}: anggaran habis ({done()}/{target} trial)")
            break
        if not ctx.can_start(est * 1.3):
            ctx.log(f"S-{model}: berhenti (deadline), {done()}/{target} trial")
            break
        t0 = time.time()
        study.optimize(objective, n_trials=1, catch=(Exception,))
        durations.append(time.time() - t0)
        tr = study.trials[-1]
        fails = fails + 1 if tr.state.name == "FAIL" else 0
        if fails >= 3:
            raise RuntimeError(f"3 trial {model} gagal berturut-turut: {tr.system_attrs}")
        best = study.best_trial if any(t.state.name == "COMPLETE" for t in study.trials) else None
        ctx.log(f"S-{model}: trial {tr.number} {tr.state.name} "
                f"{'' if tr.value is None else f'AUC {tr.value:.5f} '}"
                f"({durations[-1] / 60:.1f} mnt) | terbaik {best.value if best else float('nan'):.5f}")
        ctx.save_status(**{f"tune_{model}": dict(trials=done(), target=target,
                                                 best=None if best is None else best.value)})
    return done() >= target


def best_tuned(ctx, model, min_trials):
    try:
        study = optuna.load_study(study_name=f"tune_{model}", storage=ctx.storage)
    except KeyError:
        return None, 0
    comp = [t for t in study.trials if t.state.name == "COMPLETE"]
    if len(comp) < (1 if ctx.smoke else min_trials):
        return None, len(comp)
    return study.best_trial.params, len(comp)


# --------------------------------------------------------------------------- S5 final
def final_specs(ctx):
    """Model final: (nama, model, parameter penuh, set fitur). Nama memuat hash parameter."""
    specs = []
    for model, label, min_trials in [("cat", "CatBoost", 8), ("xgb", "XGBoost", 12),
                                     ("lgb", "LightGBM", 15)]:
        tuned, n = best_tuned(ctx, model, min_trials)
        if tuned is None:
            ctx.log(f"S5: {label} dilewati (baru {n} trial selesai)")
            continue
        h = params_hash(model, tuned)
        p = full_params(model, tuned, FINAL_LR[model])
        specs.append((f"T_{model}_b3_{h}", model, p, "b3"))
        if model == "lgb":
            specs.append((f"T_lgb_selb2_{h}", model, p, "selb2"))
    return specs


def stage_final(ctx):
    est = {"cat": 1200, "xgb": 420, "lgb": 240}
    all_done = True
    for name, model, params, fs in final_specs(ctx):
        Xtr, Xte = (ctx.X_tr, ctx.X_te) if fs == "b3" else (ctx.Xb2_tr, ctx.Xb2_te)
        for s in SEEDS:
            path = ctx.oof_dir / f"{name}_s{s}.npy"
            if path.exists():
                continue
            if not ctx.can_start(est[model] * (0.1 if ctx.smoke else 1)):
                ctx.log(f"S5: berhenti sebelum {name} seed {s} (deadline)")
                return False
            oof, pred, aucs, iters = np.zeros(len(ctx.y)), np.zeros(len(Xte)), [], []
            with Timer() as t:
                for k in range(N_FOLDS):
                    tr, va = ctx.fold != k, ctx.fold == k
                    p_va, p_te, it = fit_fold(model, params, Xtr[tr], ctx.y[tr], Xtr[va],
                                              ctx.y[va], Xte, s)
                    oof[va], pred = p_va, pred + p_te / N_FOLDS
                    aucs.append(roc_auc_score(ctx.y[va], p_va))
                    iters.append(it)
            np.save(path, oof)
            np.save(ctx.preds_dir / f"{name}_s{s}.npy", pred)
            ctx.log(f"S5: {name} seed {s}: AUC mean-fold {np.mean(aucs):.5f} KS "
                    f"{ks_score(ctx.y, oof):.4f} iter {iters} ({t.minutes:.1f} mnt)")
        bag_oof, bag_pred = bag(ctx, name)
        if bag_oof is None:
            all_done = False
            continue
        np.save(ctx.oof_dir / f"{name}.npy", bag_oof)
        np.save(ctx.preds_dir / f"{name}.npy", bag_pred)
        aucs = [roc_auc_score(ctx.y[ctx.fold == k], bag_oof[ctx.fold == k]) for k in range(N_FOLDS)]
        ctx.log(f"S5: {name} BAG 3 seed: AUC mean-fold {np.mean(aucs):.5f} KS "
                f"{ks_score(ctx.y, bag_oof):.4f}")
        if not ctx.smoke and not ctx.status.get("logged", {}).get(name):
            log_experiment(name, f"{model} tuned (Optuna), lr {params['learning_rate']}, bag 3 seed, "
                           f"fitur {fs}", "B3" if fs == "b3" else "sel_b2", Xtr.shape[1],
                           f"{model}_tuned_bag3", params, aucs,
                           roc_auc_score(ctx.y, bag_oof), 0, "kandidat", ks_score(ctx.y, bag_oof),
                           [ks_score(ctx.y[ctx.fold == k], bag_oof[ctx.fold == k])
                            for k in range(N_FOLDS)])
            ctx.save_status(logged={**ctx.status.get("logged", {}), name: True})
        (ctx.dir / f"params_{name}.json").write_text(json.dumps(params, indent=2, default=str))
    return all_done


def bag(ctx, name):
    """Rata-rata OOF & prediksi test 3 seed (seed 42 model dev Fase 4 = file tanpa akhiran)."""
    oofs, preds = [], []
    for s in SEEDS:
        o = ctx.oof_dir / f"{name}_s{s}.npy"
        if not o.exists() and s == 42:
            o = ctx.oof_dir / f"{name}.npy"
        if not o.exists():
            return None, None
        oofs.append(np.load(o))
        preds.append(np.load(ctx.preds_dir / o.name))
    if len(oofs[0]) != len(ctx.y):
        return None, None
    return np.mean(oofs, axis=0), np.mean(preds, axis=0)


# --------------------------------------------------------------------------- S6 ensemble
def mean_fold(ctx, p, folds=None):
    folds = range(N_FOLDS) if folds is None else folds
    return float(np.mean([roc_auc_score(ctx.y[ctx.fold == k], p[ctx.fold == k]) for k in folds]))


def caruana(ctx, R, folds, steps=40):
    """Hill climbing Caruana dengan pengembalian, memaksimalkan mean-fold AUC pada `folds`."""
    names = list(R)
    counts = {n: 0 for n in names}
    first = max(names, key=lambda n: mean_fold(ctx, R[n], folds))
    counts[first], cur, best = 1, R[first].copy(), mean_fold(ctx, R[first], folds)
    for _ in range(steps):
        tot = sum(counts.values())
        cand = {n: (cur * tot + R[n]) / (tot + 1) for n in names}
        n_best = max(names, key=lambda n: mean_fold(ctx, cand[n], folds))
        score = mean_fold(ctx, cand[n_best], folds)
        if score <= best + 1e-6:
            break
        counts[n_best] += 1
        cur, best = cand[n_best], score
    tot = sum(counts.values())
    return {n: c / tot for n, c in counts.items() if c}


def stage_ensemble(ctx):
    cands = {}
    names = ["E71b", "E71c", "E87"] + DEV_MODELS + [s[0] for s in final_specs(ctx)]
    for n in names:
        o, p = bag(ctx, n)
        if o is not None:
            cands[n] = (o, p)
    if len(cands) < 2:
        ctx.log("S6: kandidat < 2, dilewati")
        return True
    R = {n: rankdata(o) / len(o) for n, (o, _) in cands.items()}
    Rt = {n: rankdata(p) / len(p) for n, (_, p) in cands.items()}
    rows = [dict(model=n, auc=mean_fold(ctx, cands[n][0]), ks=ks_score(ctx.y, cands[n][0]))
            for n in cands]
    single = pd.DataFrame(rows).sort_values("auc", ascending=False)
    best_single = single.iloc[0]["model"]

    # Nested CV: bobot Caruana dicari di 4 fold, dinilai di fold ke-5.
    nested = np.zeros(len(ctx.y))
    for k in range(N_FOLDS):
        w = caruana(ctx, R, [j for j in range(N_FOLDS) if j != k])
        nested[ctx.fold == k] = sum(w[n] * R[n][ctx.fold == k] for n in w)
    w_all = caruana(ctx, R, list(range(N_FOLDS)))
    blend_oof = sum(w_all[n] * R[n] for n in w_all)
    blend_test = sum(w_all[n] * Rt[n] for n in w_all)

    # Stabilitas KS: seberapa sering blend nested > model tunggal terbaik di potongan 10%/30%.
    rng = np.random.RandomState(SEED)
    stab = {}
    for frac in (0.1, 0.3):
        n = int(len(ctx.y) * frac)
        d = [ks_score(ctx.y[i], nested[i]) - ks_score(ctx.y[i], R[best_single][i])
             for i in (rng.choice(len(ctx.y), n, replace=False) for _ in range(300))]
        stab[frac] = (float(np.mean(d)), float(np.mean(np.array(d) > 0)))

    lines = [f"# Laporan Fase 5 / pratinjau Fase 6 ({datetime.now():%Y-%m-%d %H:%M})", "",
             "Semua angka memakai OOF rata-rata 3 seed (aturan dari pelajaran P4).", "",
             "## Model tunggal (bag 3 seed)", "", "| Model | Mean-fold AUC | OOF KS |", "|---|---|---|"]
    lines += [f"| {r.model} | {r.auc:.5f} | {r.ks:.4f} |" for r in single.itertuples()]
    lines += ["", "## Ensemble", "",
              f"- Nested CV Caruana: mean-fold AUC {mean_fold(ctx, nested):.5f}, KS "
              f"{ks_score(ctx.y, nested):.4f} (vs tunggal terbaik {best_single}: "
              f"{single.iloc[0]['auc']:.5f} / {single.iloc[0]['ks']:.4f})",
              f"- Bobot Caruana (semua fold): " + ", ".join(f"{n} {w:.2f}" for n, w in w_all.items()),
              f"- Stabilitas KS nested blend - {best_single}: potongan 10% rata-rata "
              f"{stab[0.1][0]:+.4f} (menang {stab[0.1][1]:.0%}), potongan 30% {stab[0.3][0]:+.4f} "
              f"(menang {stab[0.3][1]:.0%})"]
    if not ctx.smoke:
        p1, h1 = write_submission(ctx.test_ids, cands[best_single][1], f"P5_single_{best_single}")
        p2, h2 = write_submission(ctx.test_ids, blend_test,
                                  f"P5_caruana_nested{mean_fold(ctx, nested):.5f}")
        lines += ["", "## File submission kandidat", "",
                  f"- `{p1.name}` (md5 {h1}): model tunggal terbaik, bag 3 seed",
                  f"- `{p2.name}` (md5 {h2}): blend Caruana"]
        np.save(ROOT / "oof" / "P5_caruana.npy", blend_oof)
        np.save(ROOT / "preds" / "P5_caruana.npy", blend_test)
    (ctx.dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    ctx.log("S6: laporan ditulis ke " + str(ctx.dir / "report.md"))
    for l in lines:
        ctx.log("    " + l)
    return True


STAGES = [
    ("S1_seedbag_dev", stage_seedbag),
    ("S2_tune_lgb", lambda c: stage_tune(c, "lgb")),
    ("S3_tune_cat", lambda c: stage_tune(c, "cat")),
    ("S4_tune_xgb", lambda c: stage_tune(c, "xgb")),
    ("S5_final_train", stage_final),
    ("S6_ensemble", stage_ensemble),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-at", default=None, help="HH:MM, berhenti rapi sebelum jam ini")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="jalankan tahap tertentu saja")
    ap.add_argument("--trials-lgb", type=int, default=100)
    ap.add_argument("--trials-cat", type=int, default=40)
    ap.add_argument("--trials-xgb", type=int, default=60)
    ap.add_argument("--budget-lgb", type=float, default=60, help="menit per run")
    ap.add_argument("--budget-cat", type=float, default=130)
    ap.add_argument("--budget-xgb", type=float, default=50)
    args = ap.parse_args()
    # Konsol/redirect Windows memakai cp1252; paksa UTF-8 agar karakter non-ASCII tidak membuat crash.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    keep_awake(True)
    ctx = Ctx(args)
    ctx.log(f"=== PIPELINE_START smoke={args.smoke} deadline={ctx.deadline} args={vars(args)}")
    for name, fn in STAGES:
        if args.only and not any(name.startswith(o) for o in args.only):
            continue
        if ctx.status.get("stages", {}).get(name) == "done" and name != "S6_ensemble":
            ctx.log(f"{name}: sudah selesai, dilewati")
            continue
        if ctx.seconds_left() < 300:
            ctx.log(f"STOP_AT tercapai sebelum {name}; lanjutkan nanti dengan perintah yang sama")
            break
        ctx.log(f">>> {name}")
        try:
            with Timer() as t:
                ok = fn(ctx)
            state = "done" if ok else "partial"
            ctx.log(f"<<< {name}: {state} ({t.minutes:.1f} mnt)")
        except Exception:
            state = "error"
            ctx.log(f"STAGE_ERROR {name}\n{traceback.format_exc()}")
        ctx.save_status(stages={**ctx.status.get("stages", {}), name: state})
    ctx.log("=== PIPELINE_END")
    keep_awake(False)


if __name__ == "__main__":
    main()
