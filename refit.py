"""Latih ulang model tuned Fase 5 dengan batas iterasi / seed berbeda (E140, E144).

Parameter diambil dari study Optuna di experiments/p5/optuna.db (sama dengan tune.py S5), data dan fold
juga sama. Hasil per seed: oof/<nama>_s<seed>.npy dan preds/<nama>_s<seed>.npy (seed yang sudah ada
dilewati, jadi bisa dihentikan dan dilanjutkan). Nama = T_<model>_<fs>_<hash><tag>.

Contoh:
  python refit.py --model lgb --fs b3 selb2 --rounds 15000 --tag _r15k            # E140a
  python refit.py --model lgb --fs b3 --rounds 15000 --tag _r15k --seeds 45 46    # E144
  python refit.py --model cat --fs b3 --rounds 15000 --tag _r15k                  # E140b
  python refit.py ... --submit                                                     # tulis submission bag
Log: experiments/refit.log
"""
import argparse
import sys
from argparse import Namespace
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import N_FOLDS, ROOT
from src.train import Timer, ks_score, log_experiment, write_submission
from src.tuning import FINAL_LR, fit_fold, full_params, params_hash
from tune import Ctx, best_tuned, keep_awake

LOG = ROOT / "experiments" / "refit.log"
MIN_TRIALS = {"cat": 8, "xgb": 12, "lgb": 15}


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgb", "cat", "xgb"], required=True)
    ap.add_argument("--fs", nargs="+", default=["b3"], choices=["b3", "selb2"])
    ap.add_argument("--rounds", type=int, default=15000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--tag", default="_r15k")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    keep_awake(True)
    ctx = Ctx(Namespace(smoke=False, stop_at=None))
    tuned, n = best_tuned(ctx, args.model, MIN_TRIALS[args.model])
    params = full_params(args.model, tuned, FINAL_LR[args.model])
    log(f"=== MULAI {vars(args)} | {n} trial selesai")
    for fs in args.fs:
        name = f"T_{args.model}_{fs}_{params_hash(args.model, tuned)}{args.tag}"
        Xtr, Xte = (ctx.X_tr, ctx.X_te) if fs == "b3" else (ctx.Xb2_tr, ctx.Xb2_te)
        for s in args.seeds:
            path = ROOT / "oof" / f"{name}_s{s}.npy"
            if path.exists():
                log(f"{name} seed {s}: sudah ada, dilewati")
                continue
            oof, pred, aucs, iters = np.zeros(len(ctx.y)), np.zeros(len(Xte)), [], []
            with Timer() as t:
                for k in range(N_FOLDS):
                    tr, va = ctx.fold != k, ctx.fold == k
                    p_va, p_te, it = fit_fold(args.model, params, Xtr[tr], ctx.y[tr], Xtr[va], ctx.y[va],
                                              Xte, s, rounds=args.rounds)
                    oof[va], pred = p_va, pred + p_te / N_FOLDS
                    aucs.append(roc_auc_score(ctx.y[va], p_va))
                    iters.append(it)
            np.save(path, oof)
            np.save(ROOT / "preds" / f"{name}_s{s}.npy", pred)
            log(f"{name} seed {s}: AUC mean-fold {np.mean(aucs):.5f} KS {ks_score(ctx.y, oof):.4f} "
                f"iter {iters} ({t.minutes:.1f} mnt)")
        all_seeds = sorted(int(p.stem.rsplit("_s", 1)[1]) for p in (ROOT / "oof").glob(f"{name}_s*.npy"))
        oof = np.mean([np.load(ROOT / "oof" / f"{name}_s{s}.npy") for s in all_seeds], axis=0)
        pred = np.mean([np.load(ROOT / "preds" / f"{name}_s{s}.npy") for s in all_seeds], axis=0)
        aucs = [roc_auc_score(ctx.y[ctx.fold == k], oof[ctx.fold == k]) for k in range(N_FOLDS)]
        kss = [ks_score(ctx.y[ctx.fold == k], oof[ctx.fold == k]) for k in range(N_FOLDS)]
        log(f"{name} BAG {len(all_seeds)} seed {all_seeds}: AUC mean-fold {np.mean(aucs):.5f} "
            f"KS {ks_score(ctx.y, oof):.4f}")
        sub, md5 = "", ""
        if args.submit:
            path, md5 = write_submission(ctx.test_ids, pred, f"{name}_bag{len(all_seeds)}")
            sub = path.name
            log(f"submission {sub} md5 {md5}")
        log_experiment(name, f"{args.model} tuned, lr {params['learning_rate']}, batas iterasi "
                       f"{args.rounds}, bag {len(all_seeds)} seed {all_seeds}, fitur {fs}",
                       "B3" if fs == "b3" else "sel_b2", Xtr.shape[1], f"{args.model}_tuned_bag",
                       params, aucs, roc_auc_score(ctx.y, oof), 0, "kandidat", ks_score(ctx.y, oof), kss,
                       submission_file=sub, md5_hash=md5)
    log("=== SELESAI")
    keep_awake(False)


if __name__ == "__main__":
    main()
