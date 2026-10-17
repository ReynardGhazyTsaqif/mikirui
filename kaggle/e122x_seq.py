# %% [markdown]
# # E122x — Model urutan GRU pada cicilan, rekening, dan pinjaman lama (DSC LOGIKA UI 2026)
#
# **Tujuan.** GRU pada urutan order e-commerce (E122) menambah +0,001 AUC di atas semua fitur agregat. Notebook
# ini menerapkan cara yang sama ke tabel lain. Tiap baris tabel menjadi satu langkah waktu, diurutkan dari yang
# paling lama ke yang paling baru:
# - `rp`: cicilan (nominal, rasio bayar, keterlambatan, DPD, posisi cicilan dalam pinjaman, data pinjamannya)
# - `ab`: bulan rekening (saldo, arus masuk/keluar, pendapatan, hari saldo rendah)
# - `pa`: aplikasi pinjaman lama (nominal, keputusan, bunga, ringkasan laporan biro milik aplikasi itu)
# - `all`: satu model dengan empat encoder GRU (rp, ab, pa, dan order e-commerce), digabung sebelum head.
#
# **Keluaran** (`/kaggle/working`), satu set per varian:
# - `e122x_<varian>_v1.npz`: fitur `seqx_<varian>` **nested** per fold (format `e121.py`): baris validasi fold k
#   dan test diprediksi model yang dilatih pada fold ≠ k; baris latih fold j ≠ k diprediksi model yang dilatih
#   pada fold di luar {k, j}.
# - `e122x_<varian>_oof.npy` / `e122x_<varian>_test.npy`: prediksi model tunggal (untuk blend).
#
# **Aturan lomba.** Hanya data panitia (turunan yang diunggah sebagai dataset privat), tanpa model pretrained;
# bobot dilatih dari nol. Notebook ini harus tetap **privat**.
#
# **Cara jalan di Kaggle.** Add Input → dataset privat `dsc-logika-exp` (versi yang berisi `seq_*_v1.parquet`);
# Settings → Accelerator → GPU; Internet tidak perlu; lalu Save Version → Save & Run All.

# %%
import gc
import glob
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SMOKE = os.environ.get("E122X_SMOKE") == "1"  # uji kecil di CPU lokal
SMOKE_N = int(os.environ.get("E122X_N", "3000"))
QUICK = os.environ.get("E122X_QUICK") == "1"  # cek lokal: hanya model luar fold 0
SHUFFLE = os.environ.get("E122X_SHUFFLE") == "1"  # audit kebocoran: label train diacak, keluaran berakhiran _shuffle
VARIANTS = ["rp", "ab", "pa"]  # notebook kedua: ["all"]
N_SEEDS = 3
if os.environ.get("E122X_VARIANTS"):
    VARIANTS = os.environ["E122X_VARIANTS"].split(",")
SEEDS = [42] if SMOKE else list(range(42, 42 + N_SEEDS))
MAX_EPOCHS = int(os.environ.get("E122X_EPOCHS", "2")) if SMOKE else 30
PATIENCE = 4
BATCH = 512
N_FOLDS = 5
H = 64
LR = 2e-3
BUDGET_H = 11.0  # sesi Kaggle maks 12 jam: varian yang mulai setelah 11 jam dilewati
MAX_LEN = {"rp": 120, "ab": 24, "pa": 12, "ec": 96}  # maks di data: 113, 23, 11, 91
TABLES = {v: [v] for v in ("rp", "ab", "pa")} | {"all": ["rp", "ab", "pa", "ec"]}

hits = glob.glob("/kaggle/input/**/apps_meta.parquet", recursive=True)
DATA = os.path.dirname(hits[0]) if hits else os.environ.get("E122X_DATA", "kaggle/dataset")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("E122X_OUT", ".")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
print("data:", DATA, "| output:", OUT, "| device:", DEVICE, "| varian:", VARIANTS, "| torch", torch.__version__)

# %% [markdown]
# ## 1. Data: tensor urutan per tabel
# Tabel `seq_*_v1.parquet` sudah berisi fitur per baris yang dihitung relatif terhadap `decision_datetime`
# aplikasi saat ini, urut kronologis per customer (dibuat oleh `make_seq_tables.py` dari data panitia). Di sini
# fitur distandarisasi (rata-rata/simpangan semua baris, tanpa label) lalu disusun menjadi tensor
# [customer, langkah, fitur], rata kanan sehingga langkah terakhir = baris terbaru. Customer tanpa baris di suatu
# tabel mendapat panjang 0.

# %%
meta = pd.read_parquet(f"{DATA}/apps_meta.parquet")
if SMOKE:
    keep = set(meta[meta.is_test == 0].customer_id.sample(SMOKE_N * 3 // 4, random_state=0)) | set(
        meta[meta.is_test == 1].customer_id.sample(SMOKE_N // 4, random_state=0))
    meta = meta[meta.customer_id.isin(keep)].reset_index(drop=True)
cust = meta.customer_id.values
cidx = pd.Series(np.arange(len(cust)), index=cust)
need = sorted({t for v in VARIANTS for t in TABLES[v]})


def ec_table():
    """Order e-commerce dengan fitur yang sama seperti E122 (kronologis per customer)."""
    od = pd.read_parquet(f"{DATA}/ec_orders_v1.parquet")
    od = od[od.customer_id.isin(cidx.index)]
    od = od.sort_values(["customer_id", "recency_h"], ascending=[True, False], kind="stable").reset_index(drop=True)
    gap = od.groupby("customer_id")["recency_h"].diff().mul(-1)
    num = pd.DataFrame({
        "customer_id": od.customer_id.values,
        "rec_d": np.log1p(od.recency_h / 24), "gap_h": np.log1p(gap.fillna(0).clip(lower=0)),
        "first": gap.isna().astype("float32"), "total": np.log1p(od.total),
        "total_vi": np.log1p(od.total_vi.clip(lower=0) * 100), "n_items": np.log1p(od.n_items),
        "qty": np.log1p(od.qty),
        "hour_s": np.sin(2 * np.pi * od.hour / 24), "hour_c": np.cos(2 * np.pi * od.hour / 24),
        "dow_s": np.sin(2 * np.pi * od.dow / 7), "dow_c": np.cos(2 * np.pi * od.dow / 7),
        "status": od.st.astype("float32"),
    })
    for c in ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]:
        num[c] = od[c].fillna(0).values
    for v in range(5):
        num[f"pm{v}"] = (od.pm == v).astype("float32").values
    return num


SEQ, LEN = {}, {}
for t in need:
    df = ec_table() if t == "ec" else pd.read_parquet(f"{DATA}/seq_{t}_v1.parquet").drop(columns="t_order")
    df = df[df.customer_id.isin(cidx.index)].reset_index(drop=True)  # urutan kronologis dari file dipertahankan
    num = df.drop(columns="customer_id").astype("float32")
    sd = num.std().replace(0, 1)
    num = ((num - num.mean()) / sd).astype("float32")
    row_c = cidx.loc[df.customer_id].values
    pos_from_end = df.groupby("customer_id").cumcount(ascending=False).values
    L = MAX_LEN[t]
    keep_rows = pos_from_end < L  # hanya L baris terbaru (di data semua muat)
    arr = np.zeros((len(cust), L, num.shape[1]), dtype="float32")
    arr[row_c[keep_rows], L - 1 - pos_from_end[keep_rows]] = num.values[keep_rows]
    SEQ[t] = arr
    LEN[t] = np.minimum(np.bincount(row_c, minlength=len(cust)), L).astype("int64")
    print(f"{t}: tensor {arr.shape}, customer dengan data {(LEN[t] > 0).mean():.1%}, "
          f"panjang median (yang ada) {int(np.median(LEN[t][LEN[t] > 0]))}", flush=True)
    del df, num
    gc.collect()

is_tr = meta.is_test.values == 0
y_all = meta.target.fillna(0).astype("int64").values
if SHUFFLE:
    y_all[is_tr] = np.random.default_rng(42).permutation(y_all[is_tr])
fold_all = meta.fold.values
tr_idx = np.where(is_tr)[0]  # urutan = urutan train di apps_meta (sama dengan e121.py)
te_idx = np.where(~is_tr)[0]
fold = fold_all[tr_idx]

# %% [markdown]
# ## 2. Model
# Satu encoder per tabel: proyeksi linear per baris → GRU → gabungan keluaran langkah terakhir + attention pooling
# (hanya atas langkah nyata; customer tanpa data di tabel itu mendapat vektor nol + penanda "tidak ada data").
# Vektor semua tabel digabung → MLP → logit. Early stopping memakai 10% customer holdout di dalam data latih
# model itu sendiri (bukan fold validasi), sama seperti E122.

# %%
class Branch(nn.Module):
    def __init__(self, f, h, p=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f, h), nn.GELU(), nn.Dropout(p))
        self.enc = nn.GRU(h, h, batch_first=True)
        self.att = nn.Linear(h, 1)

    def forward(self, x, lens):
        L = x.shape[1]
        mask = torch.arange(L, device=x.device)[None, :] >= (L - lens)[:, None]  # True = baris nyata
        o = self.enc(self.inp(x))[0]
        has = (lens > 0).float()[:, None]
        a = self.att(o).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(a, dim=1) * mask.float()
        pooled = (w.unsqueeze(-1) * o).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6)
        return torch.cat([o[:, -1] * has, pooled * has, has], 1)


class MultiSeq(nn.Module):
    def __init__(self, tables, h=64, p=0.1):
        super().__init__()
        self.tables = tables
        self.br = nn.ModuleDict({t: Branch(SEQ[t].shape[2], h, p) for t in tables})
        d = len(tables) * (2 * h + 1)
        self.head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, xs, ls):
        return self.head(torch.cat([self.br[t](xs[t], ls[t]) for t in self.tables], 1)).squeeze(-1)


def batch(tables, j):
    return ({t: torch.from_numpy(SEQ[t][j]).to(DEVICE) for t in tables},
            {t: torch.from_numpy(LEN[t][j]).to(DEVICE) for t in tables})


def predict(model, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), 2048):
            out.append(model(*batch(model.tables, idx[b:b + 2048])).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(tables, train_idx, seed):
    """Latih pada indeks global train_idx; early stopping di 10% holdout acak dari train_idx."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_es = len(perm) // 10
    es, tr = perm[:n_es], perm[n_es:]
    model = MultiSeq(tables, H).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, wait = -1, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), BATCH):
            j = np.sort(order[b:b + BATCH])
            xs, ls = batch(tables, j)
            yb = torch.from_numpy(y_all[j].astype("float32")).to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xs, ls), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auc = roc_auc_score(y_all[es], predict(model, es))
        if auc > best + 1e-5:
            best, wait = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1

# %% [markdown]
# ## 3. Pelatihan nested per varian (5 fold luar × (1 + 4) model × seed)
# Hasil tiap varian langsung disimpan, jadi kalau sesi berhenti di tengah, varian yang sudah selesai tetap ada.

# %%
t0 = time.time()
for v in VARIANTS:
    if (time.time() - t0) / 3600 > BUDGET_H:
        print(f"anggaran {BUDGET_H} jam habis, varian {v} dilewati", flush=True)
        continue
    tables = TABLES[v]
    res = {}
    oof = np.zeros(len(tr_idx))
    test_pred = np.zeros(len(te_idx))
    for k in range(N_FOLDS):
        va_k, trn_k = tr_idx[fold == k], tr_idx[fold != k]
        f_va, f_te = np.zeros(len(va_k)), np.zeros(len(te_idx))
        f_tr = pd.Series(0.0, index=trn_k)
        for s in SEEDS:
            m_k, es_auc, ep = fit(tables, trn_k, s + 100 * k)
            f_va += predict(m_k, va_k) / len(SEEDS)
            f_te += predict(m_k, te_idx) / len(SEEDS)
            if QUICK:
                print(f"QUICK [{v}] fold {k}: AUC val {roc_auc_score(y_all[va_k], f_va):.4f}, holdout {es_auc:.4f}, epoch {ep}", flush=True)
                break
            for j in range(N_FOLDS):
                if j == k:
                    continue
                rows_j = tr_idx[fold == j]
                m_j, _, _ = fit(tables, tr_idx[(fold != k) & (fold != j)], s + 100 * k + 10 * j)
                f_tr.loc[rows_j] += predict(m_j, rows_j) / len(SEEDS)
                del m_j
            del m_k
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        if QUICK:
            break
        res[f"tr{k}"] = f_tr.values.reshape(-1, 1).astype("float32")  # urutan = tr_idx[fold != k]
        res[f"va{k}"] = f_va.reshape(-1, 1).astype("float32")
        res[f"te{k}"] = f_te.reshape(-1, 1).astype("float32")
        oof[fold == k] = f_va
        test_pred += f_te / N_FOLDS
        print(f"[{v}] fold {k}: AUC val {roc_auc_score(y_all[va_k], f_va):.4f} | AUC fitur latih "
              f"{roc_auc_score(y_all[trn_k], f_tr.values):.4f} | epoch terakhir {ep} | "
              f"{(time.time() - t0) / 60:.1f} mnt", flush=True)
    if QUICK:
        continue
    aucs = [roc_auc_score(y_all[tr_idx][fold == k], oof[fold == k]) for k in range(N_FOLDS)]
    print(f"== varian {v} ({'+'.join(tables)}): mean-fold AUC {np.mean(aucs):.5f}", np.round(aucs, 5), flush=True)
    res["cols"] = np.array([f"seqx_{v}"])
    res["ids_train"] = meta.application_id.values[tr_idx]
    res["ids_test"] = meta.application_id.values[te_idx]
    tag = f"{v}_shuffle" if SHUFFLE else v
    np.savez(f"{OUT}/e122x_{tag}_v1.npz", **res)
    np.save(f"{OUT}/e122x_{tag}_oof.npy", oof)
    np.save(f"{OUT}/e122x_{tag}_test.npy", test_pred)

print("selesai:", sorted(os.listdir(OUT)))
