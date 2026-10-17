# %% [markdown]
# # E122 — Model urutan GRU pada riwayat order e-commerce (DSC LOGIKA UI 2026)
#
# **Tujuan.** Fitur `ecm_*` (E121) berasal dari model per order yang tiap ordernya dinilai sendiri lalu
# diagregasi. GRU membaca **seluruh urutan order** seorang customer (kapan, berapa, kategori apa, jarak antar
# order), jadi bisa menangkap pola urutan, misalnya belanja yang tiba-tiba rapat atau membesar menjelang
# pengajuan pinjaman.
#
# **Keluaran** (folder `/kaggle/working`):
# - `e122_gru_v1.npz`: fitur `gru_logit` **nested** per fold, format sama dengan `e121.py`. Untuk fold luar k,
#   baris validasi k dan test diprediksi model yang dilatih pada fold ≠ k. Baris latih fold j ≠ k diprediksi
#   model yang dilatih pada fold di luar {k, j}, sehingga label fold k tidak pernah terlihat.
# - `e122_gru_oof.npy` / `e122_gru_test.npy`: prediksi GRU sebagai model tunggal (untuk blend).
#
# **Aturan lomba.** Hanya data panitia (turunan yang diunggah sebagai dataset privat), tanpa model
# pretrained; bobot GRU dilatih dari nol. Notebook ini harus tetap **privat**.
#
# **Cara jalan di Kaggle.** Add Input → dataset privat `dsc-logika-exp`; Settings → Accelerator → GPU;
# lalu Save Version → Save & Run All. Perkiraan waktu dengan GPU T4: ±1–1,5 jam (3 seed × 25 model).

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

SMOKE = os.environ.get("E122_SMOKE") == "1"  # uji kecil di CPU lokal
SHUFFLE = os.environ.get("E122_SHUFFLE") == "1"  # audit kebocoran: label diacak
QUICK = os.environ.get("E122_QUICK") == "1"  # cek lokal: hanya model luar fold 0
SMOKE_N = int(os.environ.get("E122_N", "4000"))
# Varian arsitektur (satu notebook per varian; nama file keluaran ikut nama varian):
#   "gru"   : GRU 1 lapis 64 unit (dasar, ±1–1,5 jam di T4)
#   "bigru" : GRU 2 lapis dua arah 128 unit (berat, ±3–4 jam)
#   "tf"    : Transformer encoder 3 lapis d=128, 4 head (berat, ±3–5 jam)
VARIANT = "gru"
N_SEEDS = 3
VARIANT = os.environ.get("E122_VARIANT", VARIANT)  # hanya untuk uji lokal
ARCH = {"gru": dict(kind="gru", h=64, layers=1, bi=False, lr=2e-3),
        "bigru": dict(kind="gru", h=128, layers=2, bi=True, lr=1e-3),
        "tf": dict(kind="tf", h=128, layers=3, bi=False, lr=5e-4)}[VARIANT]
SEEDS = [42] if SMOKE else list(range(42, 42 + N_SEEDS))
MAX_LEN = 96  # maks order per customer di data = 91
MAX_EPOCHS = int(os.environ.get("E122_EPOCHS", "2")) if SMOKE else 30
PATIENCE = 4
BATCH = 512
N_FOLDS = 5

hits = glob.glob("/kaggle/input/**/apps_meta.parquet", recursive=True)
DATA = os.path.dirname(hits[0]) if hits else os.environ.get("E122_DATA", "kaggle/dataset")
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("E122_OUT", ".")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
print("data:", DATA, "| output:", OUT, "| device:", DEVICE, "| torch", torch.__version__)

# %% [markdown]
# ## 1. Data: urutan order per customer
# Tiap order = 1 langkah waktu, diurutkan dari yang **paling lama** ke yang **paling baru** (langkah terakhir =
# order terdekat dengan keputusan pinjaman). Fitur per order hanya milik order itu sendiri, ditambah jarak
# waktu ke order sebelumnya. Semua waktu relatif ke `decision_datetime` (tidak ada informasi masa depan).

# %%
meta = pd.read_parquet(f"{DATA}/apps_meta.parquet")
od = pd.read_parquet(f"{DATA}/ec_orders_v1.parquet")
if SMOKE:
    keep = set(meta[meta.is_test == 0].customer_id.sample(SMOKE_N * 3 // 4, random_state=0)) | set(
        meta[meta.is_test == 1].customer_id.sample(SMOKE_N // 4, random_state=0))
    meta = meta[meta.customer_id.isin(keep)].reset_index(drop=True)
    od = od[od.customer_id.isin(keep)].reset_index(drop=True)

CATS = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
# od: urut per customer, recency_h naik (terbaru dulu) -> balik jadi kronologis
od = od.sort_values(["customer_id", "recency_h"], ascending=[True, False], kind="stable").reset_index(drop=True)
gap = od.groupby("customer_id")["recency_h"].diff().mul(-1)  # jam sejak order sebelumnya (NaN untuk order pertama)
num = pd.DataFrame({
    "rec_d": np.log1p(od.recency_h / 24),
    "gap_h": np.log1p(gap.fillna(0).clip(lower=0)),
    "first": gap.isna().astype("float32"),
    "total": np.log1p(od.total),
    "total_vi": np.log1p(od.total_vi.clip(lower=0) * 100),
    "n_items": np.log1p(od.n_items),
    "qty": np.log1p(od.qty),
    "hour_s": np.sin(2 * np.pi * od.hour / 24), "hour_c": np.cos(2 * np.pi * od.hour / 24),
    "dow_s": np.sin(2 * np.pi * od.dow / 7), "dow_c": np.cos(2 * np.pi * od.dow / 7),
    "status": od.st.astype("float32"),  # kode status order (2 nilai: batal / selesai)
})
for c in CATS:
    num[c] = od[c].fillna(0)
for v in range(5):
    num[f"pm{v}"] = (od.pm == v).astype("float32")
cont = ["rec_d", "gap_h", "total", "total_vi", "n_items", "qty"]
num[cont] = (num[cont] - num[cont].mean()) / num[cont].std()  # standarisasi tanpa label
F = num.shape[1]

# tensor [customer, MAX_LEN, F], rata kanan (order terbaru di posisi terakhir)
cust = meta.customer_id.values
cidx = pd.Series(np.arange(len(cust)), index=cust)
row_c = cidx.loc[od.customer_id].values
pos_from_end = od.groupby("customer_id").cumcount(ascending=False).values
assert pos_from_end.max() < MAX_LEN
SEQ = np.zeros((len(cust), MAX_LEN, F), dtype="float32")
SEQ[row_c, MAX_LEN - 1 - pos_from_end] = num.values.astype("float32")
LEN = np.bincount(row_c, minlength=len(cust)).astype("int64")
del od, num, gap
gc.collect()
print("SEQ", SEQ.shape, "panjang min/median/max", LEN.min(), int(np.median(LEN)), LEN.max())

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
# Proyeksi linear per order → encoder urutan (GRU atau Transformer, sesuai `VARIANT`) → gabungan
# keluaran langkah terakhir + attention pooling → MLP → logit. Transformer tidak memakai positional
# encoding karena posisi waktu sudah ada di fitur (`rec_d` = umur order, `gap_h` = jarak antar order). Early stopping memakai 10% customer holdout di dalam data latih model itu sendiri
# (bukan fold validasi), sama seperti model order E121.

# %%
class SeqNet(nn.Module):
    def __init__(self, f, kind="gru", h=64, layers=1, bi=False, p=0.1, **_):
        super().__init__()
        self.kind = kind
        self.inp = nn.Sequential(nn.Linear(f, h), nn.GELU(), nn.Dropout(p))
        if kind == "gru":
            self.enc = nn.GRU(h, h, num_layers=layers, batch_first=True, bidirectional=bi,
                              dropout=p if layers > 1 else 0.0)
            d = h * (2 if bi else 1)
        else:
            layer = nn.TransformerEncoderLayer(h, 4, 2 * h, p, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            d = h
        self.att = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.Linear(2 * d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x, lens):
        L = x.shape[1]
        mask = torch.arange(L, device=x.device)[None, :] >= (L - lens)[:, None]  # True = order nyata
        z = self.inp(x)
        o = self.enc(z)[0] if self.kind == "gru" else self.enc(z, src_key_padding_mask=~mask)
        last = o[:, -1]
        a = self.att(o).squeeze(-1).masked_fill(~mask, -1e4)
        pooled = (torch.softmax(a, dim=1).unsqueeze(-1) * o).sum(1)
        return self.head(torch.cat([last, pooled], 1)).squeeze(-1)


def predict(model, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(idx), 2048):
            j = idx[b:b + 2048]
            out.append(model(torch.from_numpy(SEQ[j]).to(DEVICE), torch.from_numpy(LEN[j]).to(DEVICE)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def fit(train_idx, seed):
    """Latih pada indeks global train_idx; early stopping di 10% holdout acak dari train_idx."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_es = len(perm) // 10
    es, tr = perm[:n_es], perm[n_es:]
    model = SeqNet(F, **ARCH).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=ARCH["lr"], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, wait = -1, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), BATCH):
            j = np.sort(order[b:b + BATCH])
            xb = torch.from_numpy(SEQ[j]).to(DEVICE)
            lb = torch.from_numpy(LEN[j]).to(DEVICE)
            yb = torch.from_numpy(y_all[j].astype("float32")).to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb, lb), yb)
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
# ## 3. Pelatihan nested (5 fold luar × (1 + 4) model × seed)

# %%
cols = [f"seq_{VARIANT}"]
res = {}
oof = np.zeros(len(tr_idx))
test_pred = np.zeros(len(te_idx))
t0 = time.time()
for k in range(N_FOLDS):
    va_k, trn_k = tr_idx[fold == k], tr_idx[fold != k]
    f_va, f_te = np.zeros(len(va_k)), np.zeros(len(te_idx))
    f_tr = pd.Series(0.0, index=trn_k)
    for s in SEEDS:
        m_k, es_auc, ep = fit(trn_k, s + 100 * k)
        f_va += predict(m_k, va_k) / len(SEEDS)
        f_te += predict(m_k, te_idx) / len(SEEDS)
        if QUICK:
            print(f"QUICK fold {k}: AUC val {roc_auc_score(y_all[va_k], f_va):.4f}, holdout {es_auc:.4f}, epoch {ep}")
            raise SystemExit
        for j in range(N_FOLDS):
            if j == k:
                continue
            rows_j = tr_idx[fold == j]
            m_j, _, _ = fit(tr_idx[(fold != k) & (fold != j)], s + 100 * k + 10 * j)
            f_tr.loc[rows_j] += predict(m_j, rows_j) / len(SEEDS)
            del m_j
        del m_k
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    res[f"tr{k}"] = f_tr.values.reshape(-1, 1).astype("float32")  # urutan = tr_idx[fold != k]
    res[f"va{k}"] = f_va.reshape(-1, 1).astype("float32")
    res[f"te{k}"] = f_te.reshape(-1, 1).astype("float32")
    oof[fold == k] = f_va
    test_pred += f_te / N_FOLDS
    print(f"fold {k}: AUC GRU val {roc_auc_score(y_all[va_k], f_va):.4f} | AUC fitur latih "
          f"{roc_auc_score(y_all[trn_k], f_tr.values):.4f} | {(time.time() - t0) / 60:.1f} mnt", flush=True)

aucs = [roc_auc_score(y_all[tr_idx][fold == k], oof[fold == k]) for k in range(N_FOLDS)]
print(f"Model urutan {VARIANT} tunggal: mean-fold AUC {np.mean(aucs):.5f}", np.round(aucs, 5))

# %% [markdown]
# ## 4. Simpan
# `application_id` ikut disimpan supaya urutan baris bisa dicek ulang saat digabung di pipeline lokal.

# %%
tag = ("" if VARIANT == "gru" else f"_{VARIANT}") + ("_shuffle" if SHUFFLE else "")
res["cols"] = np.array(cols)
res["ids_train"] = meta.application_id.values[tr_idx]
res["ids_test"] = meta.application_id.values[te_idx]
np.savez(f"{OUT}/e122_gru_v1{tag}.npz", **res)
np.save(f"{OUT}/e122_gru_oof{tag}.npy", oof)
np.save(f"{OUT}/e122_gru_test{tag}.npy", test_pred)
print("selesai:", sorted(os.listdir(OUT)))
