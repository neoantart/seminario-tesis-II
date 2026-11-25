"""LSTM y TFT simplificado (PyTorch) multitarea: PT(t+1), TM(t+1), evento lluvia, evento frío."""
import sys, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from common import *
torch.set_num_threads(2)

def construir(S, U, C):
    P = panel_supervisado(S, U, MODELADAS)
    # normalización con entrenamiento
    st = {}
    X, Y, E, ST, ID, DT, SP = [], [], [], [], [], [], []
    for si, k in enumerate(MODELADAS):
        d = P[k]; tr = d.loc[:TRAIN_END]
        mu = dict(PT=np.log1p(tr.PT.dropna()).mean(), TM=tr.TM.mean(), HR=tr.HR.mean())
        sd = dict(PT=np.log1p(tr.PT.dropna()).std(), TM=tr.TM.std(), HR=tr.HR.std())
        pt = (np.log1p(d.PT) - mu["PT"])/sd["PT"]; tm = (d.TM-mu["TM"])/sd["TM"]; hr = (d.HR-mu["HR"])/sd["HR"]
        doy = d.index.dayofyear.values
        arr = np.stack([pt, tm, hr, pt.isna(), tm.isna(), hr.isna(),
                        np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)], 1).astype("float32")
        arr[:, :3] = np.nan_to_num(arr[:, :3], nan=0.0)
        n = len(d)
        idx = np.arange(WIN-1, n)
        win = np.stack([arr[i-WIN+1:i+1] for i in idx])
        y = d[["y_PT","y_TM"]].values[idx]; e = d[["e_lluvia","e_frio"]].values[idx]
        ok = ~np.isnan(y).all(1)
        X.append(win[ok]); Y.append(y[ok]); E.append(e[ok]); ST.append(np.full(ok.sum(), si)); DT.append(d.index[idx][ok])
        st[k] = dict(mu=mu, sd=sd)
    X = np.concatenate(X); Y = np.concatenate(Y); E = np.concatenate(E); ST = np.concatenate(ST); DT = np.concatenate(DT)
    SP = split_de(pd.DatetimeIndex(DT))
    return X, Y, E, ST, DT, SP, st

class LSTMNet(nn.Module):
    def __init__(s, nin=8, h=64, ns=7):
        super().__init__()
        s.emb = nn.Embedding(ns, 8); s.lstm = nn.LSTM(nin, h, batch_first=True, num_layers=1)
        s.head = nn.Sequential(nn.Linear(h+8, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 4))
    def forward(s, x, st):
        o, (hn, _) = s.lstm(x); return s.head(torch.cat([hn[-1], s.emb(st)], 1))

class GLU_GRN(nn.Module):
    """Gated Residual Network (Lim et al., 2021)."""
    def __init__(s, d, dout=None, dctx=None, drop=0.1):
        super().__init__(); dout = dout or d
        s.l1 = nn.Linear(d, d); s.ctx = nn.Linear(dctx, d, bias=False) if dctx else None
        s.l2 = nn.Linear(d, dout*2); s.drop = nn.Dropout(drop); s.norm = nn.LayerNorm(dout)
        s.skip = nn.Linear(d, dout) if d != dout else nn.Identity(); s.dout = dout
    def forward(s, x, c=None):
        h = s.l1(x) + (s.ctx(c) if (c is not None and s.ctx) else 0)
        h = s.l2(s.drop(nn.functional.elu(h))); a, b = h.chunk(2, -1)
        return s.norm(s.skip(x) + a*torch.sigmoid(b))

class TFT(nn.Module):
    """TFT simplificado: selección de variables (GRN+softmax), codificación estática por estación,
    LSTM codificador, auto-atención multi-cabeza interpretable y compuertas residuales."""
    def __init__(s, nvar=6, dm=48, ns=7, heads=4):
        super().__init__()
        s.nvar = nvar; s.dm = dm
        s.emb = nn.Embedding(ns, dm); s.static_ctx = GLU_GRN(dm)
        s.var_proj = nn.ModuleList([nn.Linear(1, dm) for _ in range(nvar)])
        s.var_grn = nn.ModuleList([GLU_GRN(dm) for _ in range(nvar)])
        s.sel = GLU_GRN(nvar*dm, nvar, dctx=dm)
        s.lstm = nn.LSTM(dm, dm, batch_first=True)
        s.gate1 = nn.Linear(dm, dm*2); s.n1 = nn.LayerNorm(dm)
        s.enh = GLU_GRN(dm, dctx=dm)
        s.att = nn.MultiheadAttention(dm, heads, batch_first=True)
        s.gate2 = nn.Linear(dm, dm*2); s.n2 = nn.LayerNorm(dm)
        s.pos = GLU_GRN(dm); s.out = nn.Linear(dm, 4)
        s.last_weights = None
    def forward(s, x, st):
        c = s.static_ctx(s.emb(st))                                  # contexto estático
        # variables de entrada continuas: PT, TM, HR, sin, cos, (3 máscaras compactadas en 1 promedio)
        v = torch.cat([x[:, :, :3], x[:, :, 6:8], x[:, :, 3:6].mean(2, keepdim=True)], 2)
        tok = torch.stack([g(p(v[:, :, i:i+1])) for i, (p, g) in enumerate(zip(s.var_proj, s.var_grn))], 2)  # B,T,V,dm
        w = torch.softmax(s.sel(tok.flatten(2), c[:, None, :].expand(-1, x.size(1), -1)), -1)   # B,T,V
        s.last_weights = w.detach()
        z = (tok*w.unsqueeze(-1)).sum(2)
        o, _ = s.lstm(z)
        a, b = s.gate1(o).chunk(2, -1); o = s.n1(z + a*torch.sigmoid(b))
        o = s.enh(o, c[:, None, :].expand(-1, o.size(1), -1))
        T = o.size(1); mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        at, _ = s.att(o, o, o, attn_mask=mask)
        a, b = s.gate2(at).chunk(2, -1); o2 = s.n2(o + a*torch.sigmoid(b))
        return s.out(s.pos(o2)[:, -1])

def entrenar(model_fn, X, Y, E, ST, SP, ym, ys, seed=42, epochs=14, bs=512, lr=2e-3, pw=(4.0, 4.0), log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    m = model_fn(); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xt = torch.tensor(X); St = torch.tensor(ST)
    Yn = torch.tensor(((Y-ym)/ys).astype("float32")); En = torch.tensor(np.nan_to_num(E).astype("float32"))
    Mr = torch.tensor(~np.isnan(Y)); Me = torch.tensor(~np.isnan(E))
    tri = np.where(SP==0)[0]; vai = np.where(SP==1)[0]
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(list(pw)))
    def loss_fn(o, i):
        lr_ = ((o[:, :2]-torch.nan_to_num(Yn[i]))**2*Mr[i]).sum()/Mr[i].sum()
        le = (bce(o[:, 2:], En[i])*Me[i]).sum()/Me[i].sum()
        return lr_ + 0.5*le
    best, bstate, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train(); np.random.shuffle(tri); t0 = time.time(); tl = 0
        for j in range(0, len(tri), bs):
            i = torch.tensor(tri[j:j+bs]); opt.zero_grad(); l = loss_fn(m(Xt[i], St[i]), i); l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tl += l.item()*len(i)
        m.eval(); vl = 0
        with torch.no_grad():
            for j in range(0, len(vai), 2048):
                i = torch.tensor(vai[j:j+2048]); vl += loss_fn(m(Xt[i], St[i]), i).item()*len(i)
        vl /= len(vai); sched.step(vl)
        msg = f"ep{ep+1} train {tl/len(tri):.4f} val {vl:.4f} ({time.time()-t0:.0f}s)"; print(msg, flush=True)
        if log is not None: log.append(msg)
        if vl < best - 1e-4: best, bstate, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4: break
    m.load_state_dict(bstate); m.eval(); return m

def predecir(m, X, ST, ym, ys):
    out = []
    with torch.no_grad():
        for j in range(0, len(X), 4096):
            out.append(m(torch.tensor(X[j:j+4096]), torch.tensor(ST[j:j+4096])).numpy())
    o = np.concatenate(out)
    return o[:, :2]*ys+ym, 1/(1+np.exp(-o[:, 2:]))

if __name__ == "__main__":
    DATA, OUT, which = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [42]
    S = cargar_todo(DATA); rangos_fisicos(S); U = umbrales(S); C = clima_doy(S, MODELADAS)
    X, Y, E, ST, DT, SP, st = construir(S, U, C)
    print(X.shape, np.bincount(SP))
    ym = np.nanmean(Y[SP==0], 0); ys = np.nanstd(Y[SP==0], 0)
    for seed in seeds:
        log = []; t0 = time.time()
        fn = (lambda: LSTMNet()) if which == "lstm" else (lambda: TFT())
        m = entrenar(fn, X, Y, E, ST, SP, ym, ys, seed=seed, log=log)
        reg, pe = predecir(m, X, ST, ym, ys)
        df = pd.DataFrame({"fecha": DT, "estacion_id": ST, "split": SP, "y_PT": Y[:, 0], "y_TM": Y[:, 1],
                           "e_lluvia": E[:, 0], "e_frio": E[:, 1], f"{which}_y_PT": reg[:, 0], f"{which}_y_TM": reg[:, 1],
                           f"{which}_p_e_lluvia": pe[:, 0], f"{which}_p_e_frio": pe[:, 1]})
        df["nombre"] = [MODELADAS[i] for i in ST]
        df.to_pickle(f"{OUT}/preds_{which}_s{seed}.pkl")
        info = dict(seed=seed, tiempo_s=time.time()-t0, params=sum(p.numel() for p in m.parameters()), log=log, torch=torch.__version__)
        json.dump(info, open(f"{OUT}/info_{which}_s{seed}.json", "w"), indent=1)
        if which == "tft":
            vi = np.where(SP==2)[0][:5000]
            with torch.no_grad(): m(torch.tensor(X[vi]), torch.tensor(ST[vi]))
            w = m.last_weights.mean((0, 1)).numpy()
            pd.Series(w, index=["PT","TM","HR","sin_doy","cos_doy","faltantes"]).to_csv(f"{OUT}/tft_pesos_variables_s{seed}.csv")
