"""Funciones comunes: carga, limpieza, eventos, ventanas, métricas."""
import glob, os, numpy as np, pandas as pd

ESTACIONES = ["PUNO","JULI","ILAVE","AYAVIRI","MAZOCRUZ","PAMPAHUTA","CHUQUIBAMBILLA","MACUSANI"]
MODELADAS  = ["PUNO","JULI","ILAVE","AYAVIRI","MAZOCRUZ","PAMPAHUTA","CHUQUIBAMBILLA"]  # Macusani excluida (cobertura TM 16 %)
TRAIN_END, VAL_END = "2010-12-31", "2017-12-31"
WIN = 30

def cargar_estacion(path):
    d = pd.read_excel(path, header=1)
    d.columns = [str(c).strip() for c in d.columns]
    fecha = pd.to_datetime(dict(year=d["Año"], month=d["Mes"], day=d["V Dia"]), errors="coerce")
    out = pd.DataFrame({"fecha": fecha, "HR": d["HR101"], "PT": d["PT101"], "TM": d["TM101"]}).dropna(subset=["fecha"])
    for c in ["HR","PT","TM"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["PT_888"] = (out["PT"] == -888)
    for c in ["HR","PT","TM"]:
        out.loc[out[c] == -999, c] = np.nan              # -999 = sin dato (SENAMHI)
    out.loc[out["PT"] == -888, "PT"] = 0.0               # -888 = traza (precipitación inapreciable) -> 0 mm
    for c in ["HR","TM"]:
        out.loc[out[c] == -888, c] = np.nan
    out = out.drop_duplicates("fecha").set_index("fecha").sort_index()
    return out

def cargar_todo(carpeta):
    S = {}
    for f in glob.glob(os.path.join(carpeta, "*.xlsx")):
        name = os.path.splitext(os.path.basename(f))[0].upper()
        if name in ESTACIONES:
            d = cargar_estacion(f)
            idx = pd.date_range("1964-01-01", "2026-08-31", freq="D")
            S[name] = d.reindex(idx)
            S[name]["PT_888"] = S[name]["PT_888"].fillna(False).astype(bool)
    return S

def rangos_fisicos(S):
    """Controles de calidad: valores fuera de rango físico -> NaN. Devuelve conteo."""
    rep = []
    for k, d in S.items():
        n0 = d[["HR","PT","TM"]].notna().sum()
        d.loc[(d.PT < 0) | (d.PT > 300), "PT"] = np.nan
        d.loc[(d.HR < 0) | (d.HR > 100), "HR"] = np.nan
        d.loc[(d.TM < -30) | (d.TM > 40), "TM"] = np.nan
        n1 = d[["HR","PT","TM"]].notna().sum()
        rep.append((k, int((n0-n1).sum())))
    return rep

def tabla_cobertura(S):
    rows = []
    for k, d in S.items():
        r = dict(Estacion=k, Inicio=d.index.min().date(), Fin=d.index.max().date(), Dias=len(d))
        for c in ["PT","TM","HR"]:
            r[f"%{c}"] = round(100*d[c].notna().mean(), 1)
        r["PT_-888"] = int(d.PT_888.sum())
        rows.append(r)
    return pd.DataFrame(rows)

def umbrales(S):
    """Umbrales de evento por estación con el periodo de entrenamiento únicamente."""
    U = {}
    for k, d in S.items():
        tr = d.loc[:TRAIN_END]
        pt = tr.PT.dropna()
        U[k] = dict(p95_lluvia=float(pt[pt >= 1.0].quantile(0.95)), p5_tm=float(tr.TM.dropna().quantile(0.05)))
    return U

def panel_supervisado(S, U, estaciones, h=1):
    """Devuelve dict con arrays por estación listos para ventanas deslizantes."""
    P = {}
    for k in estaciones:
        d = S[k].copy()
        d["y_PT"] = d.PT.shift(-h); d["y_TM"] = d.TM.shift(-h)
        d["e_lluvia"] = (d.y_PT >= U[k]["p95_lluvia"]).astype(float).where(d.y_PT.notna())
        d["e_frio"] = (d.y_TM <= U[k]["p5_tm"]).astype(float).where(d.y_TM.notna())
        P[k] = d
    return P

def clima_doy(S, estaciones):
    """Climatología por día del año (media suavizada) usando solo entrenamiento."""
    C = {}
    for k in estaciones:
        tr = S[k].loc[:TRAIN_END]
        g = tr[["PT","TM","HR"]].groupby(tr.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
        g = pd.concat([g.iloc[-15:], g, g.iloc[:15]]).rolling(31, center=True).mean().iloc[15:-15]
        g.index = range(1, 367); C[k] = g
    return C

def features_tabulares(P, S, C):
    """Rezagos y estadísticos móviles para XGBoost. Retorna DataFrame concatenado."""
    frames = []
    for si, (k, d) in enumerate(P.items()):
        f = pd.DataFrame(index=d.index)
        for v in ["PT","TM","HR"]:
            for L in [0,1,2,3,4,5,6]:
                f[f"{v}_l{L}"] = d[v].shift(L)
        f["PT_s3"]  = d.PT.rolling(3, min_periods=2).sum()
        f["PT_s7"]  = d.PT.rolling(7, min_periods=4).sum()
        f["PT_s15"] = d.PT.rolling(15, min_periods=8).sum()
        f["PT_s30"] = d.PT.rolling(30, min_periods=15).sum()
        f["TM_m7"]  = d.TM.rolling(7, min_periods=4).mean()
        f["TM_m30"] = d.TM.rolling(30, min_periods=15).mean()
        f["HR_m7"]  = d.HR.rolling(7, min_periods=4).mean()
        f["HR_m30"] = d.HR.rolling(30, min_periods=15).mean()
        f["TM_s7"]  = d.TM.rolling(7, min_periods=4).std()
        wet = (d.PT >= 1.0).astype(float).where(d.PT.notna())
        f["dias_hum30"] = wet.rolling(30, min_periods=15).sum()
        # días desde la última lluvia >= 1 mm (tope 60)
        last = pd.Series(np.where(wet == 1, np.arange(len(d)), np.nan), index=d.index).ffill()
        f["dias_sin_lluvia"] = (np.arange(len(d)) - last).clip(upper=60)
        doy = d.index.dayofyear.values
        f["sin_doy"] = np.sin(2*np.pi*(doy+1)/365.25); f["cos_doy"] = np.cos(2*np.pi*(doy+1)/365.25)
        f["TM_anom"] = d.TM - C[k]["TM"].reindex(doy).values
        f["estacion"] = si
        for t in ["y_PT","y_TM","e_lluvia","e_frio"]:
            f[t] = d[t]
        f["clima_PT"] = C[k]["PT"].reindex(((doy) % 366) + 1).values   # climatología del día objetivo (aprox.)
        f["clima_TM"] = C[k]["TM"].reindex(((doy) % 366) + 1).values
        f["nombre"] = k
        frames.append(f)
    return pd.concat(frames)

def split_de(idx):
    s = np.where(idx <= pd.Timestamp(TRAIN_END), 0, np.where(idx <= pd.Timestamp(VAL_END), 1, 2))
    return s

def metricas_reg(y, p):
    m = ~(np.isnan(y) | np.isnan(p)); y, p = y[m], p[m]
    rmse = float(np.sqrt(np.mean((y-p)**2))); mae = float(np.mean(np.abs(y-p)))
    r2 = float(1 - np.sum((y-p)**2)/np.sum((y-y.mean())**2))
    return dict(RMSE=rmse, MAE=mae, R2=r2, n=int(m.sum()))

def prf(y, pred):
    y = y.astype(int); pred = pred.astype(int)
    tp = int(((y==1)&(pred==1)).sum()); fp = int(((y==0)&(pred==1)).sum()); fn = int(((y==1)&(pred==0)).sum())
    pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
    return dict(Precision=pr, Recall=rc, F1=f1, TP=tp, FP=fp, FN=fn)

def mejor_umbral(y, score, grid=None):
    """Umbral de decisión que maximiza F1 en validación."""
    m = ~(np.isnan(y) | np.isnan(score)); y, s = y[m], score[m]
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 200))) if grid is None else grid
    best, bt = -1, grid[0]
    for t in grid:
        f = prf(y, s >= t)["F1"]
        if f > best: best, bt = f, t
    return float(bt)

def auc_pr(y, s):
    m = ~(np.isnan(y) | np.isnan(s)); y, s = y[m], s[m]
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1-y)
    prec = tp/np.maximum(tp+fp, 1); rec = tp/max(y.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], rec]))*prec))
