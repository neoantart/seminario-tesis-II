import sys, glob, json, pickle, numpy as np, pandas as pd
from common import *
OUT = sys.argv[1]
U = pickle.load(open(f"{OUT}/umbrales.pkl","rb"))["U"]
B = pd.read_pickle(f"{OUT}/base.pkl").reset_index().rename(columns={"index":"fecha"})
X = pd.read_pickle(f"{OUT}/preds_xgb.pkl").reset_index().rename(columns={"index":"fecha"})
X = X[["fecha","nombre","xgb_y_PT","xgb_y_TM","xgb_p_e_lluvia","xgb_p_e_frio"]]
D = B.merge(X, on=["fecha","nombre"])
# umbral por fila
D["thr_ll"] = D.nombre.map(lambda k: U[k]["p95_lluvia"]); D["thr_fr"] = D.nombre.map(lambda k: U[k]["p5_tm"])
dl = {}
for w in ["lstm","tft"]:
    fs = sorted(glob.glob(f"{OUT}/preds_{w}_s*.pkl")); dl[w] = fs
    for j, f in enumerate(fs):
        p = pd.read_pickle(f)[["fecha","nombre",f"{w}_y_PT",f"{w}_y_TM",f"{w}_p_e_lluvia",f"{w}_p_e_frio"]]
        p.columns = ["fecha","nombre"]+[f"{c}_{j}" for c in p.columns[2:]]
        D = D.merge(p, on=["fecha","nombre"], how="inner")
print("filas", len(D), D.split.value_counts().to_dict())
def split(s): return D[D.split==s]
def eval_reg(col, tgt):
    return {sp: metricas_reg(split(sp)[tgt].values, split(sp)[col].values) for sp in [1,2]}
rows = []; ev_rows = []
def add_reg(modelo, cols_pt, cols_tm):
    for var, cols, tgt in [("PT", cols_pt, "y_PT"), ("TM", cols_tm, "y_TM")]:
        ms = [metricas_reg(split(2)[tgt].values, split(2)[c].values) for c in cols]
        r = dict(Modelo=modelo, Variable=var)
        for k in ["RMSE","MAE","R2"]:
            v = np.array([m[k] for m in ms]); r[k] = v.mean(); r[k+"_sd"] = v.std() if len(v) > 1 else np.nan
        r["n"] = ms[0]["n"]; r["semillas"] = len(cols); rows.append(r)
D["clim_y_PT"] = D.clima_PT; D["clim_y_TM"] = D.clima_TM
D["pers_y_PT"] = D.PT_l0; D["pers_y_TM"] = D.TM_l0
add_reg("Climatología (línea base)", ["clim_y_PT"], ["clim_y_TM"])
add_reg("Persistencia (línea base)", ["pers_y_PT"], ["pers_y_TM"])
add_reg("XGBoost", ["xgb_y_PT"], ["xgb_y_TM"])
for w, nm in [("lstm","LSTM"),("tft","TFT")]:
    n = len(dl[w])
    if n: add_reg(nm, [f"{w}_y_PT_{j}" for j in range(n)], [f"{w}_y_TM_{j}" for j in range(n)])
REG = pd.DataFrame(rows)
# eventos
def add_ev(modelo, ev, scols, sign=1, tune=True):
    ys = [] ; res = []
    for sc in scols:
        va, te = split(1), split(2)
        sv = sign*va[sc].values; st_ = sign*te[sc].values
        t = mejor_umbral(va[ev].values, sv) if tune else 0.5
        m = ~(np.isnan(te[ev].values) | np.isnan(st_))
        r = prf(te[ev].values[m], st_[m] >= t); r["AUC_PR"] = auc_pr(te[ev].values[m], st_[m]); res.append(r)
    out = dict(Modelo=modelo, Evento={"e_lluvia":"Lluvia intensa (≥P95)","e_frio":"Día frío extremo (≤P5)"}[ev],
               prevalencia=float(split(2)[ev].mean()), semillas=len(scols))
    for k in ["Precision","Recall","F1","AUC_PR"]:
        v = np.array([r[k] for r in res]); out[k] = v.mean(); out[k+"_sd"] = v.std() if len(v) > 1 else np.nan
    for k in ["TP","FP","FN"]: out[k] = float(np.mean([r[k] for r in res]))
    ev_rows.append(out)
# persistencia de evento: valor de hoy supera umbral
D["pers_ll"] = (D.PT_l0 >= D.thr_ll).astype(float); D["pers_fr"] = (D.TM_l0 <= D.thr_fr).astype(float)
D["xgbr_ll"] = D.xgb_y_PT; D["xgbr_fr"] = -D.xgb_y_TM
add_ev("Persistencia (línea base)", "e_lluvia", ["pers_ll"], tune=False)
add_ev("Persistencia (línea base)", "e_frio", ["pers_fr"], tune=False)
add_ev("XGBoost (regresión + umbral)", "e_lluvia", ["xgbr_ll"]); add_ev("XGBoost (regresión + umbral)", "e_frio", ["xgbr_fr"])
add_ev("XGBoost (clasificador)", "e_lluvia", ["xgb_p_e_lluvia"]); add_ev("XGBoost (clasificador)", "e_frio", ["xgb_p_e_frio"])
for w, nm in [("lstm","LSTM"),("tft","TFT")]:
    n = len(dl[w])
    if n:
        add_ev(nm, "e_lluvia", [f"{w}_p_e_lluvia_{j}" for j in range(n)]); add_ev(nm, "e_frio", [f"{w}_p_e_frio_{j}" for j in range(n)])
EV = pd.DataFrame(ev_rows)
REG.to_csv(f"{OUT}/resultados_regresion.csv", index=False); EV.to_csv(f"{OUT}/resultados_eventos.csv", index=False)
# por estación (XGB clasificador y mejor DL) y por estación regresión XGB
por = []
for k in MODELADAS:
    t = D[(D.split==2)&(D.nombre==k)]
    r = dict(Estacion=k)
    for var, c, tg in [("PT","xgb_y_PT","y_PT"),("TM","xgb_y_TM","y_TM")]:
        m = metricas_reg(t[tg].values, t[c].values); r[f"RMSE_{var}"] = m["RMSE"]; r[f"R2_{var}"] = m["R2"]
    por.append(r)
pd.DataFrame(por).to_csv(f"{OUT}/resultados_por_estacion_xgb.csv", index=False)
D.to_pickle(f"{OUT}/D_eval.pkl")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(REG.round(3).to_string()); print(EV.round(3).drop(columns=["TP","FP","FN"]).to_string()); print(pd.DataFrame(por).round(3))
