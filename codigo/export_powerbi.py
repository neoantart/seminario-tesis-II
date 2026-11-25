"""Exporta CSV para Power BI: observado, predicho y nivel de alerta por estación/día (periodo de prueba)."""
import sys, pickle, numpy as np, pandas as pd
from common import *
OUT = sys.argv[1]
D = pd.read_pickle(f"{OUT}/D_eval.pkl"); U = pickle.load(open(f"{OUT}/umbrales.pkl","rb"))["U"]
def prob_cols(w, ev):
    return [c for c in D.columns if c.startswith(f"{w}_p_{ev}_")]
out = D[["fecha","nombre","split","y_PT","y_TM","e_lluvia","e_frio","xgb_y_PT","xgb_y_TM"]].copy()
sel = {}
for ev in ["e_lluvia","e_frio"]:
    cand = {"XGBoost": D[f"xgb_p_{ev}"]}
    for w, nm in [("lstm","LSTM"),("tft","TFT")]:
        cs = prob_cols(w, ev)
        if cs: cand[nm] = D[cs].mean(1)
    va = D.split == 1
    best = max(cand, key=lambda k: auc_pr(D.loc[va, ev].values, cand[k][va].values))   # selección con validación
    p = cand[best]; t = mejor_umbral(D.loc[va, ev].values, p[va].values)
    out[f"prob_{ev}"] = p; out[f"nivel_{ev}"] = np.where(p >= t, "Alerta", np.where(p >= 0.5*t, "Vigilancia", "Normal"))
    sel[ev] = dict(modelo=best, umbral=float(t))
out["fecha_objetivo"] = out.fecha + pd.Timedelta(days=1)
out = out.rename(columns={"nombre":"estacion","y_PT":"PT_obs_t+1","y_TM":"TM_obs_t+1","xgb_y_PT":"PT_pred_xgb","xgb_y_TM":"TM_pred_xgb",
                          "e_lluvia":"evento_lluvia_obs","e_frio":"evento_frio_obs","prob_e_lluvia":"prob_lluvia_intensa","prob_e_frio":"prob_frio_extremo",
                          "nivel_e_lluvia":"nivel_lluvia","nivel_e_frio":"nivel_frio"})
out["umbral_lluvia_mm"] = out.estacion.map(lambda k: U[k]["p95_lluvia"]); out["umbral_frio_C"] = out.estacion.map(lambda k: U[k]["p5_tm"])
out["periodo"] = out.split.map({0:"entrenamiento",1:"validacion",2:"prueba"}); out = out.drop(columns="split")
out.to_csv(f"{OUT}/powerbi_predicciones_alertas.csv", index=False, encoding="utf-8-sig")
import json; json.dump(sel, open(f"{OUT}/powerbi_modelo_seleccionado.json","w"), indent=1)
# resumen de alertas en prueba
t = out[out.periodo=="prueba"]
for ev, c in [("lluvia","evento_lluvia_obs"),("frio","evento_frio_obs")]:
    g = t.groupby(f"nivel_{ev}").agg(dias=(c,"size"), eventos=(c,"sum")); g["tasa_evento_%"] = 100*g.eventos/g.dias; print(ev, sel["e_lluvia" if ev=="lluvia" else "e_frio"]); print(g.round(2))
    g.to_csv(f"{OUT}/alertas_resumen_{ev}.csv")
