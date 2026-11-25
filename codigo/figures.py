import sys, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
DATA, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
S = cargar_todo(DATA); rangos_fisicos(S)
# F1: disponibilidad anual PT/TM/HR
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, v in zip(axs, ["PT","TM","HR"]):
    M = pd.DataFrame({k: S[k][v].notna().groupby(S[k].index.year).mean()*100 for k in ESTACIONES}).T
    im = ax.imshow(M.values, aspect="auto", cmap="Blues", vmin=0, vmax=100, extent=[1963.5, 2026.5, len(M)-0.5, -0.5])
    ax.set_title({"PT":"Precipitación (PT101)","TM":"Temperatura (TM101)","HR":"Humedad rel. (HR101)"}[v], fontsize=9)
    ax.set_xlabel("Año")
axs[0].set_yticks(range(len(ESTACIONES))); axs[0].set_yticklabels([k.title() for k in ESTACIONES])
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02); cb.set_label("% de días con dato")
fig.savefig(f"{OUT}/fig_cobertura.png", bbox_inches="tight"); plt.close()
# F2: ciclo anual Puno
d = S["PUNO"]; m = d.groupby(d.index.month).agg(PT=("PT", lambda x: x.mean()*30.4), TM=("TM","mean"), HR=("HR","mean"))
fig, ax = plt.subplots(figsize=(5.5, 3)); ax.bar(range(1,13), m.PT, color="#9ec5e8", label="Precipitación media mensual (mm)")
ax.set_xticks(range(1,13)); ax.set_xticklabels(list("ENEFMAMJJASOND"[0:0]) or ["E","F","M","A","M","J","J","A","S","O","N","D"][0:12]) if False else ax.set_xticklabels(["E","F","M","A","M","J","J","A","S","O","N","D"])
ax.set_ylabel("mm/mes"); ax2 = ax.twinx(); ax2.plot(range(1,13), m.TM, "o-", color="#c0392b", label="Temp. media (°C)"); ax2.set_ylabel("°C"); ax2.spines["top"].set_visible(False)
ax.set_title("Régimen climático medio, estación Puno (1964–2026)", fontsize=9)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1+h2, l1+l2, fontsize=7, loc="upper center")
fig.savefig(f"{OUT}/fig_ciclo_puno.png", bbox_inches="tight"); plt.close()
# F3: observado vs predicho TM Puno (test, 2 años)
D = pd.read_pickle(f"{OUT}/D_eval.pkl"); p = D[(D.nombre=="PUNO")&(D.split==2)].sort_values("fecha")
p = p[(p.fecha >= "2024-01-01") & (p.fecha <= "2025-12-31")]
fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axs[0].plot(p.fecha, p.y_TM, color="k", lw=0.8, label="Observado"); axs[0].plot(p.fecha, p.xgb_y_TM, color="#d35400", lw=0.8, label="XGBoost (t+1)")
axs[0].set_ylabel("Temp. media (°C)"); axs[0].legend(fontsize=7, ncol=2); axs[0].set_title("Estación Puno, periodo de prueba (2024–2025)", fontsize=9)
axs[1].bar(p.fecha, p.y_PT, color="#2980b9", width=1, label="Observado"); axs[1].plot(p.fecha, p.xgb_y_PT, color="#d35400", lw=0.7, label="XGBoost (t+1)")
axs[1].axhline(pickle.load(open(f"{OUT}/umbrales.pkl","rb"))["U"]["PUNO"]["p95_lluvia"], ls="--", color="gray", lw=0.8); axs[1].set_ylabel("Precip. (mm)"); axs[1].legend(fontsize=7, ncol=2)
fig.savefig(f"{OUT}/fig_obs_pred_puno.png", bbox_inches="tight"); plt.close()
# F4: importancia XGB
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, t, ti in zip(axs, ["y_PT","y_TM"], ["Precipitación (t+1)","Temperatura (t+1)"]):
    imp = pd.read_csv(f"{OUT}/xgb_importancia_{t}.csv", index_col=0).iloc[:10, 0][::-1]
    ax.barh(imp.index, imp.values, color="#34495e"); ax.set_title(ti, fontsize=9); ax.set_xlabel("Importancia (ganancia relativa)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_importancia.png", bbox_inches="tight"); plt.close()
# F5: F1 por modelo y evento
EV = pd.read_csv(f"{OUT}/resultados_eventos.csv"); ev = EV.Evento.unique()
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, e in zip(axs, ev):
    t = EV[EV.Evento==e]; ax.barh(t.Modelo, t.F1, xerr=t.F1_sd.fillna(0), color="#16a085"); ax.set_title(e, fontsize=9); ax.set_xlabel("F1 (periodo de prueba)"); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(f"{OUT}/fig_f1.png", bbox_inches="tight"); plt.close()
# F6: pesos de selección de variables TFT
import glob
W = pd.concat([pd.read_csv(f, index_col=0).iloc[:, 0] for f in sorted(glob.glob(f"{OUT}/tft_pesos_variables_s*.csv"))], axis=1).mean(1)
fig, ax = plt.subplots(figsize=(5, 2.8)); ax.bar(W.index, W.values, color="#8e44ad"); ax.set_ylabel("Peso medio de selección"); ax.set_title("TFT: importancia de variables de entrada", fontsize=9)
fig.savefig(f"{OUT}/fig_tft_pesos.png", bbox_inches="tight"); plt.close()
print("ok")
