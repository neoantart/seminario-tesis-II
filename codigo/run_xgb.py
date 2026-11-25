import sys, json, pickle, numpy as np, pandas as pd, xgboost as xgb, time
from common import *
DATA = sys.argv[1]; OUT = sys.argv[2]
S = cargar_todo(DATA); rangos_fisicos(S)
U = umbrales(S); C = clima_doy(S, MODELADAS)
P = panel_supervisado(S, U, MODELADAS)
F = features_tabulares(P, S, C)
F["split"] = split_de(F.index)
feat = [c for c in F.columns if c not in ["y_PT","y_TM","e_lluvia","e_frio","split","nombre","clima_PT","clima_TM"]]
res = {}; preds = F[["nombre","split","y_PT","y_TM","e_lluvia","e_frio"]].copy()
t0 = time.time()
for tgt, ev in [("y_PT","e_lluvia"),("y_TM","e_frio")]:
    m = F[tgt].notna()
    tr = F[m & (F.split==0)]; va = F[m & (F.split==1)]; te = F[m & (F.split==2)]
    reg = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=5, early_stopping_rounds=40, tree_method="hist", n_jobs=2, random_state=42)
    reg.fit(tr[feat], tr[tgt], eval_set=[(va[feat], va[tgt])], verbose=False)
    preds.loc[F.index.isin(F.index) , f"xgb_{tgt}"] = np.nan
    preds[f"xgb_{tgt}"] = reg.predict(F[feat])
    me = F[ev].notna()
    tr = F[me & (F.split==0)]; va = F[me & (F.split==1)]
    spw = float((1-tr[ev].mean())/tr[ev].mean())
    clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=5, scale_pos_weight=min(spw, 8), early_stopping_rounds=40, eval_metric="aucpr",
                            tree_method="hist", n_jobs=2, random_state=42)
    clf.fit(tr[feat], tr[ev], eval_set=[(va[feat], va[ev])], verbose=False)
    preds[f"xgb_p_{ev}"] = clf.predict_proba(F[feat])[:, 1]
    res[tgt] = dict(best_iter_reg=int(reg.best_iteration), best_iter_clf=int(clf.best_iteration))
    imp = pd.Series(reg.feature_importances_, index=feat).sort_values(ascending=False)
    imp.to_csv(f"{OUT}/xgb_importancia_{tgt}.csv")
res["tiempo_s"] = time.time()-t0; res["n_features"] = len(feat); res["xgboost"] = xgb.__version__
json.dump(res, open(f"{OUT}/xgb_info.json","w"), indent=1)
preds.to_pickle(f"{OUT}/preds_xgb.pkl")
pickle.dump(dict(U=U), open(f"{OUT}/umbrales.pkl","wb"))
F[["nombre","split","y_PT","y_TM","e_lluvia","e_frio","clima_PT","clima_TM","PT_l0","TM_l0"]].to_pickle(f"{OUT}/base.pkl")
tabla_cobertura(S).to_csv(f"{OUT}/cobertura.csv", index=False)
print(res)
