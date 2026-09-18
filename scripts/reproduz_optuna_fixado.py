#!/usr/bin/env python3
"""Reavalia a busca de hiperparametros no ambiente que o repositorio fixa.

A tabela do Optuna nasceu com as tres colunas medidas no ambiente de quem rodou a busca, que
nao e o que `requirements-optional.txt` fixa. Isso produzia duas versoes do mesmo ganho do
XGBoost, 0,21 pp contra a base de la e 0,23 pp contra a base daqui, e foi o que a revisao
marcou.

A busca NAO precisa ser refeita: os hiperparametros vencedores estao nos JSON de
`results/revisao/`. O que este script faz e AVALIA-LOS aqui, para que as quatro colunas da
tabela venham do mesmo lugar e o texto possa citar um numero so.

O CatBoost reproduz os dois valores de origem exatamente. O XGBoost nao, pelo motivo
documentado em docs/xgboost_reprodutibilidade.md, e e justamente por isso que o valor medido
aqui e o que deve ir para a tabela.

Uso:
    PYTHONPATH=src python scripts/reproduz_optuna_fixado.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SERIE = RES / "series" / "serie_eventos_sp_sim_real_2010_2023.csv"

# Hiperparametros vencedores, lidos dos JSON da rodada de revisao. Repetidos aqui em vez de
# carregados para que o script diga, no proprio corpo, o que esta sendo avaliado.
CONFIG = {
    "catboost": {
        "dev": dict(lags=16, iterations=1100, depth=3, learning_rate=0.0068450253,
                    l2_leaf_reg=1.2134146906, random_strength=2.0086316467,
                    bagging_temperature=1.9646333079, verbose=False),
        "oracle": dict(lags=24, iterations=850, depth=6,
                       learning_rate=0.022248131976214240,
                       l2_leaf_reg=22.410817278880092,
                       random_strength=4.596077231954323,
                       bagging_temperature=4.875282126753238, verbose=False),
    },
    "xgboost": {
        "dev": dict(lags=16, n_estimators=900, max_depth=9, learning_rate=0.0310279213,
                    subsample=0.5446910613, colsample_bytree=0.8531596335,
                    min_child_weight=5, reg_lambda=1.3552382246, reg_alpha=0.3223374176),
        "oracle": dict(lags=24, n_estimators=1350, max_depth=8,
                       learning_rate=0.007050194075901503,
                       subsample=0.5004852259032829, colsample_bytree=0.6774005912388466,
                       min_child_weight=8, reg_lambda=22.96597265312771,
                       reg_alpha=0.013708455885086304),
    },
}

# O que cada JSON de origem relata, para a divergencia ficar explicita na saida.
ORIGEM = {
    "catboost": {"base": 6.5893, "dev": 6.4162903089, "oracle": 6.3471},
    "xgboost": {"base": 6.8535, "dev": 7.0648751870, "oracle": 6.2244},
}


def smape(yt, yp):
    return float(np.mean(200.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp))))


def backtest(modelo, serie):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cv_timeseries.evaluate import rolling_origin_splits

    yt, yp = [], []
    for train, test in rolling_origin_splits(serie, horizon=6, min_train_size=60):
        yp.append(np.asarray(modelo.forecast(train, horizon=len(test)), dtype=float))
        yt.append(test.to_numpy(dtype=float))
    return np.concatenate(yt), np.concatenate(yp)


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import catboost
    import xgboost
    from cv_timeseries.data import load_and_aggregate_series
    from cv_timeseries.models import CatBoostForecaster, XGBoostForecaster

    serie = load_and_aggregate_series(str(SERIE), "date", "value", "MS")
    classes = {"catboost": CatBoostForecaster, "xgboost": XGBoostForecaster}
    versoes = {"xgboost": xgboost.__version__, "catboost": catboost.__version__}
    print(f"  ambiente: xgboost {versoes['xgboost']}, catboost {versoes['catboost']}")

    out = {"_meta": {"versoes": versoes, "n_janelas": 103, "horizonte": 6,
                     "min_train": 60,
                     "nota": ("as tres colunas avaliadas no mesmo ambiente; a busca nao foi "
                              "refeita, so os hiperparametros vencedores reavaliados")},
           "modelos": {}}

    for nome, cls in classes.items():
        # sem ajuste: a configuracao padrao de src/cv_timeseries/models.py
        yt, yp = backtest(cls(**({"verbose": False} if nome == "catboost" else {})), serie)
        base = smape(yt, yp)
        linha = {"base": base}
        for modo in ("dev", "oracle"):
            kw = dict(CONFIG[nome][modo])
            lags = kw.pop("lags")
            yt, yp = backtest(cls(lags=lags, **kw), serie)
            linha[modo] = smape(yt, yp)
        linha["ganho_dev"] = base - linha["dev"]
        linha["ganho_oracle"] = base - linha["oracle"]
        linha["divergencia_contra_origem"] = {
            k: abs(linha[k] - ORIGEM[nome][k]) for k in ("base", "dev", "oracle")}
        out["modelos"][nome] = linha

        print(f"  {nome:9s} base={base:.6f}  dev={linha['dev']:.6f}  "
              f"oracle={linha['oracle']:.6f}")
        for k in ("base", "dev", "oracle"):
            d = linha["divergencia_contra_origem"][k]
            print(f"      {k:7s} origem={ORIGEM[nome][k]:.6f}  div={d:.4f}"
                  f"  {'reproduz' if d < 1e-3 else 'DIVERGE (ver docs/xgboost_...)'}")

    dest = RES / "revisao" / "optuna_ambiente_fixado.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n  results/revisao/{dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
