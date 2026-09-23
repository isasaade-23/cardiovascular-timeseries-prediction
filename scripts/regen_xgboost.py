#!/usr/bin/env python3
"""Prepara a regeneracao da linha do XGBoost, SEM aplicar.

A decisao de trocar o numero publicado nao e deste script. O que ele faz e deixar a troca
pronta e medida, para que a decisao seja tomada olhando o antes e o depois de cada numero
afetado, e nao no escuro.

Duas partes:

1. DETERMINISMO. O XGBoostForecaster usa `n_jobs=-1`, que significa "todos os nucleos". O
   numero de nucleos muda entre maquinas, e a ordem de soma dos gradientes muda com ele, o
   que move o resultado em ate 0,05 pp sozinho (docs/xgboost_reprodutibilidade.md). Aqui
   fixamos n_jobs num valor constante e confirmamos que duas execucoes seguidas dao
   exatamente o mesmo numero. A classe do repositorio NAO e alterada; a fixacao vale so
   para esta rodada, por parametro.

2. REGENERACAO EM DIRETORIO SEPARADO. Com xgboost fixado e n_jobs constante, regenera a
   linha do XGBoost e tudo que depende dela em results/regen_xgb/: previsoes, metricas,
   comparacoes pareadas, Diebold-Mariano e o veredito do criterio. results/ nao e tocado.

Uso:
    PYTHONPATH=src python scripts/regen_xgboost.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DEST = RES / "regen_xgb"
SERIE = RES / "series" / "serie_eventos_sp_sim_real_2010_2023.csv"
BENCH = RES / "benchmark_sim_real_sp_2010_2023_predictions.csv"
BASE = RES / "benchmark_baselines_2010_2023_predictions.csv"

# Escolhido por ser o menor valor que nao depende da maquina. n_jobs=1 e reproduzivel em
# qualquer lugar, inclusive num runner de CI com um nucleo; qualquer valor maior volta a
# depender de quantos nucleos existem.
N_JOBS = 1
B, SEED = 10_000, 20260817
TOP3 = ["prophet", "sarima", "timesfm"]
HORIZONTES = 6


def smape_vec(yt, yp):
    return 200.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp))


def dm_test(d, h):
    n = len(d)
    db = d.mean()
    var = np.sum((d - db) ** 2) / n
    for lag in range(1, h):
        var += 2.0 * np.sum((d[lag:] - db) * (d[:-lag] - db)) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = db / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))


def matriz(df, modelo, nw, nh=HORIZONTES):
    g = df[df.model == modelo].sort_values(["window", "horizon"])
    if len(g) != nw * nh:
        raise ValueError(f"{modelo}: {len(g)} previsoes, esperado {nw * nh}")
    return g.y_true.to_numpy(float).reshape(nw, nh), g.y_pred.to_numpy(float).reshape(nw, nh)


def roda_xgboost(serie, n_jobs):
    """Backtest do XGBoost com n_jobs fixo. Devolve o DataFrame de previsoes."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cv_timeseries.evaluate import rolling_origin_splits
    from cv_timeseries.models import XGBoostForecaster

    modelo = XGBoostForecaster(n_jobs=n_jobs)
    linhas = []
    for wid, (train, test) in enumerate(
        rolling_origin_splits(serie, horizon=HORIZONTES, min_train_size=60), start=1
    ):
        yp = np.asarray(modelo.forecast(train, horizon=len(test)), dtype=float)
        for h, (dt, a, b_) in enumerate(zip(test.index, test.to_numpy(float), yp), start=1):
            linhas.append({"model": "xgboost", "date": dt, "y_true": a, "y_pred": b_,
                           "window": wid, "horizon": h, "train_end": train.index[-1]})
    return pd.DataFrame(linhas)


def main() -> int:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import xgboost
    from cv_timeseries.data import load_and_aggregate_series

    serie = load_and_aggregate_series(str(SERIE), "date", "value", "MS")
    print(f"  xgboost {xgboost.__version__}, n_jobs fixado em {N_JOBS}")

    # ---------------- parte 1: determinismo -------------------------------
    print("\n  Determinismo com n_jobs fixo, duas execucoes seguidas:")
    corridas = []
    for i in (1, 2):
        d = roda_xgboost(serie, N_JOBS)
        s = float(smape_vec(d.y_true.to_numpy(float), d.y_pred.to_numpy(float)).mean())
        corridas.append((d, s))
        print(f"    execucao {i}: sMAPE = {s:.12f}")
    d1, s1 = corridas[0]
    d2, s2 = corridas[1]
    identicas = bool(np.array_equal(d1.y_pred.to_numpy(), d2.y_pred.to_numpy()))
    print(f"    previsao a previsao identica: {identicas}   |sMAPE1-sMAPE2| = {abs(s1-s2):.2e}")
    if not identicas:
        raise SystemExit("n_jobs fixo NAO bastou para tornar deterministico; pare aqui")

    # ---------------- parte 2: regeneracao --------------------------------
    DEST.mkdir(parents=True, exist_ok=True)
    d1.to_csv(DEST / "xgboost_predictions.csv", index=False)

    antigo = pd.read_csv(BENCH, parse_dates=["date"])
    baselines = pd.read_csv(BASE, parse_dates=["date"])
    nw = int(antigo.window.nunique())

    # o novo conjunto: os quatro modelos antigos mais o XGBoost regenerado
    novo = pd.concat([antigo[antigo.model != "xgboost"], d1, baselines], ignore_index=True)
    velho = pd.concat([antigo, baselines], ignore_index=True)

    saida = {"_meta": {"xgboost": xgboost.__version__, "n_jobs": N_JOBS, "B": B,
                       "seed": SEED, "n_janelas": nw,
                       "determinismo": {"identicas": identicas, "smape": s1}},
             "linha1": {}, "pares": {}, "vs_snaive": {}}

    for rotulo, dados in (("antes", velho), ("depois", novo)):
        sm, ae = {}, {}
        for m in TOP3 + ["catboost", "xgboost", "snaive"]:
            yt, yp = matriz(dados, m, nw)
            sm[m] = smape_vec(yt, yp)
            ae[m] = np.abs(yp - yt)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, nw, size=(B, nw))
        boot = {m: sm[m][idx].mean(axis=(1, 2)) for m in sm}

        # Tabela 1: a linha do XGBoost e o que muda; as outras entram como controle
        yt, yp = matriz(dados, "xgboost", nw)
        lo, hi = np.percentile(boot["xgboost"], [2.5, 97.5])
        saida["linha1"][rotulo] = {
            "mae": float(np.abs(yp - yt).mean()),
            "rmse": float(np.sqrt(((yp - yt) ** 2).mean())),
            "smape": float(sm["xgboost"].mean()),
            "ic_low": float(lo), "ic_high": float(hi),
        }
        # pares do top-3, que e o achado central e nao envolve o XGBoost
        saida["pares"][rotulo] = {}
        for a, b_ in combinations(TOP3, 2):
            bd = boot[a] - boot[b_]
            l, h = np.percentile(bd, [2.5, 97.5])
            nsig = sum(dm_test(ae[a][:, k] - ae[b_][:, k], k + 1)[1] < 0.05
                       for k in range(HORIZONTES))
            saida["pares"][rotulo][f"{a}-{b_}"] = {
                "delta": float(sm[a].mean() - sm[b_].mean()),
                "ic_low": float(l), "ic_high": float(h), "dm": int(nsig),
                "distinguiveis": bool(not (l <= 0 <= h) and nsig >= 3)}
        # XGBoost contra o naive sazonal
        bd = boot["xgboost"] - boot["snaive"]
        l, h = np.percentile(bd, [2.5, 97.5])
        nsig = sum(dm_test(ae["xgboost"][:, k] - ae["snaive"][:, k], k + 1)[1] < 0.05
                   for k in range(HORIZONTES))
        saida["vs_snaive"][rotulo] = {
            "delta": float(sm["xgboost"].mean() - sm["snaive"].mean()),
            "ic_low": float(l), "ic_high": float(h), "dm": int(nsig),
            "pior_que_snaive": bool(l > 0 and nsig >= 3)}
        saida.setdefault("ranking", {})[rotulo] = sorted(
            sm, key=lambda m: float(sm[m].mean()))

    (DEST / "antes_depois.json").write_text(json.dumps(saida, indent=2) + "\n",
                                            encoding="utf-8")

    # ---------------- relatorio --------------------------------------------
    a, dp = saida["linha1"]["antes"], saida["linha1"]["depois"]
    print("\n  Linha do XGBoost na Tabela 1")
    print(f"    {'metrica':<10}{'antes':>12}{'depois':>12}{'delta':>10}")
    for k, casas in (("mae", 1), ("rmse", 1), ("smape", 4)):
        print(f"    {k:<10}{a[k]:>12.{casas}f}{dp[k]:>12.{casas}f}{dp[k]-a[k]:>+10.{casas}f}")
    ic_a = f"[{a['ic_low']:.2f}, {a['ic_high']:.2f}]"
    ic_d = f"[{dp['ic_low']:.2f}, {dp['ic_high']:.2f}]"
    print(f"    {'IC sMAPE':<10}{ic_a:>12}{ic_d:>12}")

    print("\n  Pares do top-3 (o achado central; nao envolve XGBoost)")
    for par in saida["pares"]["antes"]:
        x, y = saida["pares"]["antes"][par], saida["pares"]["depois"][par]
        igual = (x["distinguiveis"] == y["distinguiveis"]) and x["dm"] == y["dm"]
        print(f"    {par:<18} antes: DM {x['dm']}/6 {'sep' if x['distinguiveis'] else 'indist'}"
              f"   depois: DM {y['dm']}/6 {'sep' if y['distinguiveis'] else 'indist'}"
              f"   {'igual' if igual else 'MUDOU'}")

    x, y = saida["vs_snaive"]["antes"], saida["vs_snaive"]["depois"]
    print("\n  XGBoost contra o naive sazonal")
    print(f"    antes : delta {x['delta']:+.3f}  IC[{x['ic_low']:+.3f},{x['ic_high']:+.3f}]"
          f"  DM {x['dm']}/6  -> {'pior' if x['pior_que_snaive'] else 'nao estabelecido'}")
    print(f"    depois: delta {y['delta']:+.3f}  IC[{y['ic_low']:+.3f},{y['ic_high']:+.3f}]"
          f"  DM {y['dm']}/6  -> {'pior' if y['pior_que_snaive'] else 'nao estabelecido'}")

    ra, rd = saida["ranking"]["antes"], saida["ranking"]["depois"]
    print(f"\n  Ranking antes : {' < '.join(ra)}")
    print(f"  Ranking depois: {' < '.join(rd)}")
    print(f"  Ranking {'inalterado' if ra == rd else 'MUDOU'}")
    print(f"\n  results/regen_xgb/antes_depois.json e xgboost_predictions.csv")
    print("  results/ NAO foi tocado; src/ NAO foi alterado; manuscrito NAO foi alterado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
