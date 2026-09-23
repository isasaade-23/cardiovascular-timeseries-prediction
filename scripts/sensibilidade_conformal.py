#!/usr/bin/env python3
"""Sensibilidade: predicao conformal corrige a subcobertura dos intervalos nativos?

SENSIBILIDADE, NAO RESULTADO. Nada daqui vai ao manuscrito antes de ser olhado.

O problema medido em Table 6: a 95% nominal, o SARIMA cobre 84,5% e o Prophet 77,2%. Os
dois intervalos vem do proprio modelo e sao otimistas. A predicao conformal nao pede nada
ao modelo: pega o erro que AQUELE modelo cometeu, NAQUELE passo de horizonte, em janelas
ANTERIORES, e usa o quantil empirico desse erro como intervalo. Sob permutabilidade a
cobertura e garantida por construcao.

Metodo emprestado de C:/Users/Fabiano/Projetos/dengue-time-series/
src/dengue_forecast/intervals.py, de onde vem o quantil conformal de amostra finita. Duas
adaptacoes, as duas com motivo:

1. HORIZONTE 6, nao 12. Aquele modulo fixa 12 no corpo.

2. ESCALA ABSOLUTA, nao log1p. La a escolha existe porque a contagem de dengue varia
   quatro ordens de grandeza entre vale e pico, e o residuo aditivo fica fortemente
   heterocedastico. Aqui a serie vai de cerca de 5.800 a 9.900 obitos por mes, menos de um
   fator dois, entao o residuo aditivo ja e aproximadamente estavel e o log1p so
   introduziria uma assimetria sem ganho.

A ARMADILHA QUE ESTE SCRIPT EVITA. A calibracao exige janelas anteriores, entao as
primeiras nao recebem intervalo. Comparar a cobertura conformal medida nas ultimas janelas
contra os 84,5% do nativo medidos em TODAS as 103 seria comparar amostras diferentes e
creditar ao metodo o que e efeito do recorte. Aqui o nativo e REMEDIDO no mesmo
subconjunto, e as duas colunas falam das mesmas linhas.

Uso:
    PYTHONPATH=src python scripts/sensibilidade_conformal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CALIB = RES / "calibracao_2010_2023_predictions.csv"

HORIZONTE = 6
NIVEL = 0.95
# Quantas janelas anteriores exigir antes de emitir um intervalo. Com 103 janelas e 20 de
# calibracao sobram 83 avaliadas. Abaixo de ~20 o quantil de amostra finita fica grosseiro
# demais para 95%: com n residuos o maior quantil alcancavel e (n)/(n+1).
MIN_CALIB = 20


def quantil_conformal(residuos: np.ndarray, nivel: float) -> float:
    """Quantil conformal de amostra finita, indice ceil((n+1)*nivel).

    O quantil empirico simples fica sistematicamente abaixo do nominal com n pequeno. Esta
    correcao e o que faz a garantia valer. Igual ao de dengue_forecast/intervals.py.
    """
    n = residuos.size
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * nivel))
    if k < 1:
        return float(np.min(residuos))
    if k > n:
        return float(np.max(residuos))
    return float(np.partition(residuos, k - 1)[k - 1])


def interval_score(y, lo, hi, alpha):
    return ((hi - lo)
            + (2.0 / alpha) * np.clip(lo - y, 0, None)
            + (2.0 / alpha) * np.clip(y - hi, 0, None))


def conformaliza(d: pd.DataFrame, nivel=NIVEL, min_calib=MIN_CALIB) -> pd.DataFrame:
    """Anexa lo_conf e hi_conf usando SO janelas estritamente anteriores."""
    alpha = 1.0 - nivel
    d = d.copy()
    d["lo_conf"] = np.nan
    d["hi_conf"] = np.nan
    d["_res"] = d.y_true - d.y_pred

    for _, sub in d.groupby("model", sort=False):
        for h in range(1, HORIZONTE + 1):
            m = sub[sub.horizon == h].sort_values("window")
            idx = m.index.to_numpy()
            res = m["_res"].to_numpy()
            pred = m["y_pred"].to_numpy()
            for pos in range(len(idx)):
                if pos < min_calib:
                    continue
                calib = res[:pos]          # estritamente anteriores: nenhum vazamento
                d.at[idx[pos], "lo_conf"] = pred[pos] + quantil_conformal(calib, alpha / 2)
                d.at[idx[pos], "hi_conf"] = pred[pos] + quantil_conformal(calib, 1 - alpha / 2)
    return d.drop(columns=["_res"])


def mede(g: pd.DataFrame, lo: str, hi: str, alpha: float) -> dict:
    y = g.y_true.to_numpy(float)
    a, b = g[lo].to_numpy(float), g[hi].to_numpy(float)
    return {"picp": float(np.mean((y >= a) & (y <= b))),
            "mpiw": float(np.mean(b - a)),
            "is": float(np.mean(interval_score(y, a, b, alpha))),
            "n": int(len(g))}


def main() -> int:
    if not CALIB.exists():
        raise SystemExit(f"{CALIB} ausente; rode scripts/run_calibracao.py antes")
    d = pd.read_csv(CALIB, parse_dates=["date"])
    alpha = 1.0 - NIVEL

    d = conformaliza(d)
    aval = d.dropna(subset=["lo_conf"]).copy()
    print(f"  {len(d)} previsoes no total, {len(aval)} com intervalo conformal "
          f"({MIN_CALIB} janelas de calibracao por horizonte)")
    print(f"  janelas avaliadas: {aval.window.min()} a {aval.window.max()}")

    saida = {"_meta": {"nivel": NIVEL, "min_calib": MIN_CALIB, "horizonte": HORIZONTE,
                       "escala": "absoluta",
                       "fonte_do_metodo": "dengue-time-series/src/dengue_forecast/intervals.py",
                       "n_total": int(len(d)), "n_avaliadas": int(len(aval))},
             "modelos": {}}

    print(f"\n  A {int(NIVEL * 100)}% nominal, TODAS as linhas no mesmo subconjunto")
    print(f"  {'modelo':<9}{'intervalo':<12}{'PICP':>8}{'MPIW':>9}{'IS':>10}")
    for modelo in ("sarima", "prophet"):
        g = aval[aval.model == modelo]
        nativo = mede(g, "lo", "hi", alpha)
        conf = mede(g, "lo_conf", "hi_conf", alpha)
        # o nativo medido em TODAS as janelas, so para mostrar o efeito do recorte
        todo = mede(d[d.model == modelo], "lo", "hi", alpha)
        saida["modelos"][modelo] = {"nativo_subconjunto": nativo, "conformal": conf,
                                    "nativo_todas_as_janelas": todo}
        for rot, v in (("nativo", nativo), ("conformal", conf)):
            print(f"  {modelo:<9}{rot:<12}{v['picp']:>8.3f}{v['mpiw']:>9.0f}{v['is']:>10.0f}")
        print(f"  {'':<9}{'(nativo em 103)':<12}{todo['picp']:>8.3f}{todo['mpiw']:>9.0f}"
              f"{todo['is']:>10.0f}")

    print(f"\n  PICP por horizonte, conformal")
    print(f"  {'modelo':<9}" + "".join(f"{h:>8}" for h in range(1, HORIZONTE + 1)))
    for modelo in ("sarima", "prophet"):
        g = aval[aval.model == modelo]
        por_h = {h: float(((g[g.horizon == h].y_true >= g[g.horizon == h].lo_conf)
                           & (g[g.horizon == h].y_true <= g[g.horizon == h].hi_conf)).mean())
                 for h in range(1, HORIZONTE + 1)}
        saida["modelos"][modelo]["conformal_por_horizonte"] = por_h
        print(f"  {modelo:<9}" + "".join(f"{por_h[h]:>8.3f}" for h in range(1, HORIZONTE + 1)))

    dest = RES / "revisao" / "conformal_sensibilidade.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(saida, indent=2) + "\n", encoding="utf-8")
    aval.to_csv(RES / "revisao" / "conformal_predictions.csv", index=False)
    print(f"\n  results/revisao/{dest.name}")
    print("  SENSIBILIDADE: nada disto entrou no manuscrito.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
