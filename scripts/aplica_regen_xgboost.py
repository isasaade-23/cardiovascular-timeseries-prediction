#!/usr/bin/env python3
"""Aplica a regeneracao do XGBoost em results/, com n_jobs fixo.

Decidido por Fabiano em 2026-09-23, depois de ver o antes e o depois produzido por
`scripts/regen_xgboost.py`. O valor guardado, 6,9350, nao era reproduzido por nenhuma das
sete versoes nem por nenhuma contagem de threads; o novo sai de um ambiente que qualquer
maquina consegue repetir.

POR QUE QUATRO ARQUIVOS E NAO UM. A linha da Tabela 1 e so a mais visivel. O XGBoost
aparece tambem na rodada com temperatura, na rodada de teto com vazamento e na janela
deslizante de 60 meses. Regenerar so a Tabela 1 deixaria a coluna "sem temperatura" da
Tabela 5 num ambiente e a coluna "com temperatura" noutro, que e exatamente o defeito que a
revisao apontou na tabela do Optuna. Ou tudo, ou nada.

O QUE NAO E TOCADO. Apenas as linhas do XGBoost sao substituidas; as dos outros modelos
sao preservadas byte a byte, porque prophet, sarima e catboost reproduzem com divergencia
0,0000 neste ambiente e o TimesFM nao e reexecutavel aqui. Reescrever o arquivo inteiro
rodando tudo de novo trocaria o TimesFM por nada.

Uso:
    PYTHONPATH=src python scripts/aplica_regen_xgboost.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SERIE = RES / "series" / "serie_eventos_sp_sim_real_2010_2023.csv"
TEMP = RES / "series" / "temperatura_sp_mensal_2010_2023.csv"
PY = sys.executable

# rotulo no CSV -> flags que produzem aquela rodada
RODADAS = {
    "benchmark_sim_real_sp_2010_2023": ("xgboost", []),
    "benchmark_exog_temp_climatology": (
        "xgboost_temp", ["--exog-csv", str(TEMP), "--exog-cols", "tmin",
                         "--exog-policy", "climatology"]),
    "benchmark_exog_temp_observed": (
        "xgboost_temp", ["--exog-csv", str(TEMP), "--exog-cols", "tmin",
                         "--exog-policy", "observed"]),
    "benchmark_slide60_2010_2023": (
        "xgboost_slide60", ["--max-train-size", "60"]),
}

# os tres conjuntos de incerteza, com a fonte de cada um. Seed 20260817, como documentado
# em docs/experiments/benchmark_2010_2023_baseline.md.
INCERTEZAS = {
    "uncertainty_2010_2023": ["benchmark_sim_real_sp_2010_2023"],
    "uncertainty_exog_climatology": ["benchmark_sim_real_sp_2010_2023",
                                     "benchmark_exog_temp_climatology"],
    "uncertainty_window_sensitivity": ["benchmark_sim_real_sp_2010_2023",
                                       "benchmark_slide60_2010_2023"],
}
SEED = 20260817


def smape(yt, yp):
    return float(np.mean(200.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp))))


def regenera(prefixo: str, rotulo: str, flags: list[str], tmp: Path) -> dict:
    """Roda so o XGBoost e devolve o antes/depois, sem escrever ainda."""
    saida = tmp / prefixo
    cmd = [PY, str(ROOT / "scripts" / "run_benchmark.py"),
           "--input-csv", str(SERIE), "--models", "xgboost",
           "--output-prefix", str(saida)] + flags
    env = {"PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={**dict(__import__("os").environ), **env})
    if r.returncode != 0:
        raise SystemExit(f"{prefixo} falhou:\n{r.stdout}\n{r.stderr}")

    novo = pd.read_csv(f"{saida}_predictions.csv", parse_dates=["date"])
    if novo.model.nunique() != 1:
        raise SystemExit(f"{prefixo}: esperava um modelo so, veio {novo.model.unique()}")
    novo["model"] = rotulo

    alvo = RES / f"{prefixo}_predictions.csv"
    antigo = pd.read_csv(alvo, parse_dates=["date"])
    linha_antiga = antigo[antigo.model == rotulo]
    if len(linha_antiga) != len(novo):
        raise SystemExit(f"{prefixo}: {len(linha_antiga)} linhas antigas, {len(novo)} novas")

    antes = smape(linha_antiga.y_true.to_numpy(float), linha_antiga.y_pred.to_numpy(float))
    depois = smape(novo.y_true.to_numpy(float), novo.y_pred.to_numpy(float))

    # ordem das linhas preservada: substitui no lugar, nao concatena no fim
    substituido = antigo.copy()
    pos = substituido.index[substituido.model == rotulo]
    for col in ("y_pred",):
        substituido.loc[pos, col] = novo[col].to_numpy()
    # y_true e window tem que bater; se nao baterem, a rodada nao e a mesma
    if not np.allclose(substituido.loc[pos, "y_true"].to_numpy(float),
                       novo.y_true.to_numpy(float)):
        raise SystemExit(f"{prefixo}: y_true divergiu, a rodada nao e a mesma")

    substituido.to_csv(alvo, index=False)

    # metricas: recalcula so a linha do rotulo, preserva as demais
    mpath = RES / f"{prefixo}_metrics.csv"
    met = pd.read_csv(mpath)
    yt = novo.y_true.to_numpy(float)
    yp = novo.y_pred.to_numpy(float)
    i = met.index[met.model == rotulo]
    met.loc[i, "mae"] = float(np.mean(np.abs(yp - yt)))
    met.loc[i, "rmse"] = float(np.sqrt(np.mean((yp - yt) ** 2)))
    met.loc[i, "smape"] = depois
    met.loc[i, "n_predictions"] = int(len(yt))
    met.to_csv(mpath, index=False)
    return {"rotulo": rotulo, "antes": antes, "depois": depois, "n": len(novo)}


def main() -> int:
    import tempfile
    print(f"  interpretador: {PY}")
    resultados = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for prefixo, (rotulo, flags) in RODADAS.items():
            print(f"  regenerando {rotulo} em {prefixo} ...", flush=True)
            resultados.append({"arquivo": prefixo, **regenera(prefixo, rotulo, flags, tmp)})

    print(f"\n  {'arquivo':<36}{'rotulo':<16}{'antes':>9}{'depois':>9}{'delta':>9}")
    for r in resultados:
        print(f"  {r['arquivo']:<36}{r['rotulo']:<16}{r['antes']:>9.4f}"
              f"{r['depois']:>9.4f}{r['depois']-r['antes']:>+9.4f}")

    print("\n  regenerando incerteza (seed 20260817) ...")
    for prefixo, fontes in INCERTEZAS.items():
        partes = [pd.read_csv(RES / f"{f}_predictions.csv", parse_dates=["date"])
                  for f in fontes]
        juntas = pd.concat(partes, ignore_index=True)
        entrada = RES / f"_tmp_{prefixo}.csv"
        juntas.to_csv(entrada, index=False)
        cmd = [PY, str(ROOT / "scripts" / "run_uncertainty.py"),
               "--predictions-csv", str(entrada),
               "--output-prefix", str(RES / prefixo),
               "--seed", str(SEED)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env={**dict(__import__("os").environ),
                                "PYTHONPATH": str(ROOT / "src")})
        entrada.unlink()
        if r.returncode != 0:
            raise SystemExit(f"{prefixo} falhou:\n{r.stdout}\n{r.stderr}")
        print(f"    {prefixo} ok")

    print("\n  Agora rode, nesta ordem:")
    print("    PYTHONPATH=src python scripts/reproduz_optuna_fixado.py")
    print("    PYTHONPATH=src python scripts/revisao/run_variants.py --variants base,roll,"
          "cal,diff12,full,direct --out results/revisao/variants")
    print("    PYTHONPATH=src python scripts/analisa_variantes.py")
    print("    PYTHONPATH=src python scripts/analisa_covid.py")
    print("    PYTHONPATH=src python scripts/build_paper_assets.py")
    print("    PYTHONPATH=src python scripts/revisao/build_revisao_tabs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
