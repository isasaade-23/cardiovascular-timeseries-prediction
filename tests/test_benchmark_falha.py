"""O benchmark tem que falhar na hora quando perde uma janela ou um modelo.

Antes ele descartava a janela com um [WARN], seguia, e o CSV saia com menos previsoes sem
nada no arquivo dizendo isso. O erro so aparecia muito depois, no `to_matrices` do
bootstrap, que exige o retangulo janela por horizonte, e ai ja era longe da causa.

Estes testes usam modelos falsos de proposito. Testar a guarda com SARIMA ou XGBoost
exigiria quebra-los artificialmente e levaria minutos; o que esta sob teste e o contrato do
`run_backtest`, que nao depende de qual modelo falhou.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "src"))

from cv_timeseries.models import Forecaster  # noqa: E402
from run_benchmark import (  # noqa: E402
    BenchmarkIncompleto,
    build_models,
    run_backtest,
)


def serie(n=80):
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    return pd.Series(np.linspace(7000, 7500, n), index=idx, name="value")


class Bom(Forecaster):
    name = "bom"

    def forecast(self, train, horizon, exog_train=None, exog_future=None):
        return np.full(horizon, float(train.iloc[-1]))


class QuebraNaJanela(Bom):
    """Levanta excecao numa janela especifica, como um modelo que nao converge."""
    name = "quebra"

    def __init__(self, alvo=3):
        self.alvo = alvo
        self.vistas = 0

    def forecast(self, train, horizon, exog_train=None, exog_future=None):
        self.vistas += 1
        if self.vistas == self.alvo:
            raise ValueError("nao convergiu")
        return super().forecast(train, horizon)


class DevolveNaN(Bom):
    name = "nan"

    def __init__(self, alvo=2):
        self.alvo = alvo
        self.vistas = 0

    def forecast(self, train, horizon, exog_train=None, exog_future=None):
        self.vistas += 1
        out = super().forecast(train, horizon)
        if self.vistas == self.alvo:
            out[0] = np.nan
        return out


class TamanhoErrado(Bom):
    name = "curto"

    def __init__(self, alvo=4):
        self.alvo = alvo
        self.vistas = 0

    def forecast(self, train, horizon, exog_train=None, exog_future=None):
        self.vistas += 1
        h = horizon - 1 if self.vistas == self.alvo else horizon
        return np.full(h, float(train.iloc[-1]))


# --------------------------------------------------------------------------- #
# o caminho feliz continua igual
# --------------------------------------------------------------------------- #
def test_modelo_completo_passa_e_conta_as_janelas():
    metric, preds = run_backtest(serie(), Bom(), horizon=6, min_train_size=60)
    assert metric is not None
    assert metric["n_predictions"] == preds.window.nunique() * 6
    assert preds.window.nunique() == 80 - 60 - 6 + 1


# --------------------------------------------------------------------------- #
# as tres formas de perder uma janela
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls,trecho", [
    (QuebraNaJanela, "nao convergiu"),
    (DevolveNaN, "não-finita"),
    (TamanhoErrado, "tamanho"),
])
def test_janela_perdida_derruba_o_backtest(cls, trecho):
    """Cada motivo tem que chegar na mensagem, e nao so a contagem.

    Contar "14 de 15" diz que quebrou; dizer POR QUE e o que permite consertar sem rodar
    de novo com print no meio.
    """
    with pytest.raises(BenchmarkIncompleto) as e:
        run_backtest(serie(), cls(), horizon=6, min_train_size=60)
    msg = str(e.value)
    assert "de 15 janelas completas" in msg, msg
    assert trecho in msg, msg


def test_a_contagem_esperada_sai_do_gerador_de_janelas():
    """Nao ha 103 escrito no codigo.

    O numero de janelas depende de serie, horizonte e treino minimo. Fixa-lo faria a
    guarda mentir em qualquer recorte diferente do do paper, que e justamente onde as
    analises de sensibilidade vivem.
    """
    with pytest.raises(BenchmarkIncompleto, match="de 9 janelas"):
        run_backtest(serie(70), QuebraNaJanela(2), horizon=6, min_train_size=56)


def test_nada_e_devolvido_quando_falha():
    """CSV parcial e pior que CSV ausente: parece completo e nao e."""
    with pytest.raises(BenchmarkIncompleto):
        run_backtest(serie(), QuebraNaJanela(), horizon=6, min_train_size=60)


# --------------------------------------------------------------------------- #
# modelo pedido e indisponivel
# --------------------------------------------------------------------------- #
def test_modelo_pedido_e_indisponivel_derruba(monkeypatch):
    """Quem pediu o modelo na linha de comando espera ele no resultado.

    Antes o benchmark seguia sem ele e o CSV saia com uma linha a menos, o que ninguem
    nota lendo so o arquivo.
    """
    import run_benchmark as rb

    class Falha:
        def __init__(self, *a, **k):
            raise ImportError("biblioteca ausente nesta maquina")

    monkeypatch.setattr(rb, "XGBoostForecaster", Falha)
    with pytest.raises(BenchmarkIncompleto, match="XGBoost"):
        build_models(["xgboost", "naive"])


def test_baselines_continuam_sempre_disponiveis():
    """Sem dependencia externa, entao nunca podem cair nesta guarda."""
    modelos = build_models(["naive", "snaive", "snaive_drift"])
    assert [m.name for m in modelos] == ["naive", "snaive", "snaive_drift"]
