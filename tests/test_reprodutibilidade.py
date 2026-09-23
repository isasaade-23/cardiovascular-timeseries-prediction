"""Trava o que a auditoria descobriu sobre reprodutibilidade dos boosters.

Estes testes NAO rodam os modelos: ajustar 103 janelas de XGBoost leva minutos e nao cabe
numa suite. O que eles guardam e a CONSEQUENCIA da descoberta, que e o que se perde primeiro
quando alguem mexe no projeto meses depois:

- a versao do xgboost tem que continuar fixada, porque afrouxa-la muda um numero publicado;
- os resultados medidos em cada ambiente ficam registrados, para que uma execucao futura
  possa ser comparada contra eles em vez de contra a memoria de ninguem.

O achado completo esta em docs/xgboost_reprodutibilidade.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
OPCIONAIS = RAIZ / "requirements-optional.txt"
DOC = RAIZ / "docs" / "xgboost_reprodutibilidade.md"

# sMAPE do XGBoost base nas 103 janelas, por versao da biblioteca. Mesmo codigo, mesmos
# dados, mesma semente: so o ambiente muda. Medido em 2026-08-19 com n_jobs=-1 numa
# maquina de 12 nucleos, que era o default de entao. Serve para mostrar a amplitude
# entre versoes, e nao para reproduzir o valor publicado, que hoje sai de n_jobs=1.
POR_VERSAO = {
    "2.0.3": 6.854448,
    "2.1.4": 6.854448,
    "3.0.2": 6.832390,
    "3.1.0": 6.832390,
    "3.2.0": 6.832390,
    "3.3.0": 6.984887,
    # Colab, rodada independente da Isabella Saade; outra maquina e outro SO.
    "3.4.1": 6.803473,
}
# Lido do arquivo, e nao escrito aqui. Estava fixo no codigo, e por isso o teste que
# deveria falhar quando o resultado fosse regenerado NAO falhou: a constante nao acompanha
# o CSV. Numero de teste que nao le a fonte testa a memoria de quem escreveu.
def _guardado_em_results() -> float:
    import csv
    with (RAIZ / "results" / "benchmark_sim_real_sp_2010_2023_metrics.csv").open(
            encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha["model"] == "xgboost":
                return float(linha["smape"])
    raise AssertionError("linha do xgboost sumiu do metrics.csv")


# Configuracao fixada pelo repositorio: xgboost 3.2.0 com n_jobs=1. E a unica combinacao
# que nao depende da maquina, porque qualquer outra contagem de threads muda a ordem de
# soma dos gradientes.
FIXADO = 6.880428


def test_a_versao_do_xgboost_continua_fixada():
    """Regressao de um risco real, nao de um bug.

    `xgboost>=2.0.0` deixava cada maquina resolver para uma versao diferente, e versoes
    diferentes dao sMAPE diferente para o mesmo codigo. Trocar o `==` de volta por `>=`
    reintroduz silenciosamente a irreprodutibilidade da Tabela 1.
    """
    texto = OPCIONAIS.read_text(encoding="utf-8")
    linha = next((l for l in texto.splitlines()
                  if l.strip().startswith("xgboost") and not l.strip().startswith("#")), None)
    assert linha is not None, "xgboost sumiu de requirements-optional.txt"
    assert re.match(r"^xgboost==\d+\.\d+\.\d+$", linha.strip()), (
        f"xgboost precisa estar fixado com '==', esta como {linha.strip()!r}. "
        "O resultado do modelo muda entre versoes; ver docs/xgboost_reprodutibilidade.md."
    )


def test_a_versao_fixada_e_uma_das_medidas():
    """Nao adianta fixar numa versao cujo resultado ninguem mediu."""
    linha = next(l for l in OPCIONAIS.read_text(encoding="utf-8").splitlines()
                 if l.strip().startswith("xgboost=="))
    versao = linha.strip().split("==")[1]
    assert versao in POR_VERSAO, (
        f"xgboost fixado em {versao}, que nao esta na tabela de valores medidos "
        f"({sorted(POR_VERSAO)}). Meça antes de fixar."
    )


def test_o_intervalo_entre_ambientes_nao_muda_conclusao():
    """A amplitude e grande para o digito publicado e pequena para a conclusao.

    Este teste e o que impede alguem de ler a descoberta como "o resultado do XGBoost nao
    vale nada": em TODOS os ambientes medidos ele continua atras do naive sazonal e muito
    atras dos tres lideres, que e o que o paper afirma.
    """
    valores = list(POR_VERSAO.values())
    amplitude = max(valores) - min(valores)
    assert amplitude > 0.1, "se a amplitude sumiu, a tabela foi editada sem medir de novo"

    SEASONAL_NAIVE, MELHOR_LIDER = 6.2693, 4.7015
    for versao, v in POR_VERSAO.items():
        assert v > SEASONAL_NAIVE, f"xgboost {versao} passou o naive sazonal: {v}"
        assert v - MELHOR_LIDER > 2.0, f"xgboost {versao} chegou perto do lider: {v}"


def test_o_valor_guardado_e_o_da_configuracao_fixada():
    """O inverso do que este teste pedia antes, e essa inversao e a noticia.

    Ate 2026-09-23 a Tabela 1 trazia 6.9350, que nenhuma versao nem contagem de threads
    reproduzia, e este teste existia para registrar isso. O resultado foi regenerado com
    xgboost 3.2.0 e n_jobs=1, e agora o valor guardado E o da configuracao fixada. O teste
    passou a cobrar a igualdade em vez da diferenca.

    Le do CSV de proposito: a versao anterior fixava 6.9350 no codigo e por isso NAO
    falhou quando o resultado mudou, que era exatamente o momento em que ela deveria ter
    falhado.
    """
    guardado = _guardado_em_results()
    assert guardado == pytest.approx(FIXADO, abs=1e-4), (
        f"results/ tem {guardado:.6f} e a configuracao fixada produz {FIXADO:.6f}. "
        "Ou o resultado foi gerado noutro ambiente, ou n_jobs deixou de ser 1 em "
        "src/cv_timeseries/models.py."
    )


def test_o_modelo_continua_fixado_em_uma_thread():
    """A metade da reprodutibilidade que a versao fixada nao cobre.

    Fixar a versao nao basta: a mesma 3.2.0 da 6.827 em 4 threads e 6.880 em 1. Se alguem
    devolver n_jobs=-1 procurando velocidade, o numero publicado deixa de ser reproduzivel
    sem nada quebrar visivelmente.
    """
    fonte = (RAIZ / "src" / "cv_timeseries" / "models.py").read_text(encoding="utf-8")
    assert "n_jobs=1," in fonte, "n_jobs deixou de ser 1 no XGBoostForecaster"
    assert "n_jobs=-1," not in fonte, "n_jobs=-1 voltou; o resultado volta a depender da maquina"


@pytest.mark.parametrize("trecho", ["xgboost", "n_jobs", "CatBoost"])
def test_o_achado_esta_documentado(trecho):
    """O numero sozinho nao se explica; o teste aponta para onde ele se explica."""
    assert DOC.exists(), "docs/xgboost_reprodutibilidade.md sumiu"
    assert trecho in DOC.read_text(encoding="utf-8")
