"""Defeitos de LaTeX que nao quebram a compilacao e por isso passam despercebidos.

Todos os casos aqui ja aconteceram neste projeto, dois deles mais de uma vez. O padrao e
sempre o mesmo: o documento compila, o PDF sai, e o erro so aparece quando alguem le a
pagina. Um teste e o unico jeito de pegar isso antes da submissao.

1. `\\ref` escrito numa string nao-raw do Python: o `\\r` vira o byte 0x0D e o arquivo fica
   com "Table~<CR>ef{...}". Compila, e o PDF imprime "Table~ef{tab:...}".
2. `%` cru vindo de um format `:.1%`: em LaTeX o `%` comenta o resto da linha, e a nota da
   tabela perde a frase inteira dali em diante.
3. Comando de referencia apontando para um rotulo que nao existe em lugar nenhum.
4. Tabela ou figura gerada que nao e `\\input` em nenhum documento: o resultado existe no
   repositorio e nao chega ao leitor.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PAPER = RAIZ / "paper"
MANUSCRITO = PAPER / "manuscript.tex"

TEX = sorted(PAPER.rglob("*.tex"))
CONTROLE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
CR = chr(13)


def _texto(f: Path) -> str:
    return f.read_bytes().decode("utf-8", errors="replace")


@pytest.mark.parametrize("f", TEX, ids=lambda f: f.name)
def test_nenhum_caractere_de_controle_solto(f):
    """Assinatura do `\\ref` mutilado, e de qualquer outro escape que vazou.

    O PDF sai bonito com "Table~ef{tab:calibracao}" no meio do texto. Ja aconteceu duas
    vezes, uma em tab1_desempenho.tex e outra no manuscrito.
    """
    for i, linha in enumerate(_texto(f).split("\n"), 1):
        corpo = linha[:-1] if linha.endswith(CR) else linha
        assert not CONTROLE.search(corpo), f"{f.name}:{i} tem caractere de controle"
        assert CR not in corpo, (
            f"{f.name}:{i} tem carriage return no meio da linha. "
            "Provavelmente um '\\\\ref' escrito numa string nao-raw do Python.")


@pytest.mark.parametrize("f", TEX, ids=lambda f: f.name)
def test_nenhuma_porcentagem_crua(f):
    """`84.5%` comenta o resto da linha. Ja comeu uma frase inteira da Tabela 6."""
    for i, linha in enumerate(_texto(f).split("\n"), 1):
        for m in re.finditer(r"(?<!\\)%", linha):
            antes = linha[:m.start()]
            if antes.lstrip().startswith("%"):
                continue                      # comentario de linha, legitimo
            assert not antes.rstrip().endswith(tuple("0123456789")), (
                f"{f.name}:{i} tem '%' cru depois de digito: {linha.strip()[:70]!r}. "
                "Em LaTeX isso comenta o resto da linha; use '\\\\%'.")


def test_toda_referencia_aponta_para_um_rotulo_existente():
    """`Table ??` no PDF e um erro que so aparece lendo a pagina certa."""
    rotulos = set()
    for f in TEX:
        rotulos |= set(re.findall(r"\\label\{([^}]+)\}", _texto(f)))
    texto = _texto(MANUSCRITO)
    # so o que o manuscrito realmente inclui conta como disponivel
    for inc in re.findall(r"\\input\{([^}]+)\}", texto):
        alvo = PAPER / (inc + ".tex")
        if alvo.exists():
            rotulos |= set(re.findall(r"\\label\{([^}]+)\}", _texto(alvo)))
    usados = set(re.findall(r"\\(?:ref|autoref|cref)\{([^}]+)\}", texto))
    faltando = sorted(usados - rotulos)
    assert not faltando, f"referencias sem rotulo alcancavel: {faltando}"


def test_toda_tabela_gerada_entra_no_documento():
    """Resultado que nao e input nao chega ao leitor.

    Foi assim que a tabela do Optuna e a de temperatura calendarizada ficaram de fora: as
    duas eram geradas, tinham rotulo, e o estudo chegava ao leitor so em prosa.
    """
    incluidas = set(re.findall(r"\\input\{tables/([^}]+)\}", _texto(MANUSCRITO)))
    existentes = {f.stem for f in (PAPER / "tables").glob("*.tex")}
    fora = sorted(existentes - incluidas)
    assert not fora, (
        f"tabelas geradas e nunca incluidas no manuscrito: {fora}. "
        "Ou entram, ou saem do gerador; nao ficam no meio.")
