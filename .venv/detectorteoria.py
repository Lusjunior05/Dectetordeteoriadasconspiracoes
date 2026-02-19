"""
FactCheck AI Pro - Sistema de Investigação Baseado no Método Científico
Versão: 4.0.0
Descrição: Análise profunda de desinformação utilizando Ciclo de Investigação Científica,
            integração com IA (Groq/LLaMA 3), busca heurística via Tavily e relatório
            pericial completo com citação de evidências e técnicas científicas validadas.
"""

import os
import re
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Optional

import markdown2
from groq import Groq
from tavily import TavilyClient
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from xhtml2pdf import pisa

# ─────────────────────────────────────────────
#  INICIALIZAÇÃO DO CONSOLE RICH
# ─────────────────────────────────────────────
console = Console()

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO E VALIDAÇÃO DE AMBIENTE
# ─────────────────────────────────────────────

@dataclass
class Config:
    """Configurações centralizadas da aplicação com foco em precisão científica."""
    groq_key: str
    tavily_key: str
    max_resultados: int = 12
    modelo_ia: str = "llama-3.3-70b-versatile"
    temperatura: float = 0.0
    max_tentativas_retry: int = 3
    pasta_relatorios: str = "laudos_periciais"


def carregar_config() -> Config:
    """Carrega chaves de API e valida o ambiente."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not groq_key or not tavily_key:
        raise EnvironmentError("APIs GROQ_API_KEY ou TAVILY_API_KEY não configuradas.")

    return Config(groq_key=groq_key, tavily_key=tavily_key)


# ─────────────────────────────────────────────
#  TÉCNICAS CIENTÍFICAS DE VERIFICAÇÃO
# ─────────────────────────────────────────────

TECNICAS_CIENTIFICAS = """
## TÉCNICAS CIENTÍFICAS DE VERIFICAÇÃO DE FATOS

As seguintes técnicas reconhecidas internacionalmente foram aplicadas nesta análise:

### 1. MÉTODO SIFT (Stop, Investigate, Find, Trace)
- **Stop**: Pausar antes de reagir ou compartilhar a informação.
- **Investigate the source**: Investigar a origem e credibilidade da fonte primária.
- **Find better coverage**: Buscar cobertura mais ampla e cruzar com outras fontes.
- **Trace claims**: Rastrear afirmações, fotos ou citações até a fonte original.

### 2. FRAMEWORK CONSPIR
Avalia cada elemento da alegação sob sete critérios:
- **C**onsistência: O fato é consistente com o conhecimento estabelecido?
- **O**riginalidade: A notícia é original ou derivada de desinformação conhecida?
- **N**otoriedade: As fontes citadas são notoriamente confiáveis ou suspeitas?
- **S**uporte: Há suporte de múltiplas fontes independentes?
- **P**lausibilidade: A alegação é plausível dentro do contexto científico/histórico?
- **I**mparcialidade: A análise considera múltiplas perspectivas?
- **R**eferências: As referências são verificáveis e rastreáveis?

### 3. ANÁLISE LATERAL DE LEITURA (Lateral Reading)
Técnica usada por checadores profissionais:
- Em vez de ler o site/fonte verticalmente (de cima a baixo), busca-se informações *sobre* a fonte em outros sites.
- Identifica o histórico de credibilidade, afiliações e possíveis vieses da fonte original.

### 4. VERIFICAÇÃO DE PROVENIÊNCIA (Provenance Check)
- Rastreia a origem temporal da afirmação.
- Verifica se conteúdos audiovisuais (imagens, vídeos) foram retirados de contexto.
- Utiliza ferramentas de busca reversa e análise de metadados.

### 5. ANÁLISE DE LINGUAGEM E RETÓRICA
Identifica padrões linguísticos associados à desinformação:
- Uso excessivo de maiúsculas, pontuação exclamativa e linguagem emocional.
- Títulos clickbait e ausência de atribuição de fontes.
- Generalizações indevidas e apelos à autoridade sem referência verificável.

### 6. CRITÉRIO POYNTER / IFCN
Baseado nos padrões do International Fact-Checking Network (IFCN):
- Comprometimento com imparcialidade e justiça.
- Transparência de fontes e metodologia.
- Transparência de financiamento e organização.
- Comprometimento com correções e responsabilidade aberta.

### 7. CROSS-REFERÊNCIA MULTI-FONTE
- Confirmação por no mínimo 3 fontes independentes e de alta credibilidade.
- Hierarquia de evidências: estudos peer-reviewed > agências oficiais > veículos especializados > imprensa geral.

### 8. ANÁLISE TEMPORAL
- Verifica se a informação é recente ou antiga sendo reapresentada como nova.
- Contextualiza eventos com sua cronologia original.
"""

# ─────────────────────────────────────────────
#  PROMPTS DE ENGENHARIA
# ─────────────────────────────────────────────

PROMPT_SISTEMA_CIENTIFICO = """Você é um Analista Pericial Sênior de Fatos e Desinformação, certificado pelo IFCN.
Sua tarefa é produzir um LAUDO TÉCNICO DE INVESTIGAÇÃO COMPLETO baseado no MÉTODO CIENTÍFICO.

Siga RIGOROSAMENTE esta estrutura:

# LAUDO PERICIAL DE VERIFICAÇÃO DE FATOS
## Investigação Nº {id_caso} | {data_hora}

---

## 1. IDENTIFICAÇÃO DO CASO
- **Objeto da Investigação:** (reproduza o fato analisado)
- **Data de Análise:** {data_hora}
- **Classificação Preliminar:** (Afirmação Factual / Opinativa / Estatística / Histórica)
- **Grau de Viralização Estimado:** (Baixo / Médio / Alto / Viral)

---

## 2. METODOLOGIA APLICADA
Liste TODAS as técnicas científicas utilizadas nesta análise:
- Método SIFT (Stop, Investigate, Find, Trace)
- Framework CONSPIR
- Leitura Lateral (Lateral Reading)
- Verificação de Proveniência
- Análise de Linguagem e Retórica
- Cross-referência Multi-fonte
- Análise Temporal
- Critério POYNTER/IFCN

Explique brevemente como cada técnica foi aplicada ao caso específico.

---

## 3. OBSERVAÇÃO E CONTEXTUALIZAÇÃO (Etapa 1 - Método Científico)
- Descreva o contexto completo em que a alegação circula.
- Identifique o público-alvo da desinformação (se aplicável).
- Mapeie os padrões de disseminação identificados.
- Descreva o ambiente informacional (redes sociais, grupos específicos, etc.).

---

## 4. FORMULAÇÃO DE HIPÓTESES (Etapa 2 - Método Científico)
Formule explicitamente as hipóteses a serem testadas:
- **H1 (Hipótese Nula):** O fato é verdadeiro e bem contextualizado.
- **H2 (Hipótese Alternativa 1):** O fato é falso ou fabricado.
- **H3 (Hipótese Alternativa 2):** O fato é verdadeiro, mas apresentado fora de contexto.
- **H4 (Hipótese Alternativa 3):** O fato é parcialmente verdadeiro (contém elementos reais distorcidos).

---

## 5. CONJUNTO PROBATÓRIO — EVIDÊNCIAS COLETADAS (Etapa 3 - Método Científico)

Para CADA fonte encontrada, apresente:

### Evidência [N]:
- **URL:** [link completo]
- **Título da Fonte:** [título]
- **Veículo/Organização:** [nome]
- **Data de Publicação:** [data]
- **Trecho Relevante:** "[citação direta ou paráfrase do conteúdo relevante]"
- **Avaliação de Credibilidade:**
  - Autoridade: [Alta / Média / Baixa] — justifique
  - Viés Identificado: [Neutro / Tendência X] — justifique
  - Independência: [Fonte independente / Afiliada a X]
  - Suporte às Hipóteses: [Confirma H1 / Refuta H1 / Inconclusivo]

*(Repita para todas as fontes encontradas)*

---

## 6. APLICAÇÃO DO FRAMEWORK CONSPIR

| Critério | Avaliação (0-10) | Justificativa |
|---|---|---|
| Consistência | X/10 | ... |
| Originalidade | X/10 | ... |
| Notoriedade das Fontes | X/10 | ... |
| Suporte Multi-fonte | X/10 | ... |
| Plausibilidade | X/10 | ... |
| Imparcialidade | X/10 | ... |
| Referências Verificáveis | X/10 | ... |
| **MÉDIA CONSPIR** | **X/10** | |

---

## 7. ANÁLISE RETÓRICA E LINGUÍSTICA
- Identifique padrões de linguagem manipulativa (se presentes).
- Avalie o uso de gatilhos emocionais.
- Analise a presença de falácias lógicas.
- Verifique se há apelos à autoridade sem referência verificável.

---

## 8. ANÁLISE TEMPORAL E DE PROVENIÊNCIA
- Quando surgiu originalmente esta afirmação?
- Foi reapresentada em novo contexto?
- Há evidências de manipulação de datas ou contextualização enganosa?

---

## 9. TESTE DAS HIPÓTESES E DISCUSSÃO (Etapa 4 - Método Científico)
- Confronte as evidências com cada hipótese levantada na Seção 4.
- Indique qual hipótese é sustentada pelo conjunto probatório.
- Quantifique o grau de certeza (Alta Certeza / Certeza Moderada / Incerteza / Indeterminado).

---

## 10. CONCLUSÃO E PARECER FINAL (Etapa 5 - Método Científico)
### 10.1 Síntese dos Achados
(Parágrafo resumindo as principais descobertas da investigação)

### 10.2 Veredito Final
**CLASSIFICAÇÃO:** [escolha UMA das opções abaixo]
- ✅ **VERDADEIRO** — A afirmação é factualmente correta e bem contextualizada.
- ❌ **FALSO** — A afirmação é factualmente incorreta ou fabricada.
- ⚠️ **PARCIALMENTE VERDADEIRO** — Contém elementos reais, mas distorcidos ou incompletos.
- 🔄 **FORA DE CONTEXTO** — A informação é real, mas apresentada de forma enganosa.
- ❓ **INCONCLUSIVO** — Evidências insuficientes para um veredito definitivo.
- 📅 **DESATUALIZADO** — A informação foi verdadeira em outro momento, mas não é mais atual.

### 10.3 Impacto Potencial
- Risco à saúde pública: [Sim/Não] — justifique
- Risco à segurança: [Sim/Não] — justifique
- Risco à democracia/processos eleitorais: [Sim/Não] — justifique
- Risco econômico: [Sim/Não] — justifique

### 10.4 Recomendações
- O que o leitor deve fazer ao encontrar esta informação?
- Quais fontes confiáveis consultar para verificação independente?

---

## 11. GLOSSÁRIO DE TERMOS TÉCNICOS
(Liste termos técnicos utilizados no laudo com definições acessíveis)

---

## 12. REFERÊNCIAS BIBLIOGRÁFICAS COMPLETAS
Liste TODAS as fontes consultadas em formato de referência acadêmica:
[N]. [Autor/Organização]. ([Data]). [Título]. Disponível em: [URL]. Acesso em: {data_hora}.

---
OBRIGATÓRIO — LINHA FINAL DO LAUDO:
SCORE_DESINFORMACAO: [0-100]
CONFIANCA_ANALISE: [ALTA/MEDIA/BAIXA]
VEREDITO_CODIGO: [VERDADEIRO/FALSO/PARCIAL/CONTEXTO/INCONCLUSIVO/DESATUALIZADO]
"""

PROMPT_RESUMO_EXECUTIVO = """Você é um comunicador científico. Com base no laudo técnico fornecido,
produza um RESUMO EXECUTIVO em linguagem acessível ao público geral (máximo 300 palavras).

O resumo deve conter:
1. O que foi verificado (1 frase)
2. O que as evidências mostram (2-3 frases)
3. O veredito final (1 frase clara e direta)
4. O que o cidadão deve fazer com esta informação (1-2 frases)

Evite jargão técnico. Use linguagem simples e direta."""

# ─────────────────────────────────────────────
#  DECORATOR E UTILITÁRIOS
# ─────────────────────────────────────────────

def com_retry(tentativas: int = 3, espera: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(tentativas):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == tentativas - 1:
                        raise
                    console.print(f"[yellow]⚠ Tentativa {i+1} falhou: {e}. Aguardando {espera}s...[/yellow]")
                    time.sleep(espera)
        return wrapper
    return decorator


def sanitizar_nome(nome: str) -> str:
    """Cria um nome de arquivo seguro."""
    return re.sub(r"[^\w\s-]", "", nome).strip().replace(" ", "_")[:50]


def extrair_metricas(laudo: str) -> dict:
    """Extrai métricas estruturadas do laudo gerado pela IA."""
    metricas = {
        "score": "N/A",
        "confianca": "N/A",
        "veredito": "N/A",
        "conspir_media": "N/A"
    }

    score_match = re.search(r"SCORE_DESINFORMACAO:\s*(\d+)", laudo)
    if score_match:
        metricas["score"] = score_match.group(1)

    confianca_match = re.search(r"CONFIANCA_ANALISE:\s*(\w+)", laudo)
    if confianca_match:
        metricas["confianca"] = confianca_match.group(1)

    veredito_match = re.search(r"VEREDITO_CODIGO:\s*(\w+)", laudo)
    if veredito_match:
        metricas["veredito"] = veredito_match.group(1)

    conspir_match = re.search(r"MÉDIA CONSPIR\s*\|\s*\*?\*?(\d+(?:\.\d+)?)/10", laudo)
    if conspir_match:
        metricas["conspir_media"] = f"{conspir_match.group(1)}/10"

    return metricas


def classificar_score(score_str: str) -> tuple[str, str]:
    """Retorna cor e label para o score de desinformação."""
    try:
        score = int(score_str)
        if score <= 20:
            return "green", "ALTAMENTE CONFIÁVEL"
        elif score <= 40:
            return "yellow", "MAIORITARIAMENTE VERDADEIRO"
        elif score <= 60:
            return "orange3", "SUSPEITO / VERIFICAR"
        elif score <= 80:
            return "red", "PROVÁVEL DESINFORMAÇÃO"
        else:
            return "bold red", "DESINFORMAÇÃO CONFIRMADA"
    except (ValueError, TypeError):
        return "white", "INDETERMINADO"


# ─────────────────────────────────────────────
#  MOTORES DE BUSCA E IA
# ─────────────────────────────────────────────

@com_retry()
def buscar_web(fato: str, config: Config) -> dict:
    """Executa busca profunda e estruturada para coleta de evidências."""
    tavily = TavilyClient(api_key=config.tavily_key)
    resultado = tavily.search(
        query=fato,
        search_depth="advanced",
        max_results=config.max_resultados,
        include_answer=True,
        include_raw_content=False
    )
    return resultado


@com_retry()
def gerar_laudo_ia(fato: str, evidencias: dict, client: Groq, config: Config) -> str:
    """Gera o laudo técnico completo com base nas evidências coletadas."""
    id_caso = datetime.now().strftime("%Y%m%d%H%M")
    data_hora = datetime.now().strftime("%d/%m/%Y às %H:%M")

    # Formata as evidências de forma estruturada para o prompt
    evidencias_formatadas = formatar_evidencias_para_prompt(evidencias)

    prompt_usuario = (
        f"OBJETO DE INVESTIGAÇÃO:\n{fato}\n\n"
        f"EVIDÊNCIAS COLETADAS PELA BUSCA WEB ({len(evidencias.get('results', []))} fontes):\n\n"
        f"{evidencias_formatadas}\n\n"
        f"RESPOSTA AUTOMÁTICA DA BUSCA (se disponível):\n"
        f"{evidencias.get('answer', 'Não disponível')}"
    )

    completion = client.chat.completions.create(
        model=config.modelo_ia,
        messages=[
            {
                "role": "system",
                "content": PROMPT_SISTEMA_CIENTIFICO.format(
                    id_caso=id_caso,
                    data_hora=data_hora
                )
            },
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=config.temperatura,
        max_tokens=8000
    )
    return completion.choices[0].message.content


@com_retry()
def gerar_resumo_executivo(laudo: str, client: Groq, config: Config) -> str:
    """Gera resumo executivo em linguagem acessível."""
    completion = client.chat.completions.create(
        model=config.modelo_ia,
        messages=[
            {"role": "system", "content": PROMPT_RESUMO_EXECUTIVO},
            {"role": "user", "content": f"LAUDO TÉCNICO:\n\n{laudo}"}
        ],
        temperature=0.1,
        max_tokens=600
    )
    return completion.choices[0].message.content


def formatar_evidencias_para_prompt(evidencias: dict) -> str:
    """Formata as evidências da busca web de forma legível para o prompt da IA."""
    resultados = evidencias.get("results", [])
    if not resultados:
        return "Nenhuma evidência encontrada."

    linhas = []
    for i, r in enumerate(resultados, 1):
        linhas.append(f"--- FONTE [{i}] ---")
        linhas.append(f"URL: {r.get('url', 'N/D')}")
        linhas.append(f"Título: {r.get('title', 'N/D')}")
        linhas.append(f"Data: {r.get('published_date', 'N/D')}")
        linhas.append(f"Conteúdo relevante: {r.get('content', 'N/D')[:800]}")
        linhas.append("")

    return "\n".join(linhas)


# ─────────────────────────────────────────────
#  GERAÇÃO DE RELATÓRIO HTML/PDF
# ─────────────────────────────────────────────

CSS_RELATORIO = """
    @page {
        margin: 2.5cm;
    }
    body {
        font-family: 'Arial', sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #222;
    }
    .capa {
        text-align: center;
        padding: 40px 0;
        border-bottom: 3px solid #002d5b;
        margin-bottom: 30px;
    }
    .capa h1 {
        color: #002d5b;
        font-size: 20pt;
        margin-bottom: 5px;
    }
    .capa .subtitulo {
        color: #555;
        font-size: 12pt;
        margin: 5px 0;
    }
    .badge-veredito {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 4px;
        font-size: 14pt;
        font-weight: bold;
        margin: 15px 0;
    }
    .badge-verdadeiro { background: #d4edda; color: #155724; border: 2px solid #155724; }
    .badge-falso { background: #f8d7da; color: #721c24; border: 2px solid #721c24; }
    .badge-parcial { background: #fff3cd; color: #856404; border: 2px solid #856404; }
    .badge-inconclusivo { background: #d1ecf1; color: #0c5460; border: 2px solid #0c5460; }
    .badge-contexto { background: #fce8d8; color: #7b3f00; border: 2px solid #7b3f00; }
    .score-box {
        background: #f4f6fb;
        border: 2px solid #002d5b;
        border-radius: 6px;
        padding: 15px 25px;
        margin: 20px 0;
        text-align: center;
    }
    .score-numero {
        font-size: 36pt;
        font-weight: bold;
        color: #002d5b;
    }
    .score-label { font-size: 11pt; color: #555; }
    h1 { color: #002d5b; font-size: 16pt; border-bottom: 2px solid #002d5b; padding-bottom: 5px; margin-top: 30px; }
    h2 { color: #0056b3; font-size: 13pt; margin-top: 22px; border-left: 5px solid #0056b3; padding-left: 10px; }
    h3 { color: #1a6b1a; font-size: 11pt; margin-top: 15px; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 10pt;
    }
    th {
        background: #002d5b;
        color: white;
        padding: 8px 10px;
        text-align: left;
    }
    td {
        border: 1px solid #ccc;
        padding: 7px 10px;
        vertical-align: top;
    }
    tr:nth-child(even) td { background: #f4f6fb; }
    blockquote {
        background: #f9f9f9;
        border-left: 5px solid #0056b3;
        margin: 10px 0;
        padding: 10px 15px;
        color: #333;
        font-style: italic;
    }
    .tecnicas-box {
        background: #eaf4ff;
        border: 1px solid #b8d4f0;
        border-radius: 6px;
        padding: 15px 20px;
        margin: 20px 0;
    }
    .resumo-executivo-box {
        background: #fffbea;
        border: 2px solid #e6c300;
        border-radius: 6px;
        padding: 15px 20px;
        margin: 20px 0;
        font-size: 11pt;
    }
    .url-fonte {
        font-family: monospace;
        font-size: 9pt;
        color: #0056b3;
        word-break: break-all;
    }
    .metadata-rodape {
        font-size: 9pt;
        color: #777;
        text-align: center;
        margin-top: 40px;
        padding-top: 10px;
        border-top: 1px solid #ccc;
    }
    code {
        background: #f0f0f0;
        padding: 2px 5px;
        border-radius: 3px;
        font-size: 9pt;
    }
    .aviso-ia {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 10px 15px;
        font-size: 10pt;
        color: #664d03;
        margin: 15px 0;
    }
"""


def determinar_badge_class(veredito: str) -> str:
    mapping = {
        "VERDADEIRO": "badge-verdadeiro",
        "FALSO": "badge-falso",
        "PARCIAL": "badge-parcial",
        "CONTEXTO": "badge-contexto",
        "INCONCLUSIVO": "badge-inconclusivo",
        "DESATUALIZADO": "badge-inconclusivo",
    }
    return mapping.get(veredito.upper(), "badge-inconclusivo")


def gerar_tabela_fontes_html(evidencias: dict) -> str:
    """Gera tabela HTML com todas as fontes coletadas e seus metadados."""
    resultados = evidencias.get("results", [])
    if not resultados:
        return "<p><em>Nenhuma fonte encontrada na busca.</em></p>"

    linhas = ""
    for i, r in enumerate(resultados, 1):
        url = r.get("url", "N/D")
        titulo = r.get("title", "N/D")
        data = r.get("published_date", "N/D")
        conteudo = r.get("content", "")[:250] + "..." if r.get("content") else "N/D"

        linhas += f"""
        <tr>
            <td><strong>[{i}]</strong></td>
            <td>{titulo}</td>
            <td class="url-fonte"><a href="{url}">{url}</a></td>
            <td>{data}</td>
            <td>{conteudo}</td>
        </tr>"""

    return f"""
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Título</th>
                <th>URL</th>
                <th>Data</th>
                <th>Trecho</th>
            </tr>
        </thead>
        <tbody>{linhas}</tbody>
    </table>"""


def exportar_laudo(
    fato: str,
    laudo_md: str,
    resumo_executivo: str,
    evidencias: dict,
    metricas: dict,
    config: Config
) -> tuple[str, str, str]:
    """Gera os arquivos de saída do laudo pericial (MD, HTML e PDF)."""
    pasta = os.path.join(config.pasta_relatorios, sanitizar_nome(fato))
    os.makedirs(pasta, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_base = f"laudo_pericial_{timestamp}"
    data_geracao = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")

    # ── Markdown completo
    md_completo = f"""# LAUDO PERICIAL — FactCheck AI Pro v4.0
**Objeto:** {fato}
**Gerado em:** {data_geracao}
**Score de Desinformação:** {metricas['score']}/100
**Veredito:** {metricas['veredito']}
**Confiança da Análise:** {metricas['confianca']}

---

## RESUMO EXECUTIVO
{resumo_executivo}

---

## TÉCNICAS CIENTÍFICAS APLICADAS
{TECNICAS_CIENTIFICAS}

---

{laudo_md}
"""
    md_path = os.path.join(pasta, f"{nome_base}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_completo)

    # ── HTML rico para PDF
    laudo_html_body = markdown2.markdown(
        laudo_md,
        extras=["tables", "fenced-code-blocks", "header-ids"]
    )
    tecnicas_html = markdown2.markdown(TECNICAS_CIENTIFICAS, extras=["tables"])
    tabela_fontes = gerar_tabela_fontes_html(evidencias)
    badge_class = determinar_badge_class(metricas.get("veredito", "INCONCLUSIVO"))

    score_val = metricas.get("score", "N/A")
    cor_score = "#155724"
    try:
        s = int(score_val)
        if s > 60:
            cor_score = "#721c24"
        elif s > 40:
            cor_score = "#856404"
        elif s > 20:
            cor_score = "#0c5460"
    except (ValueError, TypeError):
        pass

    html_completo = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Laudo Pericial — FactCheck AI Pro</title>
    <style>{CSS_RELATORIO}</style>
</head>
<body>

<!-- CAPA -->
<div class="capa">
    <h1>🔍 LAUDO PERICIAL DE VERIFICAÇÃO DE FATOS</h1>
    <div class="subtitulo">FactCheck AI Pro — Versão 4.0.0</div>
    <div class="subtitulo">Módulo de Investigação Científica com IA</div>
    <hr>
    <div class="subtitulo"><strong>Objeto de Investigação:</strong></div>
    <blockquote style="font-size: 12pt; font-style: italic; margin: 10px auto; max-width: 80%;">
        "{fato}"
    </blockquote>
    <div class="subtitulo">Gerado em: {data_geracao}</div>
</div>

<!-- PAINEL DE MÉTRICAS -->
<div class="score-box">
    <table style="border: none; margin: 0;">
        <tr>
            <td style="border: none; text-align: center; width: 33%;">
                <div style="font-size: 9pt; color: #555; margin-bottom: 5px;">SCORE DE DESINFORMAÇÃO</div>
                <div class="score-numero" style="color: {cor_score};">{score_val}/100</div>
                <div class="score-label">(0 = Verdadeiro | 100 = Falso)</div>
            </td>
            <td style="border: none; text-align: center; width: 33%;">
                <div style="font-size: 9pt; color: #555; margin-bottom: 5px;">VEREDITO</div>
                <div class="badge-veredito {badge_class}">{metricas.get('veredito', 'N/A')}</div>
            </td>
            <td style="border: none; text-align: center; width: 33%;">
                <div style="font-size: 9pt; color: #555; margin-bottom: 5px;">CONFIANÇA DA ANÁLISE</div>
                <div style="font-size: 18pt; font-weight: bold; color: #002d5b;">{metricas.get('confianca', 'N/A')}</div>
                <div class="score-label">Score CONSPIR: {metricas.get('conspir_media', 'N/A')}</div>
            </td>
        </tr>
    </table>
</div>

<div class="aviso-ia">
    ⚠️ <strong>Nota:</strong> Este laudo foi gerado com auxílio de Inteligência Artificial e deve ser
    interpretado como apoio à investigação, não como conclusão jurídica definitiva. Verifique sempre
    as fontes originais referenciadas antes de tomar decisões com base neste documento.
</div>

<!-- RESUMO EXECUTIVO -->
<h1>📋 Resumo Executivo</h1>
<div class="resumo-executivo-box">
    {markdown2.markdown(resumo_executivo)}
</div>

<!-- FONTES COLETADAS -->
<h1>🔗 Índice de Fontes Coletadas ({len(evidencias.get("results", []))} fontes)</h1>
<p>A seguir, todas as fontes consultadas automaticamente durante a investigação:</p>
{tabela_fontes}

<!-- TÉCNICAS CIENTÍFICAS -->
<h1>🧪 Técnicas Científicas Aplicadas</h1>
<div class="tecnicas-box">
    {tecnicas_html}
</div>

<!-- LAUDO TÉCNICO COMPLETO -->
<h1>📄 Laudo Técnico Completo</h1>
{laudo_html_body}

<!-- RODAPÉ -->
<div class="metadata-rodape">
    Documento gerado automaticamente por <strong>FactCheck AI Pro v4.0.0</strong> em {data_geracao}<br>
    Modelo de IA: LLaMA 3.3 70B (Groq) | Busca: Tavily Advanced Search<br>
    Este documento segue os padrões metodológicos do IFCN (International Fact-Checking Network)
</div>

</body>
</html>"""

    # Salva HTML
    html_path = os.path.join(pasta, f"{nome_base}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_completo)

    # Gera PDF a partir do HTML
    pdf_path = os.path.join(pasta, f"{nome_base}.pdf")
    with open(pdf_path, "wb") as f_pdf:
        pisa.CreatePDF(html_completo, dest=f_pdf, encoding="utf-8")

    return md_path, html_path, pdf_path


# ─────────────────────────────────────────────
#  EXIBIÇÃO DE RESULTADOS NO TERMINAL
# ─────────────────────────────────────────────

def exibir_painel_resultados(fato: str, metricas: dict, md_file: str, html_file: str, pdf_file: str, n_fontes: int):
    """Exibe painel visual com os resultados no terminal."""
    score_str = metricas.get("score", "N/A")
    cor_score, label_score = classificar_score(score_str)

    console.print("\n")
    console.print(Panel.fit("✅  INVESTIGAÇÃO CONCLUÍDA", style="bold white on green"))
    console.print()

    # Tabela de métricas
    tabela = Table(title="📊 Resumo das Métricas", show_header=True, header_style="bold white on navy_blue")
    tabela.add_column("Métrica", style="bold cyan", min_width=25)
    tabela.add_column("Resultado", min_width=30)

    tabela.add_row("Objeto Investigado", fato[:60] + ("..." if len(fato) > 60 else ""))
    tabela.add_row("Score de Desinformação", f"[{cor_score}]{score_str}/100 — {label_score}[/{cor_score}]")
    tabela.add_row("Veredito Final", f"[bold]{metricas.get('veredito', 'N/A')}[/bold]")
    tabela.add_row("Confiança da Análise", metricas.get("confianca", "N/A"))
    tabela.add_row("Score CONSPIR", metricas.get("conspir_media", "N/A"))
    tabela.add_row("Fontes Analisadas", f"{n_fontes} fontes")
    tabela.add_row("Técnicas Aplicadas", "SIFT, CONSPIR, IFCN, Lateral Reading +4")

    console.print(tabela)
    console.print()

    # Localização dos arquivos
    console.print(Panel(
        f"[bold]📄 PDF:[/bold]  {pdf_file}\n"
        f"[bold]🌐 HTML:[/bold] {html_file}\n"
        f"[bold]📝 MD:[/bold]   {md_file}",
        title="📁 Arquivos Gerados",
        style="green"
    ))


# ─────────────────────────────────────────────
#  FLUXO PRINCIPAL
# ─────────────────────────────────────────────

def executar_investigacao(fato: str, client: Groq, config: Config):
    """Orquestra todas as etapas da investigação científica."""
    console.print(Panel(
        f"[bold blue]Iniciando Investigação Científica Completa[/bold blue]\n\n"
        f"[italic]Objeto:[/italic] {fato}\n\n"
        f"[dim]Técnicas: SIFT | CONSPIR | IFCN | Lateral Reading | Cross-referência Multi-fonte[/dim]",
        title="🔬 FactCheck AI Pro v4.0",
        border_style="blue"
    ))

    evidencias = {}
    laudo = ""
    resumo = ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Etapa 1: Coleta de evidências
        t1 = progress.add_task(description="[cyan]🔎 Coletando evidências na web (busca avançada)...", total=None)
        evidencias = buscar_web(fato, config)
        n_fontes = len(evidencias.get("results", []))
        progress.update(t1, description=f"[cyan]✔ {n_fontes} fontes coletadas.[/cyan]")

        # Etapa 2: Geração do laudo técnico
        t2 = progress.add_task(
            description="[magenta]🧠 Gerando laudo técnico com análise científica (IA)...", total=None
        )
        laudo = gerar_laudo_ia(fato, evidencias, client, config)
        progress.update(t2, description="[magenta]✔ Laudo técnico gerado.[/magenta]")

        # Etapa 3: Resumo executivo
        t3 = progress.add_task(
            description="[yellow]📝 Gerando resumo executivo em linguagem acessível...", total=None
        )
        resumo = gerar_resumo_executivo(laudo, client, config)
        progress.update(t3, description="[yellow]✔ Resumo executivo gerado.[/yellow]")

        # Etapa 4: Exportação
        t4 = progress.add_task(
            description="[green]📄 Exportando laudo pericial completo (PDF/HTML/MD)...", total=None
        )
        metricas = extrair_metricas(laudo)
        md_file, html_file, pdf_file = exportar_laudo(
            fato, laudo, resumo, evidencias, metricas, config
        )
        progress.update(t4, description="[green]✔ Arquivos gerados com sucesso.[/green]")

    # Exibe painel de resultados
    exibir_painel_resultados(fato, metricas, md_file, html_file, pdf_file, n_fontes)

    return metricas


# ─────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────

def main():
    try:
        config = carregar_config()
        client = Groq(api_key=config.groq_key)

        console.clear()
        console.print(Panel.fit(
            "[bold white]FACTCHECK AI PRO — MÓDULO PERICIAL v4.0[/bold white]\n"
            "[dim]Sistema de Verificação de Fatos Baseado em Método Científico[/dim]\n"
            "[dim]Técnicas: SIFT | CONSPIR | IFCN | Lateral Reading | Análise Temporal[/dim]",
            style="bold white on blue"
        ))

        fato = Prompt.ask("\n[bold]Insira o fato, afirmação ou notícia para investigação[/bold]")
        if fato.strip():
            executar_investigacao(fato.strip(), client, config)
        else:
            console.print("[red]Nenhuma entrada fornecida. Encerrando.[/red]")

    except EnvironmentError as e:
        console.print(f"[bold red]⚠ Erro de Configuração:[/bold red] {e}")
        console.print("[yellow]Configure as variáveis de ambiente GROQ_API_KEY e TAVILY_API_KEY.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Erro Crítico:[/bold red] {e}")
        logging.exception("Erro não tratado")


if __name__ == "__main__":
    main()