# CHECKPOINT_02_SERS_2SEM

# Avaliação — APIs, energias renováveis e aprendizado de máquina

**Integrantes**

- Felipe Pereira Restivo — RM 570712
- Gabriel Rodrigues Zappelloni — RM 572060
- Kenichi Caio Yamamoto — RM 569815
- Maykon de Lima Silva — RM 574022
- Rodger Costa Rios — RM 571438

## Objetivo

Este projeto consulta duas APIs públicas de dados energéticos e meteorológicos e aplica aprendizado de máquina supervisionado em duas tarefas independentes:

1. **Classificação:** identificar a fonte de um empreendimento de geração (Solar, Eólica ou Hidráulica) a partir da potência outorgada e da localização.
2. **Regressão:** estimar a radiação solar horária em Petrolina (PE) a partir de variáveis meteorológicas e da hora do dia.

Em cada tarefa, três algoritmos foram treinados e comparados em Python (scikit-learn), e a análise foi reproduzida no Orange Data Mining.

## Fontes e período dos dados

| Tarefa | Fonte | Recorte | Arquivo |
|---|---|---|---|
| Classificação | [SIGA — ANEEL](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel) | Até 1.200 empreendimentos por sigla (UFV, EOL, UHE, PCH, CGH), agrupados em três classes | `aneel_classificacao_orange.csv` |
| Regressão | [Open-Meteo — API histórica](https://open-meteo.com/en/docs/historical-weather-api) | Petrolina (PE), −9,39 / −40,50, de 01/04/2025 a 30/06/2025, das 7h às 17h (fuso `America/Recife`) | `meteo_regressao_orange.csv` |

Nenhuma das consultas exige token ou credenciais.

**Observação sobre a ANEEL:** durante a execução, a API da ANEEL não respondeu (timeout). Por isso, o notebook carrega o CSV de referência publicado no repositório da disciplina, que tem a mesma estrutura gerada pela consulta. A célula original de consulta foi mantida no notebook. A API do Open-Meteo respondeu normalmente.

## Estrutura do repositório

```
├── README.md
├── Aula_APIs_Energia_Renovavel_ML.ipynb    # notebook completo (APIs, análise e modelos)
├── dados/                                  # CSVs gerados pelo notebook
│   ├── aneel_classificacao_orange.csv
│   ├── meteo_regressao_orange.csv
│   ├── aneel_treino_orange.csv / aneel_teste_orange.csv
│   └── meteo_treino_orange.csv / meteo_teste_orange.csv
└── orange/
    ├── *.ows                               # fluxos do Orange
    └── imagens/                            # capturas dos fluxos e resultados
```

## Como executar

1. Abra o notebook no [Google Colab](https://colab.research.google.com/) ou no Jupyter.
2. Execute as células em ordem (**Ambiente de execução → Executar tudo**). As bibliotecas utilizadas (pandas, NumPy, Matplotlib e scikit-learn) já vêm instaladas no Colab.
3. As células iniciais consultam as APIs e geram os CSVs na pasta do notebook. Se a API da ANEEL estiver indisponível, a célula de consulta exibirá um erro de timeout; nesse caso, siga a partir da célula seguinte, que carrega o CSV de referência.
4. Para o Orange, abra os arquivos `.ows` da pasta `orange/` e aponte os widgets **File** para os CSVs da pasta `dados/`.

## Tarefa 1 — Classificação da fonte renovável

**Configuração:** entradas `potencia_kw`, `latitude` e `longitude`; alvo `fonte`. Foram removidos 47 registros com coordenadas (0, 0), restando 3.829 empreendimentos. Divisão estratificada de 80% para treino e 20% para teste (`random_state=42`). KNN e Regressão Logística foram padronizados com `StandardScaler` dentro de um `Pipeline`, ajustado apenas no treino. Precision, Recall e F1 usam **média macro**.

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| KNN (k = 5) | 0,960 | 0,960 | 0,959 | 0,959 |
| Regressão Logística | 0,830 | 0,837 | 0,829 | 0,826 |
| **Random Forest** (200 árvores) | **0,977** | **0,977** | **0,976** | **0,977** |

**Conclusões:** o Random Forest obteve o melhor desempenho, seguido de perto pelo KNN. Ambos capturam fronteiras não lineares, compatíveis com a distribuição geográfica das fontes. A Regressão Logística, limitada a fronteiras lineares, teve como principal erro classificar empreendimentos solares como eólicos, fontes que se sobrepõem em potência e região, especialmente no interior do Nordeste.

O resultado, porém, reflete padrões regionais do cadastro, e não uma relação física entre potência, localização e tipo de fonte. Um parque solar instalado em uma região tradicionalmente eólica tenderia a ser classificado de forma errada. Além disso, a quantidade de registros por classe foi limitada na consulta e não representa a participação das fontes na matriz elétrica brasileira.

## Tarefa 2 — Regressão da radiação solar

**Configuração:** entradas `temperatura_c`, `umidade_pct`, `nuvens_pct`, `vento_kmh` e `hora`; alvo `radiacao_w_m2`; `data_hora` usada apenas para ordenação. Divisão temporal sem embaralhamento: as primeiras 800 horas (01/04 a 12/06/2025) para treino e as 201 finais (12/06 a 30/06/2025) para teste.

| Modelo | MAE (W/m²) | MSE ((W/m²)²) | R² |
|---|---|---|---|
| Regressão Linear | 145,205 | 30.034,201 | 0,360 |
| Árvore de Decisão (profundidade 6) | 90,938 | 15.127,027 | 0,678 |
| **Random Forest** (300 árvores) | **69,277** | **7.859,748** | **0,832** |

**Conclusões:** o Random Forest apresentou o menor erro e o maior R². A Regressão Linear teve o pior desempenho porque a relação entre hora e radiação não é linear: a radiação cresce pela manhã, atinge o máximo perto do meio-dia e decresce à tarde. A **hora** foi a variável mais importante, por determinar a elevação solar; as demais variáveis explicam os desvios em relação a esse ciclo diário. O período de teste, no fim de junho, tem radiação média inferior à do treino, o que impõe ao modelo uma leve mudança de distribuição.

Estimar radiação **não equivale a prever geração elétrica**. A radiação (W/m²) é a potência solar incidente em uma superfície horizontal. A energia produzida (kWh) depende ainda da área e da eficiência dos módulos, da inclinação e orientação dos painéis, da temperatura das células, de sombreamento e sujeira, e das perdas no inversor e no cabeamento. Os dados, além disso, vêm de modelos de reanálise, não de medições em uma usina.

## Atividade complementar — Orange Data Mining

As duas tarefas foram reproduzidas no Orange usando **as mesmas divisões do notebook**: os conjuntos de treino e teste foram exportados pelo Python e avaliados no widget **Test and Score** com a opção **Test on test data**. kNN e Logistic Regression receberam padronização pelo widget **Preprocess** (*Standardize to μ=0, σ²=1*).

### Classificação

| Modelo | CA | Precision | Recall | F1 |
|---|---|---|---|---|
| kNN (k = 5) | [preencher] | [preencher] | [preencher] | [preencher] |
| Logistic Regression | [preencher] | [preencher] | [preencher] | [preencher] |
| Random Forest (200 árvores) | [preencher] | [preencher] | [preencher] | [preencher] |

No Orange, Precision, Recall e F1 com a opção *Average over classes* usam média **ponderada** pela frequência das classes, enquanto o notebook usa média macro. Como as classes estão razoavelmente equilibradas, os valores ficam próximos.

### Regressão

| Modelo | MAE (W/m²) | MSE ((W/m²)²) | R² |
|---|---|---|---|
| Linear Regression | [preencher] | [preencher] | [preencher] |
| Tree (profundidade 6) | [preencher] | [preencher] | [preencher] |
| Random Forest (300 árvores) | [preencher] | [preencher] | [preencher] |

### Análise

[preencher após os resultados finais do Orange]
