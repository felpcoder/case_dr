# Instruções para rodar

Basta criar ambiente virtual com as bibliotecas definidas no requeriments.txt e em seguida rodar os 3 notebooks em ordem:

- exploratory_analysis.ipynb
- credit_model_notebook.ipynb
- scoring_submission.ipynb

Ao fim do processo, deve ser gerado arquivo submissao.csv

# Explicação da Abordagem

## Definição dos Targets: 

Foram feitos testes com `ever30mob3`, `ever30mob6`, `ever30mob9`, `ever30mob12` e o target escolhido para ser o target foi o: `ever30mob6`


O `dias_atraso` utilizado como insumo vem da função `cal_kpi`, calculado como:

```python
parcelas['dias_atraso'] = (data_final - data_prevista_pagamento).dt.days.clip(lower=0)
```

onde `data_final` é a data de quitação da parcela (ou o último pagamento registrado, caso não tenha quitado).

---

## 1. Cálculo do MOB

Para cada parcela, o MOB é calculado como a diferença em meses entre a `data_prevista_pagamento` e a data da **primeira parcela do contrato** (`data_referencia_contrato`), somando +1:

```python
df['mob'] = (
    (df['data_prevista_pagamento'].dt.year - df['data_referencia_contrato'].dt.year) * 12 +
    (df['data_prevista_pagamento'].dt.month - df['data_referencia_contrato'].dt.month) + 1
)
```

---

## 2. Flags no nível de parcela

Para cada parcela, verifica-se se o MOB está dentro da janela **e** se o atraso é maior ou igual a 30 dias:

```python
df['flag_30_mob3'] = ((df['mob'] <= 3) & (df['dias_atraso'] >= 30)).astype(int)
df['flag_30_mob6'] = ((df['mob'] <= 6) & (df['dias_atraso'] >= 30)).astype(int)
```

---

## 3. Agregação no nível de contrato e cliente

O `max` das flags por contrato resulta em `1` se **qualquer** parcela dentro da janela teve atraso ≥ 30 dias:

```python
ever30mob3=('flag_30_mob3', 'max'),
ever30mob6=('flag_30_mob6', 'max'),
```

---

## 4. Filtro de maturidade

Por fim, contratos muito jovens são removidos — mantém-se apenas aqueles com `max_mob >= 3` ou que já quebraram antes disso:

```python
base_modelagem = base_modelagem[
    (base_modelagem['max_mob'] >= 3) | (base_modelagem['ever30mob3'] == 1)
]
```

# Política de Crédito Proposta

## Limitações do modelo atual

Antes de definir a regra de decisão, é necessário registrar o desempenho real do modelo disponível:

| Métrica | Valor | Referência mínima desejável |
|---|---|---|
| AUC-ROC | 0,60 | > 0,75 |
| KS | 0,22 | > 0,35 |
| Bad rate da base | 0,58% | — |
| Gini | 0,20 | > 0,40 |

> ⚠️ **O modelo apresenta poder discriminatório limitado**, consequência direta do forte desbalanceamento da base (0,58% de bads). Nesse cenário, a probabilidade de default (PD) absoluta estimada pelo modelo **não é suficientemente calibrada** para sustentar um limiar fixo de corte (ex.: "recusar quem tem PD ≥ X%") com confiança estatística. Qualquer limiar absoluto seria arbitrário.

---

## Regra de decisão: corte direto por faixa de probabilidade

O modelo utilizado é uma **Árvore de Decisão**, cuja saída de probabilidade é **categórica por natureza** — cada cliente recebe a PD média da folha à qual pertence, resultando em apenas 8 valores distintos de probabilidade para toda a base. Isso inviabiliza a construção de decis contínuos e exige um corte direto sobre as faixas de PD produzidas pelo modelo.

A distribuição observada das probabilidades na base é a seguinte:

| Probabilidade estimada (PD) | Volume | % do total |
|---|---|---|
| 0,0000 | 2.367 | 5,9% |
| 0,1409 | 1.222 | 3,1% |
| 0,2097 | 8.035 | 20,1% |
| 0,4983 | 5.842 | 14,6% |
| 0,6033 | 324 | 0,8% |
| 0,6187 | 21.849 | 54,6% |
| 0,7901 | 226 | 0,6% |
| 0,9193 | 135 | 0,3% |

A política adota então um **corte direto por limiar de PD**:

| Faixa de PD | Perfil de risco | Decisão |
|---|---|---|
| PD < 0,50 | Baixo | ✅ **Aprovado** |
| 0,50 ≤ PD < 0,79 | Médio-alto | ⚠️ **Análise manual** |
| PD ≥ 0,79 | Alto | ❌ **Recusado automaticamente** |

> ⚠️ Um ponto crítico: **~54,6% da base** recebe PD = 0,6187 — a folha mais populosa da árvore. O modelo não consegue discriminar internamente esses contratos, reforçando a necessidade de análise manual nessa faixa e de evolução do modelo no médio prazo.

## Justificativas da escolha

**1. Adequação ao poder discriminatório real do modelo**

Com KS de 0,22, o modelo não gera PDs confiáveis em escala absoluta — mas ainda produz um ordenamento com algum valor informativo. A política de faixas de probabilidade aproveita exatamente esse ordenamento sem exigir calibração que o modelo ainda não comporta.

**2. Política transitória com revisão estruturada**

Esta é uma **política de primeira geração**, válida enquanto o modelo não atinge discriminação suficiente para sustentar um corte por PD absoluta. A revisão está condicionada ao acúmulo de eventos de bad na base:


# Decisões e justificativas relevantes

Nao serão usadas informações sensíveis que possam induzir a discriminação. Então sexo e data_nascimento estão descartadas do modelo como variáveis preditivas. Nesse sentido, embora sexo e data de nascimento possam eventualmente apresentar correlação estatística com o evento-alvo do modelo, sua utilização introduziria risco de viés discriminatório sistêmico, penalizando ou beneficiando indivíduos com base em características protegidas por lei e não em seu comportamento financeiro efetivo. Tal prática violaria diretamente os princípios da LGPD e os direitos fundamentais assegurados pela Constituição Federal.

O modelo não foi treinado com variáveis de produtos ou canais de venda. Acrescentar essas variáveis no modelo certamente aumentaria o poder preditivo possivelmente aumentando a assertividade ao distinguir classes, conforme IV calculado no notebook exploratory_analysis.ipynb. 

Foram treinados modelos em árvores pois o ideal seria identificar os clientes a serem aprovados e reprovados de acordo com if statements. Assim, teriamos regras facilmente interpretável pelos tomadores de decisão para embassar políticas em ambientes que são na prática fortemente regulados.

Com tempo limitado, não foi possível testar outras abordagens com detecção de eventos raros e algoritmos como isolation forest ou melhora na engenharia de features.