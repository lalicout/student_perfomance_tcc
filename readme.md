
# Análise de Desempenho Escolar com Pipeline de Dados Python

## Visão Geral

Este projeto implementa um pipeline de análise de dados em Python para investigar o desempenho escolar de estudantes do ensino médio em Portugal, utilizando dados do dataset "Student Performance" da UCI Machine Learning Repository. O pipeline segue a metodologia CRISP-DM (Cross-Industry Standard Process for Data Mining) para extrair insights, preparar dados, selecionar atributos relevantes e construir modelos preditivos de classificação para prever a aprovação dos alunos.

## Metodologia

O projeto adota a metodologia CRISP-DM, abrangendo as seguintes etapas:

1.  **Business Understanding**: Definição dos objetivos do projeto, focando na predição da aprovação escolar.
2.  **Data Understanding**: Coleta, descrição e exploração inicial dos dados para identificar características e qualidade.
3.  **Data Preparation**: Limpeza, transformação (tradução, codificação de variáveis categóricas, scaling opcional), integração e formatação dos dados para análise. Inclui também a seleção de atributos relevantes.
4.  **Modeling**: Construção e avaliação de modelos de classificação (Regressão Logística, Árvore de Decisão, Random Forest, Gradient Boosting, SVM) para prever a aprovação.
5.  **Evaluation**: Análise dos resultados dos modelos em relação aos objetivos definidos, revisão do processo e determinação dos próximos passos.
6.  **Deployment**: (Não abordado nos módulos, mas o pipeline prepara os modelos para possível implantação).

## Fonte dos Dados

- **Nome:** Student Performance Dataset  
- **Origem:** UCI Machine Learning Repository  
- **Link:** [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)  
- **Conteúdo:** Desempenho acadêmico de estudantes do ensino médio em Portugal (disciplinas de Português e Matemática) e fatores socioeconômicos associados.

## Módulos do Pipeline

1.  **`pre_modelagem.py`**  
    - Carregamento dos dados brutos, limpeza, tradução, codificação, remoção de colunas irrelevantes e balanceamento de classes.
    - Funções: `importar_base`, `preparar_dados`, `balancear_dados`

2.  **`eda_functions.py`**  
    - Realização da EDA com visualizações, estatísticas descritivas, análise de outliers e correlações.
    - Funções: `plot_distribuicao_quantitativas`, `custom_heatmap`, `resumir_outliers`, `perfil_categorico_outliers`, `comparar_materias`

3.  **`feature_selection.py`**  
    - Seleção de atributos com base em testes estatísticos, VIF e regressões.
    - Funções: `calcular_vif`, `relatorio_multicolinearidade`, `selecionar_nominais_relevantes`, `selecionar_ordinais_relevantes`, `stepwise_selection`, `regressao_multipla`, `avaliar_residuos_regressao`

4.  **`modelagem.py`**  
    - Treinamento, avaliação e comparação de modelos com múltiplas métricas e validação cruzada.
    - Funções: `avaliar_classificadores_binarios_otimizados`, `verificar_overfitting`, `comparar_resultados_classificacao`

5.  **`documentar_resultados.py`**  
    - Padronização visual de gráficos e exportação de resultados para LaTeX.
    - Utilizado como suporte interno pelos demais módulos.

## Fluxo de Trabalho

Os módulos são executados na seguinte sequência:

1. `pre_modelagem.py` → 2. `eda_functions.py` → 3. `feature_selection.py` → 4. `modelagem.py`  
(Módulo `documentar_resultados.py` é usado ao longo do processo)

## Mapeamento CRISP-DM vs Módulos

| Etapa CRISP-DM          | Módulos Principais                      |
|------------------------|------------------------------------------|
| Business Understanding | (Conceitual)                            |
| Data Understanding     | `pre_modelagem.py`, `eda_functions.py`  |
| Data Preparation       | `pre_modelagem.py`, `feature_selection.py` |
| Modeling               | `feature_selection.py`, `modelagem.py`  |
| Evaluation             | `modelagem.py`, `feature_selection.py`  |
| Deployment             | (Fora do escopo dos módulos)            |

## Notebooks do Projeto

Além dos módulos Python, o projeto é acompanhado por notebooks organizados por fase da análise. Eles documentam o uso prático das funções e reúnem visualizações e resultados intermediários importantes.

### Notebooks Disponíveis

| Notebook                        | Objetivo Principal                                                                                 |
|--------------------------------|-----------------------------------------------------------------------------------------------------|
| `eda_por_disciplina.ipynb`     | Análise exploratória individual dos dados de Matemática e Português. |
| `eda_integrada.ipynb`          | Análise comparativa com amostra pareada (alunos cursando ambas as disciplinas). |
| `selecao_atributos.ipynb`      | Aplicação de testes estatísticos, verificação de VIF e regressões para seleção de variáveis. |
| `classificacao_portugues.ipynb`| Modelagem preditiva para prever aprovação em Português, com análise de overfitting e comparação de modelos. |
| `classificacao_matematica.ipynb`| Modelagem preditiva para prever aprovação em Matemática, com análise de overfitting e comparação de modelos. |


## Dependências Principais

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- statsmodels
