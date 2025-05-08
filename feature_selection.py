# ======================================
# Módulo: feature_selection.py
# ======================================
"""Seleção de atributos e análises estatísticas para modelagem.

Este módulo agrupa funcionalidades voltadas para a etapa de seleção de
atributos em pipelines de machine learning e análise de dados. Ele oferece
ferramentas para identificar multicolinearidade, selecionar variáveis
preditoras relevantes (tanto categóricas quanto numéricas) através de
critérios estatísticos e algoritmos de seleção, e para construir e avaliar
modelos de regressão como meio de inferir a importância dos atributos.

Principais Capacidades:
    - Análise de multicolinearidade via Fator de Inflação de Variância (VIF)
      e correlações entre preditores.
    - Seleção de variáveis categóricas (nominais e ordinais) com base em
      testes de associação estatística (Qui-quadrado, Coeficiente de
      Contingência, Correlação de Spearman, Kruskal-Wallis).
    - Implementação de seleção de atributos stepwise para modelos lineares.
    - Funções para ajuste e interpretação de modelos de regressão linear
      múltipla, incluindo a seleção de atributos mais significativos.
    - Diagnóstico e avaliação de resíduos de modelos de regressão.
    - Geração de estatísticas descritivas detalhadas e avaliação de
      variáveis para construção de perfis.

O módulo visa auxiliar na redução da dimensionalidade dos dados, na melhoria
do desempenho e interpretabilidade dos modelos, e na validação de
suposições estatísticas. É primariamente aplicável às fases de Preparação
de Dados e Modelagem do CRISP-DM.
"""

# ==============================================================================
# ========================== IMPORTAÇÃO DE BIBLIOTECAS =========================
# ==============================================================================

# Importa funções de outros módulos
from eda_functions import aplicar_estilo_visual
from documentar_resultados import salvar_figura

# Bibliotecas python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr, kruskal, shapiro, zscore, entropy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Import para exibição em notebooks 
from IPython.display import display

# ==============================================================================
# ==================== SEÇÃO: ANÁLISE DE MULTICOLINEARIDADE ====================
# ==============================================================================

def relatorio_multicolinearidade(df, limite_vif=5.0, limite_corr=0.7):
    """Gera um relatório detalhado de multicolinearidade entre variáveis numéricas.

    Analisa um DataFrame de preditores numéricos para identificar
    multicolinearidade, calculando VIF e correlações entre pares. Combina
    essas informações em um relatório resumido e lista pares altamente
    correlacionados.

    Args:
        df (pd.DataFrame): DataFrame contendo as variáveis preditoras
            numéricas a serem analisadas. Recomenda-se tratar NaNs antes.
        limite_vif (float, optional): Limiar de VIF acima do qual a variável
            é considerada problemática. Default 5.0.
        limite_corr (float, optional): Limiar de correlação absoluta para
            identificar pares fortemente correlacionados. Default 0.7.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Uma tupla contendo:
            - resumo_vif (pd.DataFrame): Tabela com VIF, variáveis altamente
              correlacionadas e avaliação textual para cada preditor.
              Ordenado por VIF decrescente.
            - pares_altamente_correlacionados (pd.DataFrame): Lista dos pares
              de variáveis com correlação acima de `limite_corr`.
    """
    # === 1. Seleciona numéricas e calcula VIF ===
    # Seleciona apenas colunas numéricas e remove linhas com NaN para os cálculos
    X = df.select_dtypes(include=[np.number]).dropna()
    colunas = X.columns
    # Calcula VIF para cada coluna em relação às outras
    # variance_inflation_factor espera X sem constante explícita
    vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    df_vif = pd.DataFrame({'variavel': colunas, 'vif': vifs})

    # === 2. Matriz de correlação absoluta ===
    corr = X.corr().abs()

    # === 3. Extrai pares únicos de correlação (abaixo da diagonal) ===

    corr_df = corr.where(np.tril(np.ones(corr.shape, dtype=bool), k=-1)).stack().reset_index()
    
    if not corr_df.empty:
        corr_df.columns = ['variavel_1', 'variavel_2', 'correlacao']
    else: # Garante DataFrame vazio com colunas se não houver pares
        corr_df = pd.DataFrame(columns=['variavel_1', 'variavel_2', 'correlacao'])

    # === 4. Filtra pares com correlação alta ===
    pares_correlacionados = corr_df[corr_df['correlacao'] >= limite_corr].copy()
    # Exibe os pares correlacionados (para uso interativo)
    # Para uso em scripts, comentar ou remover esta linha:
    display(pares_correlacionados)

    # === 5. Agrupa variáveis correlacionadas por variável ===
    if not pares_correlacionados.empty:
        # Cria uma lista de relações (var1 -> var2 e var2 -> var1)
        relacoes_pares = pd.concat([
            pares_correlacionados[['variavel_1', 'variavel_2']],
            pares_correlacionados[['variavel_2', 'variavel_1']].rename(
                columns={'variavel_2': 'variavel_1', 'variavel_1': 'variavel_2'}
            )
        ])
        # Agrupa por variável e junta os nomes das correlacionadas
        relacoes = (
            relacoes_pares.groupby('variavel_1')['variavel_2']
            .apply(lambda x: ', '.join(sorted(x.unique())))
            .reset_index()
            .rename(columns={'variavel_1': 'variavel', 'variavel_2': 'Alta correlação com'})
        )
    else: # Garante DataFrame vazio com colunas se não houver relações
        relacoes = pd.DataFrame(columns=['variavel', 'Alta correlação com'])

    # === 6. Junta VIF com informações de correlação ===
    resumo = pd.merge(df_vif, relacoes, on='variavel', how='left')
    # Preenche NaN se uma variável não tem alta correlação com nenhuma outra
    resumo['Alta correlação com'] = resumo['Alta correlação com'].fillna('—')

    # === 7. Define avaliação textual ===
    def avaliar(row):
        # Função interna para classificar o nível de multicolinearidade
        if row['vif'] >= limite_vif and row['Alta correlação com'] != '—':
            return 'VIF alto + correlação alta'
        elif row['vif'] >= limite_vif:
            return 'VIF elevado'
        elif row['Alta correlação com'] != '—':
            return 'Correlação elevada'
        else:
            return 'Sem alerta'
    resumo['avaliacao'] = resumo.apply(avaliar, axis=1)

    # Ordena e retorna o resumo e os pares
    resumo_final = resumo.sort_values(by='vif', ascending=False)
    return resumo_final, pares_correlacionados


def calcular_vif(df, variaveis):
    """Calcula o Fator de Inflação de Variância (VIF) para variáveis especificadas.

    Calcula o VIF para cada variável na lista `variaveis` dentro do DataFrame `df`.
    Remove linhas com NaNs nas colunas especificadas e adiciona uma constante
    antes do cálculo (o VIF da constante é excluído do resultado).

    Args:
        df (pd.DataFrame): O DataFrame que contém os dados das variáveis.
        variaveis (List[str]): Lista dos nomes das colunas (variáveis
            independentes) para as quais o VIF será calculado.

    Returns:
        pd.DataFrame: DataFrame com colunas 'variavel' e 'VIF'.
    """
    # Seleciona as colunas e remove linhas com NaN
    X = df[variaveis].dropna()
    # Adiciona uma coluna constante (intercepto) ao DataFrame X
    X_with_const = sm.add_constant(X, has_constant='add')

    # Cria DataFrame para armazenar os resultados
    vif_data = pd.DataFrame()
    vif_data['variavel'] = X_with_const.columns
    # Calcula o VIF para cada coluna (incluindo a constante)
    vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i)
                       for i in range(X_with_const.shape[1])]

    # Retorna o DataFrame filtrado, removendo a linha da constante 'const'
    return vif_data[vif_data['variavel'] != 'const'].reset_index(drop=True)

# ==============================================================================
# ================== SEÇÃO: SELEÇÃO ESTATÍSTICA DE VARIÁVEIS ===================
# ==============================================================================

def selecionar_nominais_relevantes(df, categoria_de_interesse, variaveis_categoricas, c_c=0.3):
    """Seleciona variáveis nominais relevantes baseadas em associação com o alvo.

    Usa teste Qui-quadrado (significância, p<0.05) e Coeficiente de
    Contingência V de Cramér (força da associação, > `c_c`) para identificar
    variáveis nominais associadas à `categoria_de_interesse`.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados.
        categoria_de_interesse (str): Nome da coluna da variável alvo (categórica).
        variaveis_categoricas (List[str]): Lista dos nomes das colunas nominais
            a serem avaliadas.
        c_c (float, optional): Limiar mínimo para o Coeficiente de Contingência
            V de Cramér. Default 0.3.

    Returns:
        List[str]: Lista com os nomes das variáveis nominais selecionadas.
                   Retorna lista vazia se nenhuma atender aos critérios.

    Notes:
        - Exibe a tabela de resultados formatada para uso interativo.
        - O coeficiente calculado é o V de Cramér.
    """
    results = [] # Lista para armazenar resultados dos testes

    # Itera sobre cada variável categórica candidata
    for column in variaveis_categoricas:
        # Cria tabela de contingência entre a variável e o alvo
        contingency_table = pd.crosstab(df[column], df[categoria_de_interesse])
        # Realiza teste Qui-quadrado
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Calcula o coeficiente V de Cramér
        n = contingency_table.sum().sum() # Total de observações
        if n == 0: continue # Pula se não houver dados
        # Graus de liberdade mínimo (para normalização do V de Cramér)
        min_dim = min(contingency_table.shape) - 1
        if min_dim <= 0: continue # Pula se tabela for 1xN ou Nx1
        # Fórmula do V de Cramér
        v_cramer = (chi2 / (n * min_dim)) ** 0.5

        # Armazena resultado se p-valor for significativo
        if p < 0.05:
            results.append({
                'Variable': column,
                'Chi2': chi2,
                'P-Value': p,
                'V de Cramér': v_cramer # Usando nome correto
            })

    # Se não houver resultados significativos, informa e retorna lista vazia
    if not results:
        print("Nenhuma dependência estatisticamente significativa (p<0.05) encontrada.")
        return []

    # Cria DataFrame com os resultados significativos
    results_df = pd.DataFrame(results)
    # Filtra adicionalmente pelo limiar do V de Cramér
    selected_vars_df = results_df[results_df['V de Cramér'] > c_c].copy()

    # Se nenhuma variável passar no limiar c_c, informa e retorna lista vazia
    if selected_vars_df.empty:
        print(f"Nenhuma variável nominal atendeu ao limiar de V de Cramér > {c_c:.2f}.")
        return []

    # Ordena as variáveis selecionadas pela força da associação (V de Cramér)
    selected_vars_df.sort_values(by='V de Cramér', ascending=False, inplace=True)

    # Exibe a tabela formatada dos resultados selecionados (para uso interativo)
    print(f"Variáveis com P-Value < 0.05 e V de Cramér > {c_c:.2f}:")
    # Formata números para exibição
    display_df = selected_vars_df[['Variable', 'P-Value', 'V de Cramér']].applymap(
        lambda x: f"{x:.2e}" if isinstance(x, (float, int)) and (abs(x) < 1e-4 or abs(x) > 1e4)
        else round(x, 3) if isinstance(x, (float, int))
        else x
    )
    display(display_df)

    # Retorna a lista de nomes das variáveis selecionadas
    return display_df, selected_vars_df['Variable'].tolist()


def selecionar_ordinais_relevantes(df, variaveis_ordinais, target):
    """Seleciona variáveis ordinais relevantes baseadas em testes estatísticos.

    Avalia variáveis ordinais em relação a um `target` (numérico ou ordinal)
    usando Correlação de Spearman (para associação monotônica) e teste de
    Kruskal-Wallis (para diferença de medianas entre grupos). Seleciona
    variáveis com p-valor de Spearman significativo (<0.05).

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados.
        variaveis_ordinais (List[str]): Lista dos nomes das colunas ordinais.
        target (str): Nome da coluna da variável alvo.

    Returns:
        pd.DataFrame: DataFrame contendo as variáveis ordinais selecionadas
                      e suas estatísticas (Spearman, Kruskal-Wallis). Ordenado
                      pela magnitude da correlação de Spearman. Retorna
                      DataFrame vazio se nenhuma for selecionada.

    Notes:
        - Exibe a tabela de resultados formatada para uso interativo.
        - O teste de Kruskal-Wallis é calculado mas não usado para filtrar.
    """
    resultados = [] # Lista para armazenar resultados

    # Itera sobre cada variável ordinal candidata
    for var in variaveis_ordinais:
        # Remove NaNs para os testes pareados
        temp_df = df[[var, target]].dropna()
        if temp_df.empty or temp_df[var].nunique() < 2: continue # Pula se não houver dados/variação

        # Calcula Correlação de Spearman
        coef, p_spear = spearmanr(temp_df[var], temp_df[target])

        # Calcula Teste de Kruskal-Wallis
        # Cria lista de grupos (valores do target para cada nível da var ordinal)
        grupos = [temp_df[target][temp_df[var] == valor] for valor in sorted(temp_df[var].unique())]
        # Roda o teste se houver pelo menos 2 grupos
        if len(grupos) >= 2:
            h_stat, p_kruskal = kruskal(*grupos)
        else: # Caso contrário, marca como NaN
            h_stat, p_kruskal = np.nan, np.nan

        # Armazena os resultados
        resultados.append({
            'Variável': var,
            'Correlação (Spearman)': coef,
            'P-valor (Spearman)': p_spear,
            'Estatística H (Kruskal)': h_stat,
            'P-valor (Kruskal)': p_kruskal
        })

    # Se não houver resultados, retorna DataFrame vazio
    if not resultados:
        print("Nenhuma variável ordinal processada.")
        return pd.DataFrame()

    # Cria DataFrame com os resultados
    resultados_df = pd.DataFrame(resultados)

    # Filtra por significância no Spearman (p < 0.05)
    selecionadas_df = resultados_df[resultados_df['P-valor (Spearman)'] < 0.05].copy()

    # Se nenhuma passar no filtro, informa e retorna DF vazio
    if selecionadas_df.empty:
        print("Nenhuma variável ordinal com P-valor (Spearman) < 0.05 encontrada.")
        return pd.DataFrame()

    # Ordena pela força (absoluta) da correlação de Spearman
    selecionadas_df['Abs Spearman Corr'] = selecionadas_df['Correlação (Spearman)'].abs()
    selecionadas_df.sort_values(by='Abs Spearman Corr', ascending=False, inplace=True)
    selecionadas_df.drop(columns=['Abs Spearman Corr'], inplace=True) # Remove coluna auxiliar

    # Exibe a tabela formatada dos resultados selecionados (para uso interativo)
    print("Variáveis ordinais relevantes com base nos testes estatísticos:")
    # Formata números para exibição
    display_df = selecionadas_df.applymap(
        lambda x: f"{x:.2e}" if isinstance(x, (float, int)) and (abs(x) < 1e-4 or abs(x) > 1e4)
        else round(x, 3) if isinstance(x, (float, int))
        else x
    )
    display(display_df)

    # Retorna o DataFrame com as variáveis selecionadas e suas estatísticas
    return display_df


# ==============================================================================
# ===================== SEÇÃO: SELEÇÃO STEPWISE ================================
# ==============================================================================

def stepwise_selection(df, target, variaveis_candidatas,
                       criterion='p-value', #ou 'bic ou 'aic'
                       threshold_in=0.05, threshold_out=0.10, # Usado apenas se criterion='p-value'
                       verbose=False):
    """Realiza seleção de atributos bidirecional (stepwise) para regressão OLS.

    Iterativamente adiciona/remove variáveis com base em um critério
    especificado ('p-value', 'aic', 'bic') usando `statsmodels.OLS`.

    Args:
        df (pd.DataFrame): DataFrame com `target` e `variaveis_candidatas`.
                           Espera-se que preditores sejam numéricos.
        target (str): Nome da coluna alvo.
        variaveis_candidatas (List[str]): Lista de nomes das colunas preditoras
            a serem consideradas.
        criterion (str, optional): Critério para adicionar/remover variáveis.
            Opções: 'p-value', 'aic', 'bic'. Default 'p-value'.
        threshold_in (float, optional): Usado apenas se `criterion='p-value'`.
            P-valor máximo para adicionar variável. Default 0.05.
        threshold_out (float, optional): Usado apenas se `criterion='p-value'`.
            P-valor mínimo para manter variável. Default 0.10.
        verbose (bool, optional): Se True, imprime informações sobre as
            variáveis adicionadas ou removidas em cada etapa. Padrão é False.

    Returns:
        List[str]: Lista com os nomes das variáveis preditoras selecionadas.

    Notes:
        - Requer `statsmodels`.
        - Tratar NaNs no `df` antes de usar é recomendado.
        - AIC/BIC buscam minimizar o valor do critério.
    """
    variaveis_incluidas = [] # Lista de variáveis atualmente no modelo

    # Função auxiliar para obter o critério do modelo
    def get_model_criterion(model_results, criterion_type):
        if criterion_type == 'aic':
            return model_results.aic
        elif criterion_type == 'bic':
            return model_results.bic
        else: # p-value é tratado separadamente
            return None

    # Calcula critério inicial (modelo apenas com constante)
    try:
        modelo_inicial = sm.OLS(df[target], sm.add_constant(df[[]])).fit() # Modelo com intercepto
        current_criterion_value = get_model_criterion(modelo_inicial, criterion)
        if verbose and criterion != 'p-value':
            print(f"Critério inicial ({criterion.upper()}): {current_criterion_value:.4f}")
    except Exception as e_init:
        print(f"Erro ao ajustar modelo inicial: {e_init}. Abortando stepwise.")
        return []

    # Loop principal: continua enquanto o modelo mudar
    while True:
        mudou_modelo = False
        variavel_para_adicionar = None
        variavel_para_remover = None

        # --- Etapa Forward: Tentar adicionar a melhor variável ---
        variaveis_excluidas = list(set(variaveis_candidatas) - set(variaveis_incluidas))

        if variaveis_excluidas:
            melhor_p_valor_entrada = threshold_in # Para critério p-value
            melhor_crit_valor_entrada = current_criterion_value if current_criterion_value is not None else float('inf') # Para AIC/BIC

            for var_teste in variaveis_excluidas:
                try:
                    modelo_temp = sm.OLS(df[target],
                                         sm.add_constant(df[variaveis_incluidas + [var_teste]])
                                        ).fit()

                    if criterion == 'p-value':
                        p_valor_var_teste = modelo_temp.pvalues.get(var_teste, 1.0)
                        if p_valor_var_teste < melhor_p_valor_entrada:
                            melhor_p_valor_entrada = p_valor_var_teste
                            variavel_para_adicionar = var_teste
                    else: # AIC ou BIC
                        crit_valor_temp = get_model_criterion(modelo_temp, criterion)
                        if crit_valor_temp < melhor_crit_valor_entrada:
                            melhor_crit_valor_entrada = crit_valor_temp
                            variavel_para_adicionar = var_teste

                except Exception as e:
                    if verbose: print(f"Debug (Forward): Erro ao testar {var_teste}: {e}")
                    pass # Ignora erro e continua

            # Adiciona a variável se ela melhora o critério
            if variavel_para_adicionar:
                if criterion == 'p-value' or melhor_crit_valor_entrada < current_criterion_value:
                    variaveis_incluidas.append(variavel_para_adicionar)
                    mudou_modelo = True
                    current_criterion_value = melhor_crit_valor_entrada # Atualiza critério atual
                    if verbose:
                        if criterion == 'p-value':
                            print(f"Adicionada: {variavel_para_adicionar} (p={melhor_p_valor_entrada:.4f})")
                        else:
                            print(f"Adicionada: {variavel_para_adicionar} ({criterion.upper()}={melhor_crit_valor_entrada:.4f})")
                # Reseta variável para adicionar para a próxima iteração ou backward
                variavel_para_adicionar = None


        # --- Etapa Backward: Tentar remover a pior variável ---
        if not variaveis_incluidas:
             if not mudou_modelo: break
             else: continue # Se adicionou, não tenta remover na mesma iteração

        # Calcula o critério do modelo atual ANTES de tentar remover
        try:
            modelo_atual_results = sm.OLS(df[target], sm.add_constant(df[variaveis_incluidas])).fit()
            current_criterion_value = get_model_criterion(modelo_atual_results, criterion)
        except Exception as e:
             if verbose: print(f"Debug (Backward): Erro ao ajustar modelo atual: {e}")
             # Se não consegue ajustar o modelo atual, não pode fazer backward
             if not mudou_modelo: break # Termina se nada mudou
             else: continue # Continua se adicionou algo

        if criterion == 'p-value':
            p_valores_atuais = modelo_atual_results.pvalues.drop('const', errors='ignore')
            if not p_valores_atuais.empty:
                pior_p_valor = p_valores_atuais.max()
                if pior_p_valor > threshold_out:
                    variavel_para_remover = p_valores_atuais.idxmax()
        else: # AIC ou BIC
            melhor_crit_valor_remocao = current_criterion_value if current_criterion_value is not None else float('inf')

            for var_teste_remocao in variaveis_incluidas:
                vars_temp_remocao = [v for v in variaveis_incluidas if v != var_teste_remocao]
                try:
                    # Ajusta modelo sem a variável de teste
                    modelo_temp_remocao = sm.OLS(df[target],
                                                 sm.add_constant(df[vars_temp_remocao])
                                                ).fit()
                    crit_valor_temp_remocao = get_model_criterion(modelo_temp_remocao, criterion)

                    # Se remover esta variável melhora (diminui) o critério
                    if crit_valor_temp_remocao < melhor_crit_valor_remocao:
                        melhor_crit_valor_remocao = crit_valor_temp_remocao
                        variavel_para_remover = var_teste_remocao
                except Exception as e:
                    if verbose: print(f"Debug (Backward): Erro ao testar remoção de {var_teste_remocao}: {e}")
                    pass # Ignora e continua

        # Remove a variável se encontrada
        if variavel_para_remover:
            variaveis_incluidas.remove(variavel_para_remover)
            mudou_modelo = True
            current_criterion_value = melhor_crit_valor_remocao # Atualiza critério
            if verbose:
                if criterion == 'p-value':
                    # Precisa recalcular o p-valor que causou a remoção
                    p_valor_removido = modelo_atual_results.pvalues.get(variavel_para_remover, np.nan)
                    print(f"Removida: {variavel_para_remover} (p={p_valor_removido:.4f})")
                else:
                    print(f"Removida: {variavel_para_remover} (Novo {criterion.upper()}={melhor_crit_valor_remocao:.4f})")


        # Se nenhuma variável foi adicionada ou removida, o processo converge
        if not mudou_modelo:
            break

    if verbose:
        print(f"\nVariáveis selecionadas final ({criterion}): {variaveis_incluidas}")
    return variaveis_incluidas


# ==============================================================================
# ===================== SEÇÃO: AJUSTE DE REGRESSÃO =============================
# ==============================================================================

def regressao_multipla(df, target, variaveis):
    """Executa uma regressão linear múltipla (OLS) simples.

    Ajusta um modelo OLS para prever `target` usando as `variaveis`.
    Remove linhas com NaNs nas colunas envolvidas e adiciona uma constante.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        target (str): Nome da coluna alvo.
        variaveis (List[str]): Lista dos nomes das colunas preditoras.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Objeto
        de resultados da regressão do statsmodels.
    """
    # Seleciona colunas preditoras e remove linhas com NaN
    X = df[variaveis].dropna()
    # Seleciona a coluna alvo correspondente às linhas válidas de X
    y = df.loc[X.index, target] # Usa .loc para garantir alinhamento pelo índice
    # Adiciona constante (intercepto)
    X_with_const = sm.add_constant(X, has_constant='add')
    # Ajusta e retorna o modelo OLS
    modelo = sm.OLS(y, X_with_const).fit()
    return modelo


def ajustar_regressao(df, target_column, top_n=10):
    """Ajusta um modelo OLS completo e identifica os principais atributos.

    Ajusta um modelo OLS usando todas as colunas de `df` (exceto
    `target_column`) como preditores. Exibe o sumário do modelo e lista os
    `top_n` atributos mais significativos (por p-valor).

    Args:
        df (pd.DataFrame): DataFrame com alvo e preditores (numéricos).
        target_column (str): Nome da coluna alvo.
        top_n (int, optional): Número de atributos mais relevantes a listar.
            Default 10.

    Returns:
        Tuple[RegressionResultsWrapper, List[str], pd.Series, pd.Series]:
            Tupla contendo:
            - resultados_modelo: Objeto de resultados do statsmodels.
            - atributos_relevantes: Lista dos nomes dos `top_n` atributos.
            - y_true_model: Valores reais usados no modelo (após dropna).
            - y_pred: Valores previstos pelo modelo.

    Notes:
        - Exibe sumário e resultados para uso interativo.
        - Assume que `df` contém apenas alvo e preditores numéricos adequados.
    """
    # Separa features (X) e alvo (Y)
    X = df.drop(columns=[target_column], axis=1)
    Y = df[target_column]

    # Adiciona constante
    X_with_const = sm.add_constant(X, has_constant='add')

    # Ajusta o modelo OLS
    # OLS lida com NaNs internamente por exclusão listwise (padrão)
    modelo_ols = sm.OLS(Y, X_with_const)
    resultados = modelo_ols.fit()

    # Exibe o sumário completo do modelo (para uso interativo)
    print("--- Sumário do Modelo de Regressão OLS ---")
    print(resultados.summary())
    print("-----------------------------------------")

    # Obtem e exibe as variáveis mais relevantes usando função auxiliar
    # selecionar_atributos_results_regressao usa display internamente
    atributos_relevantes = selecionar_atributos_results_regressao(resultados, top_n=top_n)

    # Imprime a lista dos atributos relevantes
    if atributos_relevantes:
        print("\nAtributos de maior relevância (por P-valor):")
        for i, atributo in enumerate(atributos_relevantes, start=1):
            print(f"{i}. {atributo}")

    # Obtém valores reais e previstos DO MODELO (após exclusão listwise de NaNs)
    # y_true_model = Y[resultados.model.data.row_labels] # Se row_labels estiver disponível
    y_true_model = resultados.model.endog # Mais direto para obter o y usado
    y_pred = resultados.predict(resultados.model.exog) # Predições para o X usado

    # Retorna resultados, lista de atributos, y verdadeiro e y previsto do modelo
    return resultados, atributos_relevantes, pd.Series(y_true_model, index=resultados.model.data.row_labels, name=target_column), pd.Series(y_pred, index=resultados.model.data.row_labels, name=f"{target_column}_pred")


def selecionar_atributos_results_regressao(resultados, top_n=10, display_results=True):
    """Seleciona os atributos mais relevantes de um modelo de regressão por p-valor.

    Extrai coeficientes e estatísticas, ordena por p-valor ('P>|t|') e
    retorna os `top_n` melhores (excluindo a constante). Opcionalmente,
    exibe uma tabela formatada dos atributos selecionados.

    Args:
        resultados (RegressionResultsWrapper): Objeto de resultados do
            método `.fit()` de um modelo `statsmodels`.
        top_n (int, optional): Número de atributos a selecionar. Default 10.
        display_results (bool, optional): Se True, exibe a tabela formatada
            dos atributos selecionados. Default True.

    Returns:
        List[str]: Lista com os nomes dos `top_n` atributos mais significativos.
                   Retorna lista vazia se não houver atributos ou `top_n` for 0.
    """
    # Extrai a tabela de coeficientes do sumário
    try:
        summary_table = resultados.summary2().tables[1]
    except (AttributeError, IndexError):
        print("Aviso: Não foi possível extrair a tabela de coeficientes do sumário.")
        return []

    # Remove a linha da constante, se existir
    if 'const' in summary_table.index:
        summary_table = summary_table.drop(index='const')

    # Verifica se há atributos restantes
    if summary_table.empty:
        if display_results: print("Nenhum atributo preditor (além da constante) encontrado.")
        return []

    # Verifica se a coluna de p-valor existe
    if 'P>|t|' not in summary_table.columns:
        print("Aviso: Coluna 'P>|t|' não encontrada. Não é possível ordenar por p-valor.")
        return [] # Ou retorna summary_table.index.tolist() não ordenado?

    # Ordena pela coluna de p-valor
    sorted_table = summary_table.sort_values('P>|t|', ascending=True)

    # Seleciona as top N variáveis (garante que top_n não exceda o disponível)
    actual_top_n = min(top_n, len(sorted_table))
    if actual_top_n <= 0: return [] # Retorna vazio se top_n for 0 ou negativo
    top_variables_df = sorted_table.head(actual_top_n)

    # Exibe a tabela formatada se solicitado
    if display_results and not top_variables_df.empty:
        # Formata números para exibição
        display_table = top_variables_df.applymap(
            lambda x: f"{x:.2e}" if isinstance(x, (float, int)) and (abs(x) < 1e-4 or abs(x) > 1e4)
            else round(x, 3) if isinstance(x, (float, int))
            else x
        )
        # Seleciona colunas comuns para exibir
        cols_to_display = ['Coef.', 'Std.Err.', 't', 'P>|t|']
        existing_cols = [col for col in cols_to_display if col in display_table.columns]
        print("\n--- Atributos Mais Relevantes (por P-valor) ---")
        if existing_cols:
            display(display_table[existing_cols])
        else: # Fallback se colunas padrão não existirem
            display(display_table)
        print("-----------------------------------------------")

    # Retorna a lista com os nomes dos atributos selecionados
    return top_variables_df.index.tolist()

# ==============================================================================
# ===================== SEÇÃO: AVALIAÇÃO DE RESÍDUOS ===========================
# ==============================================================================

def avaliar_residuos_regressao(y_true, y_pred, nome_modelo='modelo', materia=None, salvar=False):
    """Avalia os resíduos de um modelo de regressão com plots e testes.

    Gera visualizações diagnósticas (resíduos vs. preditos, histograma, Q-Q plot,
    etc.) e calcula testes estatísticos (Shapiro-Wilk, Breusch-Pagan,
    Durbin-Watson) para verificar as suposições do modelo linear.

    Args:
        y_true (Union[pd.Series, np.ndarray]): Valores reais do alvo.
        y_pred (Union[pd.Series, np.ndarray]): Valores preditos pelo modelo.
        nome_modelo (str, optional): Nome do modelo para títulos. Default 'modelo'.
        materia (Optional[str], optional): Categoria ('portugues', 'matematica')
            para estilização e títulos. Default None.
        salvar (bool, optional): Se True, salva os gráficos. Default False.

    Returns:
        pd.DataFrame: DataFrame com resultados dos testes estatísticos.

    Notes:
        - O teste de Breusch-Pagan usa `y_pred` como regressor por padrão.
    """
    # Garante que y_true e y_pred tenham o mesmo comprimento
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` e `y_pred` devem ter o mesmo comprimento.")

    # Calcula os resíduos
    residuos = np.asarray(y_true) - np.asarray(y_pred)
    media_res = np.mean(residuos)

    # Define cor e colormap baseado na matéria
    cor = 'gray'
    cmap_cor = None
    sufixo_titulo_materia = ''
    if materia == 'portugues':
        sufixo_titulo_materia = ' - Português'
        cor = aplicar_estilo_visual('azul')[3] # Pega uma cor específica da paleta
        cmap_cor = aplicar_estilo_visual('azul', retornar_cmap=True)
    elif materia == 'matematica':
        sufixo_titulo_materia = ' - Matemática'
        cor = aplicar_estilo_visual('verde')[3]
        cmap_cor = aplicar_estilo_visual('verde', retornar_cmap=True)
    elif materia:
        sufixo_titulo_materia = f' - {materia.capitalize()}'

    # --- Gera Gráficos Diagnósticos ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Resíduos vs Valores Preditos
    axs[0, 0].scatter(y_pred, residuos, c=y_pred if cmap_cor else cor,
                      cmap=cmap_cor, alpha=0.7, edgecolors='k', linewidths=0.5)
    axs[0, 0].axhline(0, color='red', linestyle='--', lw=1)
    axs[0, 0].set_title('Resíduos vs. Valores Preditos')
    axs[0, 0].set_xlabel('Valores Preditos')
    axs[0, 0].set_ylabel('Resíduos')

    # 2. Histograma dos Resíduos
    sns.histplot(residuos, kde=True, ax=axs[0, 1], color=cor, bins='auto')
    axs[0, 1].set_title('Distribuição dos Resíduos')
    axs[0, 1].set_xlabel('Resíduos')

    # 3. Q-Q Plot
    sm.qqplot(residuos, line='45', ax=axs[0, 2], fit=True,
              markerfacecolor=cor, markeredgecolor=cor, alpha=0.7)
    axs[0, 2].set_title('Q-Q Plot dos Resíduos')
    # Tenta colorir a linha de referência
    try:
        if axs[0,2].lines:
            for line in axs[0,2].lines:
                if line.get_linestyle() == '--': line.set_color('red'); line.set_linewidth(1); break
    except Exception: pass # Ignora erro se não conseguir acessar/modificar linhas

    # 4. Resíduos vs Ordem dos Dados
    axs[1, 0].plot(residuos, marker='o', linestyle='-', color=cor, alpha=0.7, markersize=4)
    axs[1, 0].axhline(0, color='red', linestyle='--', lw=1)
    axs[1, 0].set_title('Resíduos (Ordem dos Dados)')
    axs[1, 0].set_xlabel('Índice da Observação')
    axs[1, 0].set_ylabel('Resíduos')

    # 5. Boxplot dos Resíduos
    sns.boxplot(y=residuos, ax=axs[1, 1], color=cor) # Orientação vertical
    axs[1, 1].set_title('Boxplot dos Resíduos')
    axs[1, 1].set_ylabel('Resíduos')

    # 6. Resíduos vs Valor Real
    axs[1, 2].scatter(y_true, residuos, alpha=0.7, color=cor, edgecolors='k', linewidths=0.5)
    axs[1, 2].axhline(0, color='red', linestyle='--', lw=1)
    axs[1, 2].set_title('Resíduos vs. Valor Real')
    axs[1, 2].set_xlabel('Valores Reais')
    axs[1, 2].set_ylabel('Resíduos')

    # Título geral e layout
    plt.suptitle(f'Análise dos Resíduos - {nome_modelo}{sufixo_titulo_materia}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Salva figura se solicitado
    if salvar:
        nome_arquivo_salvo = f"analise_residuos_{nome_modelo.lower().replace(' ', '_')}"
        if materia: nome_arquivo_salvo += f"_{materia.lower()}"
        try:
            # Tenta usar salvar_figura (requer importação)
            salvar_figura(nome_arquivo_salvo, materia=None, diretorio='analise_residuos')
        except NameError:
            # Fallback se salvar_figura não estiver disponível
            plt.savefig(f"{nome_arquivo_salvo}.png", dpi=300, bbox_inches='tight')
            print(f"Gráfico de análise de resíduos salvo como {nome_arquivo_salvo}.png")

    plt.show() # Exibe o gráfico

    # --- Calcula Testes Estatísticos ---
    # Shapiro-Wilk (Normalidade)
    stat_shapiro, p_shapiro = np.nan, np.nan
    if len(residuos) >= 3: # Teste requer >= 3 amostras
        try: stat_shapiro, p_shapiro = shapiro(residuos)
        except Exception: pass # Ignora erro (e.g., variância zero)

    # Breusch-Pagan (Homoscedasticidade) - Usa y_pred como regressor
    bp_lm_p_value, bp_f_p_value = np.nan, np.nan
    try:
        # Adiciona constante aos valores preditos para usar no teste
        X_bp = sm.add_constant(np.asarray(y_pred), has_constant='add')
        if X_bp.shape[0] > X_bp.shape[1]: # Verifica graus de liberdade
            _, bp_lm_p_value, _, bp_f_p_value = het_breuschpagan(residuos, X_bp)
    except Exception: pass # Ignora erro

    # Durbin-Watson (Autocorrelação)
    dw_stat = np.nan
    if len(residuos) > 1: # Teste requer > 1 amostra
        try: dw_stat = durbin_watson(residuos)
        except Exception: pass

    # Outliers (Z-score > 3)
    try:
        z_scores = np.abs(zscore(residuos, nan_policy='omit'))
        outliers_z3 = np.sum(z_scores > 3)
    except Exception:
        outliers_z3 = np.nan # Marca como NaN se zscore falhar

    # Monta DataFrame de resultados
    resultados_testes = {
        'Média dos Resíduos': [round(media_res, 4)],
        'Shapiro-Wilk Estatística': [round(stat_shapiro, 4) if not np.isnan(stat_shapiro) else np.nan],
        'Shapiro-Wilk P-valor': [round(p_shapiro, 4) if not np.isnan(p_shapiro) else np.nan],
        'Normalidade (Shapiro-Wilk > 0.05)': ['Sim' if p_shapiro > 0.05 else 'Não' if not np.isnan(p_shapiro) else 'N/A'],
        'Breusch-Pagan (LM) P-valor': [round(bp_lm_p_value, 4) if not np.isnan(bp_lm_p_value) else np.nan],
        'Homoscedasticidade (BP > 0.05)': ['Sim' if bp_lm_p_value > 0.05 else 'Não' if not np.isnan(bp_lm_p_value) else 'N/A'],
        'Durbin-Watson Estatística': [round(dw_stat, 4) if not np.isnan(dw_stat) else np.nan],
        'Autocorrelação (DW ≈ 2)': ['Ausente (aprox.)' if 1.5 < dw_stat < 2.5 else 'Presente (possível)' if not np.isnan(dw_stat) else 'N/A'],
        'Outliers (|z| > 3)': [outliers_z3]
    }
    df_resultados_testes = pd.DataFrame(resultados_testes)

    # Exibe resultados dos testes (para uso interativo)
    display(df_resultados_testes)

    return df_resultados_testes

# ==============================================================================
# ================= SEÇÃO: ESTATÍSTICAS DESCRITIVAS AVANÇADAS ==================
# ==============================================================================

def add_features_describe_pd(df, colunas, estudo_frequencia=False,
                              shapiro_values=True, dict_input=None, shannon=False):
    """Gera estatísticas descritivas detalhadas para colunas selecionadas.

    Calcula estatísticas para colunas numéricas (`estudo_frequencia=False`)
    ou categóricas (`estudo_frequencia=True`). Inclui opções para teste de
    Shapiro-Wilk, Coeficiente de Variação e Entropia de Shannon. Permite
    renomear as colunas do resumo.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        colunas (List[str]): Lista dos nomes das colunas a analisar.
        estudo_frequencia (bool, optional): Se True, trata como categóricas
            (converte para string). Se False (padrão), trata como numéricas.
        shapiro_values (bool, optional): Se True e `estudo_frequencia=False`,
            inclui p-valor do teste de Shapiro-Wilk. Default True.
        dict_input (Optional[Dict[str, str]], optional): Dicionário para
            renomear colunas do resumo. Chaves são nomes padrão (e.g., 'mean'),
            valores são novos nomes. Se None, usa padrão em português. Default None.
        shannon (bool, optional): Se True e `estudo_frequencia=True`, inclui
            Entropia de Shannon. Default False.

    Returns:
        pd.DataFrame: DataFrame com estatísticas descritivas. A coluna
                      'Contagem' é removida após ser impressa.
    """
    # Define nomes padrão para as colunas do resumo
    nomes_describe_padrao = {
        # Numéricas
        'count': 'Contagem', 'mean': 'Média', 'std': 'Desvio Padrão',
        'min': 'Mínimo', '25%': '1º Quartil (25%)', '50%': 'Mediana (50%)',
        '75%': '3º Quartil (75%)', 'max': 'Máximo',
        'Shapiro (p)': 'Shapiro-Wilk (p-valor)', 'CV': 'Coeficiente de Variação (CV)',
        'Moda': 'Moda',
        # Categóricas
        'unique': 'Total de Categorias', 'top': 'Categoria Dominante',
        'freq': 'Frequência Absoluta',
        'freq rel. top (%)': 'Frequência Relativa Dominante(%)',
        '% únicas': 'Diversidade de Categorias (%)',
        'Entropia (Shannon)': 'Entropia (Shannon)'
    }


    # Usa o dicionário fornecido ou o padrão
    col_renaming_map = dict_input if dict_input is not None else nomes_describe_padrao

    # --- Estudo de Frequência (Categóricas) ---
    if estudo_frequencia:
        # Converte colunas selecionadas para string para análise categórica
        cat_df = df[colunas].astype(str)
        # Obtém descrição básica para objetos/strings
        resumo_bruto = cat_df.describe(include=['object'])
        resumo_bruto = resumo_bruto.T

        # Imprime tamanho da amostra (se disponível)
        n_amostra = resumo_bruto['count'].iloc[0] if not resumo_bruto.empty else 'N/A'
        print(f"Tamanho da amostra (categórico): {n_amostra}")

        # Cria DataFrame de resumo e calcula métricas adicionais
        resumo = pd.DataFrame(index=resumo_bruto.index)
        resumo['count'] = resumo_bruto['count']
        resumo['unique'] = resumo_bruto['unique']
        resumo['top'] = resumo_bruto['top']
        resumo['freq'] = resumo_bruto['freq']
        # Calcula frequência relativa da moda e % de categorias únicas
        resumo['freq rel. top (%)'] = (resumo['freq'] / resumo['count'] * 100)
        resumo['% únicas'] = (resumo['unique'] / resumo['count'] * 100)

        # Calcula Entropia de Shannon se solicitado
        if shannon: # CORRIGIDO: usa o parâmetro 'shannon'
            entropies = {}
            for col_name in cat_df.columns:
                counts = cat_df[col_name].value_counts(normalize=True)
                entropies[col_name] = entropy(counts, base=2) if not counts.empty else np.nan
            resumo['Entropia (Shannon)'] = pd.Series(entropies).round(3)

        # Renomeia colunas usando o mapa definido
        resumo_renomeado = resumo.rename(columns=col_renaming_map)
        # Remove a coluna 'Contagem' (ou seu nome renomeado)
        col_contagem_nome = col_renaming_map.get('count', 'Contagem')
        if col_contagem_nome in resumo_renomeado.columns:
             resumo_final = resumo_renomeado.drop(columns=[col_contagem_nome])
        else:
             resumo_final = resumo_renomeado

    # --- Estudo Numérico ---
    else:
        # Obtém descrição básica para números
        resumo_bruto = df[colunas].describe().T

        # Imprime tamanho da amostra (se disponível)
        n_amostra = resumo_bruto['count'].iloc[0] if not resumo_bruto.empty else 'N/A'
        print(f"Tamanho da amostra (numérico): {n_amostra}")

        resumo = resumo_bruto.copy() # Começa com a descrição padrão

        # Calcula Moda (pega a primeira se houver múltiplas)
        modas_dict = {}
        for col_name in colunas:
            if not df[col_name].dropna().empty:
                modas = df[col_name].mode()
                modas_dict[col_name] = modas.iloc[0] if not modas.empty else np.nan
            else:
                modas_dict[col_name] = np.nan
        resumo['Moda'] = pd.Series(modas_dict)

        # Calcula p-valor de Shapiro-Wilk se solicitado
        if shapiro_values:
            shapiro_p_list = []
            for c in colunas:
                data_col = df[c].dropna()
                # Teste requer pelo menos 3 amostras não NaN
                if len(data_col) >= 3:
                    try:
                        shapiro_p_list.append(shapiro(data_col)[1]) # Pega apenas o p-valor
                    except Exception: shapiro_p_list.append(np.nan) # Em caso de erro (e.g., variância zero)
                else: shapiro_p_list.append(np.nan) # Marca como NaN se < 3 amostras
            # Formata p-valor como string com notação científica
            resumo['Shapiro (p)'] = [f"{p:.2e}" if not pd.isna(p) else np.nan for p in shapiro_p_list]

        # Calcula Coeficiente de Variação (CV = std / mean)
        # Evita divisão por zero ou CV sem sentido se média for zero ou negativa
        resumo['CV'] = np.where(resumo['mean'] > 0, (resumo['std'] / resumo['mean']).round(3), np.nan)

        # Renomeia colunas
        resumo_renomeado = resumo.rename(columns=col_renaming_map)
        # Remove coluna 'Contagem'
        col_contagem_nome = col_renaming_map.get('count', 'Contagem')
        if col_contagem_nome in resumo_renomeado.columns:
             resumo_final = resumo_renomeado.drop(columns=[col_contagem_nome])
        else:
            resumo_final = resumo_renomeado

    # Define nome do índice e retorna
    resumo_final.index.name = "Variável"
    return resumo_final


def avaliacao_variacao_pontuacao_media_por_categoria(df, atributos, coluna_avaliada='nota_final', marcar_alerta=True):
    """Avalia e pontua variáveis categóricas para relevância em perfilamento.

    Analisa `atributos` categóricos considerando diversidade (entropia) e
    a variação (gap) na média da `coluna_avaliada` entre suas categorias.
    Filtra e pontua atributos com base em critérios heurísticos internos.

    Args:
        df (pd.DataFrame): DataFrame de entrada com atributos e coluna avaliada.
        atributos (List[str]): Lista dos nomes das colunas categóricas a avaliar.
        coluna_avaliada (str, optional): Nome da coluna numérica (e.g., nota)
            a ser usada para calcular o gap de desempenho. Default 'nota_final'.
        marcar_alerta (bool, optional): Se True, adiciona coluna de alerta para
            alta dispersão de categorias. Default True.

    Returns:
        pd.DataFrame: DataFrame com atributos selecionados e suas métricas
                      (Entropia Normalizada, Variação Desempenho, Score Perfil),
                      ordenado por relevância.

    Notes:
        - Usa `add_features_describe_pd` internamente.
        - Contém limiares e pesos heurísticos codificados internamente.
    """
    # Dicionário com nomes para exibição no TCC
    colunas_exibidas_tcc = {
        'Total de Categorias':'Total de Categorias',
        'Categoria Mais Comum(CMC)': 'Categoria Dominante',
        'Frequência Relativa CMC (%)': 'Frequência Relativa Dominante(%)',
        'Entropia Relativa': 'Entropia Normalizada',
        'Gap Desempenho': 'Variação de Desempenho por Categoria',
        'PerilScore': 'Score - Perfil'
        }

    col_freq_abs = 'Frequência Absoluta'
    col_freq_rel = 'Frequência Relativa Dominante(%)'
    col_n_cat =   'Total de Categorias'
    col_prop_cat = 'Diversidade de Categorias (%)'
    col_entropia = 'Entropia (Shannon)'

    df_describe = add_features_describe_pd(df,colunas=atributos,estudo_frequencia=True,shannon=True) 
    
    # Estima o tamanho da amostra original
    freq_total = df_describe[col_freq_abs] / (df_describe[col_freq_rel] / 100)
    n_total = int(freq_total.median())
    freq_min = max(int(n_total * 0.01), 3)

    # Frequência relativa mínima
    df_describe['freq_rel_min'] = df_describe[col_n_cat].apply(lambda x: 5.0 if x <= 5 else 2.0)

    # Entropia relativa
    df_describe['Entropia Relativa'] = df_describe[col_entropia] / np.log2(df_describe[col_n_cat].replace(0, np.nan).astype(float))
    entropia_adequada = df_describe['Entropia Relativa'] >= 0.5

    # Gap de desempenho por variável
    gaps = {}
    for col in df_describe.index:
        if col in df.columns:
            medias = df.groupby(col)[coluna_avaliada].mean()
            gaps[col] = medias.max() - medias.min()
        else:
            gaps[col] = np.nan
    df_describe['Gap Desempenho'] = pd.Series(gaps)

    # Filtros
    freq_valida = df_describe[col_freq_abs] >= freq_min
    prop_valida = df_describe[col_freq_rel] >= df_describe['freq_rel_min']

    df_filtrado = df_describe[freq_valida & prop_valida & entropia_adequada].copy()

    # Escore composto padronizado
    escore_entropia = df_filtrado['Entropia Relativa'] / df_filtrado['Entropia Relativa'].max()
    escore_gap = df_filtrado['Gap Desempenho'] / df_filtrado['Gap Desempenho'].max()
    df_filtrado['PerfilScore'] = 0.5 * escore_entropia + 0.5 * escore_gap

    # Alerta de dispersão
    if marcar_alerta:
        dispersao = (df_filtrado[col_prop_cat] > 80.0) & (df_filtrado[col_n_cat] > 2)
        df_filtrado['Alerta Dispersão'] = dispersao.map({True: 'Alta dispersão (>80%)', False: ''})
    
    # Filtro final para selecionar variáveis mais relevantes
    
    df_final = df_filtrado[
        (df_filtrado['Entropia Relativa'] >= 0.6) &
        (df_filtrado['Gap Desempenho'] >= 1.0) &
        ~((df_filtrado[col_freq_rel] > 70.0) & (df_filtrado[col_n_cat] <= 3)) &
        (df_filtrado['Alerta Dispersão'] != 'Alta dispersão (>80%)')
    ].copy()
    # Ordenação
    df_final.sort_values(by='PerfilScore', ascending=False, inplace=True)

    # Filtra e renomeia apenas as colunas selecionadas
    df_final = df_final[[col for col in colunas_exibidas_tcc.keys() if col in df_final.columns]]
    df_final.rename(columns=colunas_exibidas_tcc, inplace=True)

    return df_final.round(3)

# ======================================
# Final do módulo feature_selection.py
# ======================================
