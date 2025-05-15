# ======================================
# Módulo: modelagem.py
# ======================================
"""Avaliação e comparação de modelos de classificação binária.

Este módulo oferece um conjunto de ferramentas para treinar, avaliar de forma
abrangente e comparar o desempenho de múltiplos algoritmos de classificação
binária. É particularmente focado em cenários de dados educacionais, mas
pode ser adaptado para outros domínios.

Principais Funcionalidades:
    - Treinamento de classificadores com e sem otimização de hiperparâmetros
      (via GridSearch).
    - Divisão estratificada de dados e opção de balanceamento do conjunto de treino.
    - Cálculo de um vasto conjunto de métricas de desempenho (Acurácia, Precisão,
      Recall, F1-Score, AUC ROC) tanto para o conjunto de teste quanto para
      validação cruzada.
    - Geração de DataFrames sumarizando os resultados para fácil análise e
      exportação.
    - Ferramentas para diagnosticar overfitting comparando métricas de teste e CV.
    - Funções para visualização comparativa do desempenho dos modelos.
    - Geração de curvas ROC e Precision-Recall comparativas (dentro da avaliação).
    - Integração com módulos de documentação para exportar tabelas (LaTeX) e
      gráficos.

O módulo visa facilitar o processo iterativo de seleção do melhor modelo de
classificação para um determinado problema, alinhando-se fortemente com as
fases de Modelagem e Avaliação do CRISP-DM.
"""

# ==============================================================================
# ========================== IMPORTAÇÃO DE BIBLIOTECAS =========================
# ==============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc, make_scorer,
    confusion_matrix
)

# Importações explícitas dos outros módulos (substituindo import *)
from eda_functions import aplicar_estilo_visual, formatar_titulo
# Assume que pre_modelagem.py contém balancear_dados
from pre_modelagem import balancear_dados
# Assume que documentar_resultados.py contém exportar_df_para_latex e salvar_figura
from documentar_resultados import exportar_df_para_latex, salvar_figura

from IPython.display import display # Usado para exibir DataFrames em notebooks

# ==============================================================================
# ================= SEÇÃO: AVALIAÇÃO DE CLASSIFICADORES ========================
# ==============================================================================

def avaliar_classificadores_binarios_otimizados(
    X_train, y_train,X_test,y_test, classificadores, param_spaces=None,
    usar_balanceamento=False, materia='portugues',
    salvar = True
):
    """Avalia e otimiza múltiplos classificadores binários, incluindo curvas ROC/PR.

    Realiza um pipeline de avaliação para classificadores fornecidos:
    1. Divide dados em treino/teste (estratificado).
    2. Opcionalmente balanceia o treino (SMOTE-Tomek).
    3. Treina/avalia modelo base (sem otimização) no teste e via CV.
    4. Se `param_spaces` fornecido, otimiza via GridSearchCV e avalia o
       melhor modelo no teste e via CV.
    5. Calcula métricas: Acurácia, Precisão(0/1), Recall(0/1), F1(Reprovado/Macro), AUC ROC.
    6. Se a otimização foi realizada, plota curvas ROC e Precision-Recall
       comparando o modelo base e o otimizado.

    Args:    
        X_train, y_train (DataFrame, Series): Conjunto de treino.
        X_test, y_test (DataFrame, Series): Conjunto de teste.
        classificadores (Dict[str, Any]): Dicionário {nome_modelo: instancia_modelo_base}.
        param_spaces (Optional[Dict[str, Dict[str, List[Any]]]], optional):
            Dicionário {nome_modelo: grid_hiperparametros} para GridSearchCV.
            Default None (sem otimização).
        usar_balanceamento (bool, optional): Se True, aplica SMOTE-Tomek no treino.
            Default False.
        materia (str, optional): Identificador do contexto (usado para cores e títulos).
            Default 'portugues'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Uma tupla contendo:
            - metrics_df: Métricas no conjunto de teste (base e otimizado).
            - cv_metrics_df: Médias das métricas da validação cruzada (base e otimizado).
            - best_params_df: Melhores hiperparâmetros encontrados pelo GridSearchCV.

    Notes:
        - CV é feita no dataset completo original (X, y).
        - Scoring do GridSearchCV é 'f1_macro'.
        - AUC ROC usa predict_proba/decision_function; pode ser NaN se não disponíveis.
        - As curvas ROC/PR são geradas apenas se a otimização for realizada e
          se ambos os modelos (base e otimizado) puderem gerar scores/probabilidades.
    """
    # Inicializa listas para armazenar resultados
    metrics_list = []
    cv_metrics_list = []
    best_params_list = []

    # Define cores baseadas na matéria
    cores_map = {
        'portugues': ['#1B4F72','#a81434'], 
        'matematica': ['#196F3D','#d9711c'] 
    }
    # Cor 0 para Sem Otimização, Cor 1 para Com Otimização
    cor_0, cor_1 = cores_map.get(materia, ['#A9A9A9', '#464646']) # Cinza claro, Cinza escuro como default

    # Aplica balanceamento ao treino se solicitado
    if usar_balanceamento:
        print("Aplicando balanceamento SMOTE-Tomek aos dados de treino...")
        X_train, y_train = balancear_dados(X_train, y_train) # random_state interno a balancear_dados

    # Define o esquema de validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define as métricas a serem calculadas
    scorers = {
        'AUC ROC': 'roc_auc',
        'Acurácia': 'accuracy',
        'Precisão(0)': make_scorer(precision_score, pos_label=0, zero_division=0),
        'Precisão(1)': make_scorer(precision_score, pos_label=1, zero_division=0),
        'Recall(0)': make_scorer(recall_score, pos_label=0, zero_division=0),
        'Recall(1)': make_scorer(recall_score, pos_label=1, zero_division=0),
        'F1 Score (Reprovado)': make_scorer(f1_score, pos_label=0, zero_division=0),
        'F1 Score (Macro)': 'f1_macro'
    }

    # Itera sobre cada classificador fornecido
    for nome_modelo, base in classificadores.items():
        print(f"\nProcessando modelo: {nome_modelo}")
        # Cria uma nova instância do modelo base
        modelo = base.__class__(**base.get_params())
        try:
            modelo.set_params(random_state=42)
        except ValueError:
            pass

        # --- Avaliação SEM OTIMIZAÇÃO ---
        print(f"  Avaliando {nome_modelo} (sem otimização)...")
        y_prob = np.nan # Inicializa y_prob para o modelo base
        try:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            # Tenta obter scores/probabilidades do modelo base
            if hasattr(modelo, 'predict_proba'):
                y_prob = modelo.predict_proba(X_test)[:, 1]
            elif hasattr(modelo, 'decision_function'):
                y_prob = modelo.decision_function(X_test)
            # else: y_prob continua NaN

            # Calcula métricas no teste
            sem_metrics_test = {'Modelo': f"{nome_modelo} Sem Otimizacao"}

            sem_metrics_test['Acurácia'] = accuracy_score(y_test, y_pred)
            sem_metrics_test['Precisão(0)'] = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            sem_metrics_test['Precisão(1)'] = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            sem_metrics_test['Recall(0)'] = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            sem_metrics_test['Recall(1)'] = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            sem_metrics_test['F1 Score (Reprovado)'] = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
            sem_metrics_test['F1 Score (Macro)'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            if not np.isnan(y_prob).all():
                 try:
                     sem_metrics_test['AUC ROC'] = roc_auc_score(y_test, y_prob)
                 except ValueError as e_auc:
                     print(f"    Aviso (AUC Teste, Sem Otim): Não foi possível calcular AUC ROC para {nome_modelo}. Erro: {e_auc}")
                     sem_metrics_test['AUC ROC'] = np.nan
            else:
                 sem_metrics_test['AUC ROC'] = np.nan

            for key in sem_metrics_test:
                if isinstance(sem_metrics_test[key], (float, np.floating)):
                    sem_metrics_test[key] = round(sem_metrics_test[key], 3)
            metrics_list.append(sem_metrics_test)

            # Calcula métricas via CV
            sem_metrics_cv = {'Modelo': sem_metrics_test['Modelo']}
            for m_nome, scorer in scorers.items():
                try:
                    cv_score = cross_val_score(modelo, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1).mean()
                    sem_metrics_cv[f'Validação Cruzada ({m_nome})'] = round(cv_score, 3)
                except Exception as e_cv:
                    print(f"    Aviso (CV {m_nome}, Sem Otim): Falha ao calcular CV para {nome_modelo}. Erro: {e_cv}")
                    sem_metrics_cv[f'Validação Cruzada ({m_nome})'] = np.nan
            cv_metrics_list.append(sem_metrics_cv)

        except Exception as e_fit:
            print(f"  ERRO ao treinar/avaliar {nome_modelo} (sem otimização): {e_fit}")
            # Adiciona NaNs aos resultados
            metrics_list.append({'Modelo': f"{nome_modelo} Sem Otimizacao", **{k: np.nan for k in scorers}, 'AUC ROC': np.nan})
            cv_metrics_list.append({'Modelo': f"{nome_modelo} Sem Otimizacao", **{f'Validação Cruzada ({k})': np.nan for k in scorers}})
            # Pula para o próximo modelo se o base falhar
            continue # Importante para não tentar otimizar um modelo que falhou no básico


        # --- Avaliação COM OTIMIZAÇÃO (GridSearchCV) ---
        params = (param_spaces or {}).get(nome_modelo, {})
        grid_performed = False # Flag para saber se a otimização foi feita
        y_prob_opt = np.nan # Inicializa y_prob_opt

        if params:
            print(f"  Otimizando {nome_modelo} com GridSearchCV...")
            try:
                estimator_gs = base.__class__(**base.get_params())
                try:
                    estimator_gs.set_params(random_state=42)
                except ValueError:
                    pass

                grid = GridSearchCV(estimator=estimator_gs, param_grid=params, cv=cv, scoring='f1_macro', n_jobs=-1)
                grid.fit(X_train, y_train)
                best = grid.best_estimator_
                best_params_list.append({'Modelo': nome_modelo, 'Melhores Parâmetros': grid.best_params_})
                print(f"    Melhores Parâmetros: {grid.best_params_}")
                grid_performed = True # Otimização realizada com sucesso

                # Avalia o modelo otimizado no teste
                y_pred_opt = best.predict(X_test)
                if hasattr(best, 'predict_proba'):
                    y_prob_opt = best.predict_proba(X_test)[:, 1]
                elif hasattr(best, 'decision_function'):
                    y_prob_opt = best.decision_function(X_test)
                # else: y_prob_opt continua NaN

                # Calcula métricas no teste para o modelo otimizado
                com_metrics_test = {'Modelo': f"{nome_modelo} Com Otimizacao"}

                com_metrics_test['Acurácia'] = accuracy_score(y_test, y_pred_opt)
                com_metrics_test['Precisão(0)'] = precision_score(y_test, y_pred_opt, pos_label=0, zero_division=0)
                com_metrics_test['Precisão(1)'] = precision_score(y_test, y_pred_opt, pos_label=1, zero_division=0)
                com_metrics_test['Recall(0)'] = recall_score(y_test, y_pred_opt, pos_label=0, zero_division=0)
                com_metrics_test['Recall(1)'] = recall_score(y_test, y_pred_opt, pos_label=1, zero_division=0)
                com_metrics_test['F1 Score (Reprovado)'] = f1_score(y_test, y_pred_opt, pos_label=0, zero_division=0)
                com_metrics_test['F1 Score (Macro)'] = f1_score(y_test, y_pred_opt, average='macro', zero_division=0)
                
                if not np.isnan(y_prob_opt).all():
                    try:
                        com_metrics_test['AUC ROC'] = roc_auc_score(y_test, y_prob_opt)
                    except ValueError as e_auc_opt:
                         print(f"    Aviso (AUC Teste, Com Otim): Não foi possível calcular AUC ROC para {nome_modelo}. Erro: {e_auc_opt}")
                         com_metrics_test['AUC ROC'] = np.nan
                else:
                     com_metrics_test['AUC ROC'] = np.nan

                for key in com_metrics_test:
                    if isinstance(com_metrics_test[key], (float, np.floating)):
                        com_metrics_test[key] = round(com_metrics_test[key], 3)
                metrics_list.append(com_metrics_test)

                # Calcula métricas via CV para o modelo otimizado
                com_metrics_cv = {'Modelo': com_metrics_test['Modelo']}
                for m_nome, scorer in scorers.items():
                     try:
                         cv_score_opt = cross_val_score(best, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1).mean()
                         com_metrics_cv[f'Validação Cruzada ({m_nome})'] = round(cv_score_opt, 3)
                     except Exception as e_cv_opt:
                         print(f"    Aviso (CV {m_nome}, Com Otim): Falha ao calcular CV para {nome_modelo} otimizado. Erro: {e_cv_opt}")
                         com_metrics_cv[f'Validação Cruzada ({m_nome})'] = np.nan
                cv_metrics_list.append(com_metrics_cv)

            except Exception as e_gs:
                print(f"  ERRO durante GridSearchCV/avaliação de {nome_modelo} (com otimização): {e_gs}")
                # Adiciona NaNs aos resultados
                metrics_list.append({'Modelo': f"{nome_modelo} Com Otimizacao", **{k: np.nan for k in scorers}, 'AUC ROC': np.nan})
                cv_metrics_list.append({'Modelo': f"{nome_modelo} Com Otimizacao", **{f'Validação Cruzada ({k})': np.nan for k in scorers}})
                best_params_list.append({'Modelo': nome_modelo, 'Melhores Parâmetros': f'Erro no GridSearchCV: {e_gs}'})
        else:
            # Se não há params, adiciona entrada indicando que não houve otimização
             best_params_list.append({'Modelo': nome_modelo, 'Melhores Parâmetros': 'N/A (sem otimização)'})

            # ----Plotar Matrizes de Confusão ===
            # Condições: Otimização realizada E predições disponíveis para ambos

        # --- Plotagem das Curvas ROC e PR ===
        
        # Condições para plotar:
        # 1. Otimização foi realizada (grid_performed = True)
        # 2. Scores/probabilidades válidos para ambos os modelos (base e otimizado)

        if grid_performed and not np.isnan(y_prob).all() and not np.isnan(y_prob_opt).all():
            try:
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))

                # --- Curva ROC ---
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fpr_opt, tpr_opt, _ = roc_curve(y_test, y_prob_opt)
                axs[0, 0].plot(fpr, tpr, label=f"Base (AUC = {auc(fpr, tpr):.3f})", color=cor_0)
                axs[0, 0].plot(fpr_opt, tpr_opt, label=f"Otimizado (AUC = {auc(fpr_opt, tpr_opt):.3f})", color=cor_1)
                axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
                axs[0, 0].set_title("Curva ROC")
                axs[0, 0].set_xlabel("FPR")
                axs[0, 0].set_ylabel("TPR")
                axs[0, 0].legend(fontsize=8)
                axs[0, 0].grid(True)

                # --- Curva Precision-Recall ---
                prec, rec, _ = precision_recall_curve(y_test, y_prob)
                prec_opt, rec_opt, _ = precision_recall_curve(y_test, y_prob_opt)
                baseline = np.mean(y_test)
                axs[0, 1].plot(rec, prec, label=f"Base (AP = {auc(rec, prec):.3f})", color=cor_0)
                axs[0, 1].plot(rec_opt, prec_opt, label=f"Otimizado (AP = {auc(rec_opt, prec_opt):.3f})", color=cor_1)
                axs[0, 1].axhline(baseline, ls='--', color='k', alpha=0.6)
                axs[0, 1].set_title("Curva Precision-Recall")
                axs[0, 1].set_xlabel("Recall")
                axs[0, 1].set_ylabel("Precisão")
                axs[0, 1].legend(fontsize=8)
                axs[0, 1].grid(True)

                # --- Matriz de Confusão Base ---
                cm_base = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm_base, annot=True, fmt='d',
                            cmap=sns.light_palette(cor_0, as_cmap=True), ax=axs[1, 0],
                            xticklabels=['Reprovado', 'Aprovado'],
                            yticklabels=['Reprovado', 'Aprovado'])
                axs[1, 0].set_title("Matriz Base")
                axs[1, 0].set_xlabel("Predito")
                axs[1, 0].set_ylabel("Verdadeiro")

                # --- Matriz de Confusão Otimizado ---
                cm_opt = confusion_matrix(y_test, y_pred_opt)
                sns.heatmap(cm_opt, annot=True, fmt='d',
                            cmap=sns.light_palette(cor_1, as_cmap=True), ax=axs[1, 1],
                            xticklabels=['Reprovado', 'Aprovado'],
                            yticklabels=['Reprovado', 'Aprovado'])
                axs[1, 1].set_title("Matriz Otimizada")
                axs[1, 1].set_xlabel("Predito")
                axs[1, 1].set_ylabel("Verdadeiro")

                plt.suptitle(f"Comparativo de Desempenho - {nome_modelo} ({formatar_titulo(materia)})", fontsize=11)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                if salvar:
                    nome_fig = f"curvas_confusao_{nome_modelo}_{materia}.png"
                    salvar_figura(nome_fig, materia)

                plt.show()

            except Exception as e_plot:
                print(f"Erro ao gerar gráficos comparativos para {nome_modelo}: {e_plot}")

        elif not grid_performed and not np.isnan(y_prob).all():
            print("  Gerando curvas ROC, PR e matriz de confusão (modelo base)...")
            try:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))  # 3 colunas

                # === Curva ROC ===
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_base = auc(fpr, tpr)
                axs[0].plot(fpr, tpr, label=f"AUC = {auc_base:.3f}", color=cor_0, lw=2)
                axs[0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
                axs[0].set_title("Curva ROC")
                axs[0].set_xlabel("FPR")
                axs[0].set_ylabel("TPR")
                axs[0].legend()
                axs[0].grid(True)

                # === Curva Precision-Recall ===
                prec, rec, _ = precision_recall_curve(y_test, y_prob)
                ap_base = auc(rec, prec)
                baseline = np.sum(y_test == 1) / len(y_test)
                axs[1].plot(rec, prec, label=f"AP = {ap_base:.3f}", color=cor_0, lw=2)
                axs[1].axhline(baseline, ls='--', color='k', alpha=0.6, label=f'Baseline (AP={baseline:.3f})')
                axs[1].set_title("Curva Precision-Recall")
                axs[1].set_xlabel("Recall")
                axs[1].set_ylabel("Precisão")
                axs[1].legend()
                axs[1].grid(True)

                # === Matriz de Confusão ===
                cm = confusion_matrix(y_test, y_pred)
                labels = ['Reprovado', 'Aprovado']
                cmap_sem_otim = sns.light_palette(cor_0, as_cmap=True)
                sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_sem_otim, ax=axs[2],
                            xticklabels=labels, yticklabels=labels, annot_kws={"size": 10}, cbar=False)
                axs[2].set_title("Matriz de Confusão")
                axs[2].set_ylabel("Verdadeiro")
                axs[2].set_xlabel("Predito")

                # Título geral
                plt.suptitle(f"{nome_modelo} - Avaliação de Modelo - ({formatar_titulo(materia)})", fontsize=12)
                plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Layout ajustado

                if salvar:
                    nome_arquivo = f"curvas_e_matriz_{nome_modelo}_{materia}.png"
                    salvar_figura(nome_arquivo=nome_arquivo, materia=materia, diretorio='curvas_comparativas_models')

                plt.show()

            except Exception as e_plot_base:
                print(f"  Erro ao gerar visualizações do modelo base: {e_plot_base}")


        else: # Se houve erro na otimização ou no cálculo das probs
             print("  Não foi possível gerar scores/probabilidades para ambos os modelos. Pulando curvas comparativas.")

    # Converte listas de resultados em DataFrames
    metrics_df = pd.DataFrame(metrics_list)
    cv_metrics_df = pd.DataFrame(cv_metrics_list)
    best_params_df = pd.DataFrame(best_params_list)

    return metrics_df, cv_metrics_df, best_params_df

# ==============================================================================
# ==================== SEÇÃO: DIAGNÓSTICO DE OVERFITTING =======================
# ==============================================================================

# ... (Código da função verificar_overfitting como antes) ...
def verificar_overfitting(df_teste, df_cv, limite_diferenca=0.10):
    """Compara métricas de teste e CV para diagnosticar overfitting/underfitting.

    Calcula a diferença percentual relativa `(Teste - CV) / CV` para várias
    métricas. Sinaliza "Overfitting Potencial" se Teste for >10% melhor que CV,
    e "Underfitting Potencial / Teste Ruim" se CV for >10% melhor que Teste.

    Args:
        df_teste (pd.DataFrame): DataFrame com métricas do teste (coluna 'Modelo').
        df_cv (pd.DataFrame): DataFrame com métricas da CV (coluna 'Modelo' e
            colunas "Validação Cruzada ({metrica})").
        limite_diferenca (float, optional): Limiar da diferença relativa para
            sinalizar overfitting/underfitting. Default 0.10 (10%).

    Returns:
        pd.DataFrame: DataFrame com diagnóstico por modelo, incluindo as
        diferenças percentuais por métrica e um diagnóstico geral.
    """
    # Métricas padrão a serem comparadas
    metricas = [
        "Acurácia", "Precisão(1)", "Precisão(0)",
        "Recall(1)",    "Recall(0)",
        "F1 Score (Reprovado)", "F1 Score (Macro)", "AUC ROC"
    ]
    # Mapeia nome base da métrica para nome da coluna em df_cv
    col_cv = {m: f"Validação Cruzada ({m})" for m in metricas}

    # Garante indexação por 'Modelo'
    try:
        df_t = df_teste.set_index("Modelo")
        df_c = df_cv.set_index("Modelo")
    except KeyError:
        raise ValueError("DataFrames `df_teste` e `df_cv` devem conter a coluna 'Modelo'.")

    resultados = []
    # Itera sobre modelos comuns a ambos os DataFrames
    modelos_comuns = df_t.index.intersection(df_c.index)

    for m in modelos_comuns:
        t_row, c_row = df_t.loc[m], df_c.loc[m] # Linhas do teste e CV para o modelo
        diffs_relativas = [] # Lista para calcular a média da diferença
        res_modelo = {"Modelo": m} # Dicionário para armazenar resultados deste modelo

        # Calcula diferença para cada métrica
        for met in metricas:
            col_cv_nome = col_cv.get(met) # Nome da coluna CV correspondente
            diff_rel = np.nan # Default como NaN

            # Verifica se a métrica existe em ambos os DFs
            if met in t_row and col_cv_nome in c_row:
                vt = t_row[met] # Valor Teste
                vc = c_row[col_cv_nome] # Valor CV

                # Calcula diferença relativa (Teste - CV) / CV se vc não for NaN ou 0
                if not pd.isna(vt) and not pd.isna(vc):
                    if vc != 0:
                        diff_rel = (vt - vc) / vc
                    elif vt != 0: # vc é 0, vt não é
                        diff_rel = np.inf if vt > 0 else -np.inf
                    # else: vt e vc são 0, diff_rel continua NaN (ou poderia ser 0)

                # Armazena a diferença formatada como string percentual
                res_modelo[f"Δ {met}"] = f"{100 * diff_rel:.1f}%" if not pd.isna(diff_rel) else "N/A"
                # Adiciona à lista para média apenas se for finita
                if not pd.isna(diff_rel) and np.isfinite(diff_rel):
                    diffs_relativas.append(diff_rel)
            else:
                # Marca como N/A se a métrica estiver faltando
                res_modelo[f"Δ {met}"] = "N/A (métrica ausente)"

        # Calcula média das diferenças relativas finitas
        media_diff = np.mean(diffs_relativas) if diffs_relativas else np.nan
        res_modelo["Média Δ (%)"] = f"{100 * media_diff:.1f}%" if not pd.isna(media_diff) else "N/A"

        # Define diagnóstico baseado na média da diferença
        if not pd.isna(media_diff) and np.isfinite(media_diff):
            if media_diff > limite_diferenca: # Teste > CV em média
                res_modelo["Diagnóstico"] = "Overfitting Potencial"
            elif media_diff < -limite_diferenca: # CV > Teste em média
                res_modelo["Diagnóstico"] = "Underfitting Potencial / Teste Ruim"
            else: # Diferença dentro do limite
                res_modelo["Diagnóstico"] = "OK"
        else: # Média não pôde ser calculada
            res_modelo["Diagnóstico"] = "N/A"

        resultados.append(res_modelo) # Adiciona resultados do modelo à lista

    # Retorna DataFrame com os diagnósticos
    return pd.DataFrame(resultados)


# ==============================================================================
# ==================== SEÇÃO: COMPARAÇÃO DE MODELOS ============================
# ==============================================================================

# ... (Código da função comparar_resultados_classificacao como antes) ...
def comparar_resultados_classificacao(
    df_test, df_cv, metrics=None, materia='portugues', salvar=False
):
    """Compara e visualiza métricas de teste vs. validação cruzada para modelos.

    Gera um gráfico de barras (Teste) com pontos sobrepostos (CV) para comparar
    o desempenho de diferentes modelos em várias métricas.

    Args:
        df_test (pd.DataFrame): DataFrame com métricas do teste (coluna 'Modelo').
        df_cv (pd.DataFrame): DataFrame com métricas da CV (coluna 'Modelo' e
            colunas "Validação Cruzada ({metrica})").
        metrics (Optional[Union[str, List[str]]], optional): Nome(s) da(s) métrica(s)
            base a comparar. Se None, usa todas as colunas numéricas de df_test
            (exceto 'Modelo'). Default None.
        materia (str, optional): Usado no título do gráfico. Default 'portugues'.
        salvar (bool, optional): Se True, tenta salvar o gráfico e a tabela LaTeX
            (requer funções de `documentar_resultados`). Default False.

    Returns:
        pd.DataFrame: DataFrame em formato longo com colunas: 'Modelo', 'Métrica',
                      'Teste', 'CV', 'Diferença (%)' (calculada como (CV-Teste)*100).
    """
    import re # Importação local para a regex

    # --- 1. Normaliza e Valida Métricas ---
    model_col_name = 'Modelo' # Assume que a coluna se chama 'Modelo'
    if model_col_name not in df_test.columns or model_col_name not in df_cv.columns:
        raise ValueError("Coluna 'Modelo' não encontrada em df_test ou df_cv.")

    if isinstance(metrics, str):
        metrics_list = [metrics]
    elif metrics is None:
        # Seleciona todas as colunas numéricas exceto 'Modelo'
        metrics_list = df_test.select_dtypes(include=np.number).columns.tolist()
        # metrics_list = [c for c in df_test.columns if c != model_col_name and pd.api.types.is_numeric_dtype(df_test[c])]
    else:
        metrics_list = list(metrics) # Garante que seja lista

    if not metrics_list:
        raise ValueError("Nenhuma métrica especificada ou encontrada para comparar.")

    # Verifica se todas as métricas existem em df_test
    missing_in_test = [m for m in metrics_list if m not in df_test.columns]
    if missing_in_test:
        raise ValueError(f"Métricas não encontradas em df_test: {missing_in_test}")

    # --- 2. Mapeia Colunas de CV ---
    cv_pattern = re.compile(r'^Validação Cruzada \((.+)\)$')
    cv_col_map = {match.group(1): col_cv
                  for col_cv in df_cv.columns if (match := cv_pattern.match(col_cv))}

    # Verifica se há colunas CV correspondentes para todas as métricas
    missing_in_cv = [m for m in metrics_list if m not in cv_col_map]
    if missing_in_cv:
        raise ValueError(f"Colunas CV correspondentes não encontradas para: {missing_in_cv}. "
                         f"Esperado formato 'Validação Cruzada (...)'.")

    # --- 3. Prepara DataFrames para Merge ---
    df_t_sel = df_test[[model_col_name] + metrics_list].copy()
    cv_cols_needed = [model_col_name] + [cv_col_map[m] for m in metrics_list]
    df_cv_sel = df_cv[cv_cols_needed].copy()

    # Renomeia colunas CV para facilitar o formato longo depois
    rename_map_cv = {model_col_name: model_col_name}
    for m in metrics_list:
        rename_map_cv[cv_col_map[m]] = f"{m}_CV"
    df_cv_renamed = df_cv_sel.rename(columns=rename_map_cv)

    # --- 4. Merge e Transformação para Formato Longo ---
    df_merged = pd.merge(df_t_sel, df_cv_renamed, on=model_col_name, how='inner')
    if df_merged.empty:
        print("Aviso: Nenhum modelo em comum encontrado entre df_test e df_cv.")
        return pd.DataFrame() # Retorna DF vazio

    # Converte para formato longo
    records = []
    for _, row in df_merged.iterrows():
        for m in metrics_list:
            teste_val = row[m]
            cv_val = row[f"{m}_CV"]
            diff_perc = (cv_val - teste_val) * 100 # Diferença absoluta * 100
            records.append({
                'Modelo': row[model_col_name],
                'Métrica': m,
                'Teste': teste_val,
                'CV': cv_val,
                'Diferença (%)': diff_perc # Nome da coluna como no original
            })
    df_comp = pd.DataFrame(records)

    # --- 5. Plotagem ---
    if not df_comp.empty:
        n_modelos = df_comp['Modelo'].nunique()
        # Tenta obter paleta de cores customizada
        try:
            # Usa a paleta padrão do módulo eda_functions
            palette = aplicar_estilo_visual('blue_to_green', n=n_modelos if n_modelos > 0 else 1)
        except NameError:
            print("Aviso: Função 'aplicar_estilo_visual' não encontrada. Usando paleta padrão.")
            palette = None # Usa paleta padrão do seaborn

        fig, ax = plt.subplots(figsize=(8, 5)) # Ajusta largura

        # Plota barras para os resultados do Teste
        sns.barplot(data=df_comp, x='Métrica', y='Teste', hue='Modelo',
                    palette=palette, ax=ax) # ci=None é padrão recente

        # Sobrepõe pontos para os resultados da CV
        sns.pointplot(data=df_comp, x='Métrica', y='CV', hue='Modelo',
                      palette=palette, markers='D', linestyles='--',
                      dodge=True, ax=ax, errorbar=None) # errorbar=None substitui ci=None

        # Ajustes do gráfico
        ax.set_title(f'Métricas Teste vs CV - {materia.capitalize()}', fontsize=14)
        ax.set_ylabel('Valor da Métrica')
        ax.tick_params() # Rotação leve para melhor leitura
        # Move a legenda para fora
        ax.legend(title='Modelo', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta para caber legenda

        # --- 6. Salvamento (Opcional) ---
        if salvar:
            output_dir_comp = 'comparacao_modelos' # Define diretório
            if not os.path.exists(output_dir_comp):
                os.makedirs(output_dir_comp)

            # Salva gráfico
            plot_filename = f"comparacao_metricas_{materia.lower()}.png"
            full_plot_path = os.path.join(output_dir_comp, plot_filename)
            try:
                # Tenta usar salvar_figura se disponível
                # Precisa garantir que titulo_para_snake_case esteja disponível ou usar nome simples
                salvar_figura(nome_arquivo=f"comparacao_metricas_{materia.lower()}",
                              materia=None, # Nome já inclui matéria
                              diretorio=output_dir_comp)
            except NameError:
                 # Fallback para plt.savefig
                 plt.savefig(full_plot_path, dpi=300, bbox_inches='tight')
                 print(f"Gráfico salvo em: {full_plot_path} (salvar_figura não encontrada).")

            # Salva tabela LaTeX
            latex_filename = f"tabela_comparacao_{materia.lower()}"
            try:
                # Tenta usar exportar_df_para_latex se disponível
                exportar_df_para_latex(
                    df_comp,
                    nome_tabela=latex_filename,
                    caminho_pasta=output_dir_comp,
                    caption=f"Comparação Teste vs CV ({materia.capitalize()})"
                )
            except NameError:
                print("Aviso: Função 'exportar_df_para_latex' não encontrada. Tabela LaTeX não salva.")

        plt.show() # Exibe o gráfico

    else:
        print("DataFrame de comparação vazio. Nenhum gráfico gerado.")

    return df_comp

