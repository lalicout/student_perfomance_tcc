# ======================================
# Módulo: eda_functions.py
# ======================================
"""Análise exploratória de dados de desempenho escolar.

Este módulo oferece um conjunto de ferramentas para realizar a Análise Exploratória
de Dados (EDA) em conjuntos de dados relacionados ao desempenho de estudantes.
As funcionalidades incluem a geração de diversas visualizações (como boxplots,
heatmaps e gráficos de dispersão), cálculo de estatísticas descritivas,
detecção e análise de outliers, e a identificação de padrões e diferenças
significativas entre grupos de alunos.

O módulo visa facilitar a compreensão inicial dos dados, a formulação de
hipóteses e a preparação para etapas subsequentes de modelagem preditiva.
Ele também integra funcionalidades para a padronização visual dos gráficos e
salvamento dos resultados.

Principais Funcionalidades:
    - Criação e personalização de estilos visuais para gráficos.
    - Funções para formatação de texto em visualizações.
    - Geração de gráficos para análise de distribuições univariadas e bivariadas.
    - Ferramentas para análise de correlação e comparação entre categorias.
    - Métodos para sumarização estatística e identificação de outliers.
    - Análise comparativa de grupos de alunos com desempenhos extremos.

Este módulo é primariamente utilizado na fase de Entendimento dos Dados (Data
Understanding) do ciclo de vida de projetos de Data Science, como o CRISP-DM.

Funções disponíveis:
    quebrar_rotulo(texto: str, max_palavras: int = 1) → str
        Quebra rótulos longos em múltiplas linhas para melhorar a visualização.

    formatar_titulo(texto: str) → str
        Formata rótulos e títulos aplicando capitalização e correções de acentuação.

    plot_distribuicao(df, notas, paleta, materia=None, nome_arquivo='boxplot_notas', mostrar_media=True)
        Gera boxplots para variáveis quantitativas de notas.

    custom_heatmap(matriz_corr, cores, titulo, n_arq, disciplina)
        Cria e salva um mapa de calor baseado em uma matriz de correlação.

    graficos_desempenho_escolar_por_categoria(df, paleta, coluna, nome_arquivo, diretorio, mat=None)
        Gera gráficos de desempenho escolar organizados por categorias.

    plot_notas_faltas(df, cor, dir, mat)
        Gera gráficos de dispersão para atributos quantitativos.

    resumir_outliers(df) → pd.DataFrame
        Calcula e exibe estatísticas de outliers por coluna numérica.

    perfil_categorico_outliers(df_outliers, df_total, variaveis_categoricas) → dict
        Compara a distribuição de categorias nos outliers com a base total.

    identificar_extremos_comparaveis(df, variavel_numerica, variaveis_categoricas, entrada=None, min_diferenca=0.15, q_limite=0.25) → tuple
        Identifica categorias com diferenças significativas entre grupos extremos.

    plot_top_diferencas_extremos(df_diferencas, materia, q1_lim, q3_lim, n_baixo, n_alto, top_n=10, diretorio='graficos_diferencas_perfil', salvar=True)
        Plota as categorias com maior diferença entre grupos de desempenho (baixo vs alto).

    comparar_materias(df, coluna_categorica, colunas_quantitativas, titulo_base, pasta_destino, cores=None, show_plot=True)
        Gera boxplots comparativos para variáveis categóricas e quantitativas.

"""

# ==============================================================================
# ========================== IMPORTAÇÃO DE BIBLIOTECAS =========================
# ==============================================================================

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_hex
import statsmodels.api as sm
from IPython.display import display

from documentar_resultados import salvar_figura

from IPython.display import display

# ==============================================================================
# ============================ FUNÇÕES AUXILIARES ==============================
# ==============================================================================

# ------------------------------------------------------------------------------
# Seção: Estilo Visual e Padronização de Figuras
# ------------------------------------------------------------------------------

def aplicar_estilo_visual(paleta, retornar_cmap=False, n=None):
    """Obtém ou gera uma lista de cores ou um colormap a partir de uma paleta.

    Recupera cores de paletas predefinidas ('azul', 'verde', 'blue_to_green',
    'vermelho'e 'laranja') ou usa uma lista de cores fornecida. 
    Pode retornar um colormap Matplotlib ou uma lista interpolada de 'n' cores.

    Args:
        paleta (Union[str, List[str]]): Nome da paleta predefinida
            (e.g., 'azul', 'verde') ou uma lista de códigos hexadecimais.
        retornar_cmap (bool, optional): Se True, retorna um objeto
            `LinearSegmentedColormap`. Default é False.
        n (Optional[int], optional): Número de cores discretas desejado.
            Se fornecido e `retornar_cmap` for False, retorna `n` cores
            interpoladas. Se None, retorna as cores base da paleta.
            Default é None.

    Returns:
        Union[List[str], matplotlib.colors.LinearSegmentedColormap]: Lista de
        cores hexadecimais ou um objeto Colormap.
    """
    paletas_predefinidas = {
        'azul': {
            'cores': ["#AED6F1", "#5DADE2", "#3498DB", "#2E86C1", "#1B4F72"],
            'auxiliar': ['#d9ecf2']
        },
        'verde': {
        'cores': ['#9aafa0', '#639374', '#3b7850', '#225639', '#153024'],
        'auxiliar': ['#EAF6EE']
        },
        'blue_to_green': {
            'cores': [
                '#d7e2e2', '#b1c6c5', '#8baaaa', '#65908f',
                '#3e7575', '#0a5c5c', '#074040'
            ]

        ,
            'auxiliar': []
        }
        ,
        'vermelho': {
            'cores': ['#fca082', '#fb7c5c', '#a81434'],  # o último é o extraído da matriz
            'auxiliar': ['#fde0d9']
        },
        'laranja': {
            'cores': ['#fee3c8', '#fdc692', '#d9711c'],  # o último é o extraído da matriz
            'auxiliar': ['#fff4e5']
        }
    }



    # Determina as cores base e auxiliares
    if isinstance(paleta, str) and paleta in paletas_predefinidas:
        base = paletas_predefinidas[paleta]
        cores_base = base['cores']
        cor_auxiliar = base['auxiliar']
    elif isinstance(paleta, list):
        cores_base = paleta
        cor_auxiliar = [] # Sem cor auxiliar para paletas customizadas
    else:
        # Considerar levantar um erro para tipo inválido de 'paleta'
        raise ValueError("O parâmetro 'paleta' deve ser str ou list.")

    # Retorna colormap se solicitado
    if retornar_cmap:
        if paleta == paletas_predefinidas['blue_to_green']:
            return LinearSegmentedColormap.from_list("custom_cmap", cores_base[len(paletas_predefinidas['blue_to_green'])] + '#65908f')
        else:
            return LinearSegmentedColormap.from_list("custom_cmap", cor_auxiliar + cores_base)

    # Retorna cores base se 'n' não for especificado
    if n is None:
        return cores_base

    # Cria colormap temporário para interpolação
    cmap_aux = aplicar_estilo_visual(paleta, retornar_cmap=True)

    # Retorna 'n' cores interpoladas
    if n > 1:
        return [to_hex(cmap_aux(i / (n - 1))) for i in range(n)]
    elif n == 1:
        # Retorna a terceira cor da paleta base para n=1 (comportamento específico)
        return [cores_base[2]]
    else: # n <= 0
        return [] # Retorna lista vazia para n inválido

def dimensionar_figuresize(n_colunas, n_linhas, largura_base=6, altura_base=6.2,
                           largura_maxima=None, altura_maxima=None, modo='relatorio'):
    """Calcula as dimensões ideais de uma figura Matplotlib.

    Determina largura e altura com base no número de subplots, dimensões
    base por subplot e limites máximos opcionais, ajustando por 'modo'.

    Args:
        n_colunas (int): Número de colunas de subplots.
        n_linhas (int): Número de linhas de subplots.
        largura_base (float, optional): Largura base por subplot. Default 6.
        altura_base (float, optional): Altura base por subplot. Default 6.2.
        largura_maxima (Optional[float], optional): Largura máxima da figura.
            Default depende do `modo`.
        altura_maxima (Optional[float], optional): Altura máxima da figura.
            Default depende do `modo`.
        modo (str, optional): Modo ('relatorio' ou outro) que define os
            máximos padrão. Default 'relatorio'.

    Returns:
        Tuple[float, float]: Tupla (largura_final, altura_final) calculada.
    """
    # Define limites máximos padrão com base no modo
    if largura_maxima is None:
        largura_maxima = 6.3 if modo == 'relatorio' else 8.0
    if altura_maxima is None:
        altura_maxima = 4.5 if modo == 'relatorio' else 6.0

    # Calcula dimensões naturais (sem limites)
    largura_natural = largura_base * n_colunas
    altura_natural = altura_base * n_linhas

    # Calcula fator de escala para caber nos limites (pega a escala mais restritiva)
    escala_final = min(
        largura_maxima / largura_natural if largura_natural > largura_maxima else 1,
        altura_maxima / altura_natural if altura_natural > altura_maxima else 1
    )

    # Retorna dimensões finais escaladas
    return largura_natural * escala_final, altura_natural * escala_final

def ajustar_fontsize_por_figsize(figwidth, base_width=6):
    """Calcula tamanhos de fonte escalados pela largura da figura.

    Gera um dicionário de tamanhos de fonte para elementos Matplotlib
    (títulos, rótulos, legendas), escalados proporcionalmente à `figwidth`
    em relação a `base_width`, com limites min/max.

    Args:
        figwidth (float): Largura atual da figura.
        base_width (float, optional): Largura de referência para escala. Default 6.

    Returns:
        Dict[str, float]: Dicionário {parametro_rcParams: tamanho_fonte}.
    """
    escala = figwidth / base_width
    # Fatores de escala e limites min/max definidos empiricamente
    return {
        'axes.titlesize': min(11, max(8, 13 * escala)),
        'axes.labelsize': min(10, max(7, 11 * escala)),
        'xtick.labelsize': min(7, max(6, 9 * escala)),
        'ytick.labelsize': min(7, max(6, 9 * escala)),
        'legend.fontsize': min(7, max(6, 9 * escala)),
        'legend.title_fontsize': min(7, max(6, 9 * escala)),
        'figure.titlesize': min(13, max(9, 15 * escala))
    }

def padronizar_figura(n_linhas, n_colunas, new_args=None, include_args_subplot=False,
                      escala=0.8, dinamico=False,salvar=False):
    """Cria uma figura Matplotlib com eixos e estilos padronizados.

    Inicializa figura e eixos (`plt.subplots()`) aplicando tamanhos de figura
    e fonte padronizados. Opera em modo estático (usando dicionários
    `static_sizes`, `static_fonts` e `escala`) ou dinâmico (usando
    `dimensionar_figuresize` e `ajustar_fontsize_por_figsize`).

    Args:
        n_linhas (int): Número de linhas de subplots.
        n_colunas (int): Número de colunas de subplots.
        new_args (Optional[dict], optional): Argumentos adicionais para
            `plt.subplots()`. Default None.
        include_args_subplot (bool, optional): Não utilizado na implementação atual.
            Default False.
        escala (float, optional): Fator de escala para figsize no modo estático.
            Default 0.8.
        dinamico (bool, optional): Se True, usa modo dinâmico. Se False (padrão),
            tenta modo estático e recorre ao dinâmico se layout não definido.
        salvar (bool, optional): Não utilizado na implementação atual. Default False.

    Returns:
        Tuple[matplotlib.figure.Figure, Any, Dict[str, float]]:
            Tupla (fig, axes, font_sizes). `axes` pode ser um único Axes ou
            um array NumPy de Axes.

    Notes:
        - Modifica `plt.rcParams` globalmente para a figura corrente.
        - Parâmetros `include_args_subplot` e `salvar` não têm efeito atualmente.
    """
    # Dicionários para tamanhos e fontes estáticos (para layouts comuns)
    static_sizes = {
        (1, 1): (6.0, 4.5), (1, 2): (6.3, 4.2), (1, 3): (6.3, 3.6),
        (2, 2): (6.3, 7),   (2, 3): (6.3, 5)
    }
    # Fontes estáticas correspondentes (ajustadas empiricamente)
    static_fonts = {
        # ATENÇÃO: tick.labelsize ≤ 5 pode ser necessário visualmente
        (1, 1): {'axes.titlesize': 9, 'axes.labelsize': 7, 'xtick.labelsize': 4,
                 'ytick.labelsize': 4, 'legend.fontsize': 4, 'figure.titlesize': 10,
                 'legend.title_fontsize': 4},
        (1, 2): {'axes.titlesize': 9, 'axes.labelsize': 5, 'xtick.labelsize': 4,
                 'ytick.labelsize': 4, 'legend.fontsize': 4, 'figure.titlesize': 10,
                 'legend.title_fontsize': 7},
        (1, 3): {'axes.titlesize': 9, 'axes.labelsize': 6, 'xtick.labelsize': 4,
                 'ytick.labelsize': 4, 'legend.fontsize': 5, 'figure.titlesize': 10,
                 'legend.title_fontsize': 7},
        (2, 2): {'axes.titlesize': 9, 'axes.labelsize': 6, 'xtick.labelsize': 4,
                 'ytick.labelsize': 4, 'legend.fontsize': 5, 'figure.titlesize': 10,
                 'legend.title_fontsize': 7},
        (2, 3): {'axes.titlesize': 8, 'axes.labelsize': 6, 'xtick.labelsize': 4,
                 'ytick.labelsize': 4, 'legend.fontsize': 5, 'figure.titlesize': 9,
                 'legend.title_fontsize': 7}
    }

    # Garante que new_args seja um dicionário
    current_new_args = new_args or {}

    # Modo Dinâmico
    if dinamico:
        figsize = dimensionar_figuresize(n_colunas=n_colunas, n_linhas=n_linhas)
        font_sizes = ajustar_fontsize_por_figsize(figsize[0])
        # Limita tamanho mínimo dos rótulos dos ticks
        font_sizes['xtick.labelsize'] = min(font_sizes['xtick.labelsize'], 5)
        font_sizes['ytick.labelsize'] = min(font_sizes['ytick.labelsize'], 5)
        fig, axes = plt.subplots(n_linhas, n_colunas, figsize=figsize, dpi=300, **current_new_args)
    # Modo Estático
    else:
        key = (n_linhas, n_colunas)
        # Fallback para dinâmico se layout não estiver pré-definido
        if key not in static_sizes:
            return padronizar_figura(n_linhas, n_colunas, new_args=current_new_args,
                                     include_args_subplot=include_args_subplot, # Não usado
                                     escala=escala, dinamico=True,
                                     salvar=salvar) # Não usado

        # Obtém tamanhos estáticos e aplica escala
        fig_w_base, fig_h_base = static_sizes[key]
        fig_w, fig_h = fig_w_base * escala, fig_h_base * escala
        font_sizes = static_fonts[key]
        fig, axes = plt.subplots(n_linhas, n_colunas, figsize=(fig_w, fig_h), dpi=300, **current_new_args)

    # Aplica tamanhos de fonte calculados aos rcParams
    for key_font, size_font in font_sizes.items():
        plt.rcParams[key_font] = size_font

    return fig, axes, font_sizes

# ------------------------------------------------------------------------------
# Seção: Formatadores de Texto
# ------------------------------------------------------------------------------

def titulo_para_snake_case(texto):
    """Converte uma string de título para o formato snake_case.

    Ideal para gerar nomes de arquivos válidos a partir de títulos.
    Usa `formatar_titulo`, remove pontuação, converte para minúsculas,
    remove espaços extras e substitui espaços por underscores.

    Args:
        texto (str): O título ou string original a ser convertido.

    Returns:
        str: A string convertida para snake_case.
    """
    texto = formatar_titulo(texto) # Aplica formatação inicial (acentos, etc.)
    texto = re.sub(r'[^\w\s]', '', texto) # Remove pontuação (exceto letras, números, espaços)
    texto = texto.lower().strip().replace(" ", "_") # Converte e substitui espaços
    texto = re.sub(r'_+', '_', texto) # Garante que não haja múltiplos underscores
    return texto

def quebrar_rotulo(texto, max_palavras=1):
    """Quebra uma string de texto em duas linhas se exceder um limite de palavras.

    Útil para rótulos de eixos em gráficos. Se o número de palavras for maior
    que `max_palavras`, divide o texto ao meio (por contagem de palavras)
    e insere uma nova linha ('\n').

    Args:
        texto (str): A string original do rótulo.
        max_palavras (int, optional): Limite de palavras para acionar a quebra.
            Não é o máximo por linha após quebra. Default é 1.

    Returns:
        str: O texto original ou quebrado em duas linhas.
    """
    # Normaliza espaços múltiplos e divide em palavras
    palavras = ' '.join(texto.split()).split()

    # Quebra se o número de palavras exceder o limite
    if len(palavras) > max_palavras:
        meio = len(palavras) // 2 # Ponto de divisão
        primeira_linha = ' '.join(palavras[:meio])
        segunda_linha = ' '.join(palavras[meio:])
        return f"{primeira_linha}\n{segunda_linha}"
    return texto

def formatar_titulo(texto):
    """Formata uma string para uso como título ou rótulo em gráficos.

    Expande abreviações de disciplinas, converte para "Title Case" com espaços,
    e aplica correções predefinidas de acentuação e capitalização para termos
    comuns (e.g., "Mae" -> "Mãe").

    Args:
        texto (str): A string original a ser formatada (pode ser snake_case).

    Returns:
        str: A string formatada, pronta para visualização.
    """
    # Dicionário de correções específicas (mantido interno)
    correcoes_titulos = {
        "Portugues": "Português", "Matematica": "Matemática",
        "Mae": "Mãe", "Saude": "Saúde", "Educacao": "Educação",
        "Proximo": "Próximo", "Reputacao": "Reputação",
        "Responsavel": "Responsável", "Area Da Saude": "Área da Saúde",
        "Outra Profissao": "Outra profissão", "Servicos": "Serviços",
        "Professor": "Professor(a)", "Dona De Casa": "Dona de casa",
        "Dono De Casa": "Dono de casa", "Nao": "Não", "Sim": "Sim",
        "Romantico": "Romântico", "Intencao": "Intenção",
        "Reprovacoes": "Reprovações", "Aprovacao": "Aprovação",
        "Reprovacao": "Reprovação"
    }

    # Expande sufixos de matéria
    if isinstance(texto, str): # Garante que texto é string
      if texto.endswith("_por"):
          texto = texto[:-4] + "_portugues"
      elif texto.endswith("_mat"):
          texto = texto[:-4] + "_matematica"

      # Converte para Title Case com espaços
      texto_formatado = texto.replace("_", " ").strip().title()

      # Aplica correções do dicionário
      for errado, certo in correcoes_titulos.items():
          texto_formatado = texto_formatado.replace(errado, certo)
    else:
      # Se não for string, retorna como está (ou lança erro/converte)
      texto_formatado = texto

    return texto_formatado

# ==============================================================================
# ========================= VISUALIZAÇÕES GERAIS ===============================
# ==============================================================================

# ------------------------------------------------------------------------------
# Seção: Boxplots,  ou Boxplots e Countplots Combinados
# ------------------------------------------------------------------------------

def plot_boxplot_countplot(df, x, y, hue=None, materia=None, paleta='blue_to_green', salvar=False, nome_arquivo='box_count'):
    """Gera visualizações combinadas de boxplot e countplot.

    Cria uma figura com dois subplots: um boxplot mostrando a distribuição
    da variável quantitativa `y` agrupada pelas categorias de `x`, e um
    countplot exibindo a frequência de cada categoria em `x`. Ambos os
    gráficos podem ser segmentados pela variável `hue`.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados.
        x (str): Nome da coluna categórica para o eixo x.
        y (str): Nome da coluna quantitativa para o eixo y (boxplot).
        hue (str, optional): Nome da coluna para segmentação por cor (e.g., 'aprovacao'). Defalut None.
        materia (Optional[str], optional): Nome da disciplina/contexto. Default None.
        paleta (str, optional): Nome da paleta de cores. Default 'blue_to_green'.
        salvar (bool, optional): Se True, salva a figura. Default False.
        nome_arquivo (str, optional): Nome base para o arquivo salvo. Default 'box_count'.

    Returns:
        Tuple[plt.Figure, np.ndarray[plt.Axes]]: Tupla (fig, axes).
    """
    # Determina número de cores baseado nas categorias de 'hue'
    num_cores = df[hue].nunique() if hue and df[hue].nunique() > 1 else 2
    cores = aplicar_estilo_visual(paleta, n=num_cores)

    # Cria figura e eixos padronizados
    fig, axes, font_sizes = padronizar_figura(1, 2)
    # plt.rcParams.update(font_sizes) # Aplicação já feita por padronizar_figura

    # --- Subplot 1: Boxplot ---
    sns.boxplot(data=df, x=x, y=y, hue=None, ax=axes[0], palette=cores,
                linewidth=0.8,
                boxprops={'edgecolor': 'black', 'linewidth': 0.8},
                whiskerprops={'color': 'black', 'linewidth': 0.8},
                capprops={'color': 'black', 'linewidth': 0.8},
                medianprops={'color': 'black', 'linewidth': 0.8})

    axes[0].set_title(f"Distribuição de {formatar_titulo(y)}\npor categoria",fontsize=8)
    axes[0].set_xlabel(quebrar_rotulo(formatar_titulo(x))) # Aplica quebra e formatação
    axes[0].set_ylabel(formatar_titulo(y))


    # --- Subplot 2: Countplot ---
    sns.countplot(data=df, x=x, hue=hue, ax=axes[1], palette=cores)
    axes[1].set_title(f"Frequência de Aprovação \npor categoria",fontsize=8)
    axes[1].set_xlabel(quebrar_rotulo(formatar_titulo(x))) # Aplica quebra e formatação
    axes[1].set_ylabel("Contagem")
    if hue:
        axes[1].legend(title=formatar_titulo(hue), loc='upper right')

    # Ajusta rótulos do eixo X para consistência e formatação
    xticks_texts_0 = [tick.get_text() for tick in axes[0].get_xticklabels()]
    # Formata/Quebra rótulos (se não já quebrados)
    xticks_texts_formatados_0 = [quebrar_rotulo(formatar_titulo(tick_text)) if '\n' not in tick_text else tick_text for tick_text in xticks_texts_0]
    axes[0].set_xticklabels(xticks_texts_formatados_0) # Aplica ao boxplot
    axes[1].set_xticklabels(xticks_texts_formatados_0) # Aplica ao countplot

    # Ajusta limite superior do eixo Y do countplot para dar espaço
    ymin, ymax = axes[1].get_ylim()
    axes[1].set_ylim(ymin, ymax + max(5, ymax * 0.05)) # Ajuste dinâmico

    # --- Título e Informações da Figura ---
    titulo_fig = f"Análise de {formatar_titulo(y)} vs {formatar_titulo(x)}"
    if hue:
        titulo_fig += f"\n segmentado por {formatar_titulo(hue)}"
    fig.suptitle(titulo_fig, fontsize=11, y=0.95)

    # Adiciona texto da disciplina se fornecido
    if materia:
        plt.figtext(
            0.5, 0.01, # Posição ajustada
            f"Disciplina: {formatar_titulo(materia)}",
            ha='center', fontsize=font_sizes.get('figure.titlesize', 10) * 0.8, style='italic'
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Ajusta layout

    # Salva figura se solicitado
    if salvar:
        # Cria nome de arquivo mais descritivo
        nome_arquivo_final = f"{nome_arquivo}_{x}"
        if hue:
            nome_arquivo_final += f"_por_{hue}"
        # Usa a função de salvamento importada
        salvar_figura(nome_arquivo_final, materia = materia if materia else '')

    plt.show()
    return fig, axes

def plot_boxplot_boxplot(df, x, y1, y2, paleta, hue=None, materia=None, salvar=False, nome_arquivo='box_box'):
    """Gera dois boxplots lado a lado para comparar distribuições.

    Cria figura com dois subplots: boxplot de `y1` vs `x` e boxplot de `y2`
    vs `x`. Ambos podem ser segmentados por `hue`.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        x (str): Nome da coluna categórica para o eixo x.
        y1 (str): Nome da coluna quantitativa para o primeiro boxplot.
        y2 (str): Nome da coluna quantitativa para o segundo boxplot.
        hue (Optional[str]): Nome da coluna para segmentação por cor. Default None.
        materia (Optional[str]): Nome da disciplina/contexto.
        paleta (str): Nome da paleta de cores.
        salvar (bool, optional): Se True, salva a figura. Default False.
        nome_arquivo (str, optional): Nome base para o arquivo salvo. Default 'box_box'.

    Returns:
        Tuple[plt.Figure, np.ndarray[plt.Axes]]: Tupla (fig, axes).

    Notes:
        A legenda para `hue` (se usada) é adicionada à figura, não aos subplots.
    """
    # Determina número de cores baseado em 'hue'
    num_cores = df[hue].nunique() if hue and df[hue].nunique() > 0 else 2
    cores = aplicar_estilo_visual(paleta, n=num_cores)

    # Cria figura e eixos padronizados
    fig, axes, font_sizes = padronizar_figura(1, 2)

    plot_vars = [y1, y2]
    for i, y_var in enumerate(plot_vars):
        ax = axes[i]
        # --- Cria o Boxplot ---
        sns.boxplot(data=df, x=x, y=y_var, hue=hue, ax=ax, palette=cores,
                    linewidth=0.8,
                    boxprops={'edgecolor': 'black', 'linewidth': 0.8},
                    whiskerprops={'color': 'black', 'linewidth': 0.8},
                    capprops={'color': 'black', 'linewidth': 0.8},
                    medianprops={'color': 'black', 'linewidth': 0.8})

        # Formata rótulos do eixo X
        xticks_texts = [tick.get_text() for tick in ax.get_xticklabels()]
        xticks_texts_formatados = [quebrar_rotulo(formatar_titulo(tick_text)) if '\n' not in tick_text else tick_text for tick_text in xticks_texts]
        ax.set_xticklabels(xticks_texts_formatados)

        # Define títulos e rótulos dos eixos
        ax.set_title(f"Distribuição por {formatar_titulo(x)}", fontsize=8)
        ax.set_xlabel(quebrar_rotulo(formatar_titulo(x)))
        ax.set_ylabel(formatar_titulo(y_var))

        # Gerencia legenda: adiciona à figura no último plot, remove dos subplots
        if hue:
            if i == len(plot_vars) - 1: # Adiciona legenda apenas uma vez
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, title=formatar_titulo(hue), loc='upper right',
                           bbox_to_anchor=(0.99, 0.95), fontsize=font_sizes.get('legend.fontsize'))
            ax.get_legend().remove() # Remove legenda do subplot individual

    # --- Define Título da Figura ---
    colunas_notas = {'nota1', 'nota2', 'nota_final',
                     'nota1_por', 'nota2_por', 'nota_final_por',
                     'nota1_mat', 'nota2_mat', 'nota_final_mat'}

    titulo_principal = ""
    if y1 in colunas_notas and y2 in colunas_notas:
        titulo_principal = f"Comparativo de Notas por {formatar_titulo(x)}"
    else:
        titulo_principal = f"Distribuição de {formatar_titulo(y1)} e {formatar_titulo(y2)} por {formatar_titulo(x)}"

    if materia:
        titulo_principal += f" - {formatar_titulo(materia)}"

    fig.suptitle(titulo_principal)
    # Ajusta layout para acomodar legenda da figura, se houver
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95] if hue else [0, 0.03, 1, 0.95])

    # Salva figura se solicitado
    if salvar:
        nome_arquivo_final = f"{nome_arquivo}_{titulo_para_snake_case(x)}_{titulo_para_snake_case(y1)}_{titulo_para_snake_case(y2)}"
        salvar_figura(nome_arquivo_final, materia=titulo_para_snake_case(materia) if materia else '')

    plt.show()
    return fig, axes



# ------------------------------------------------------------------------------
# Seção: Visualização de Variáveis Quantitativas
# ------------------------------------------------------------------------------

def plot_distribuicao_quantitativas(df, colunas, modo='box', mostrar_media=False, mostrar_mediana=False,
                                     titulo=None, sub_titulo=None, paleta='azul', materia=None,
                                     show_figure=True, salvar=False):
    """
    Visualiza múltiplas variáveis quantitativas com boxplots ou histogramas.

    Gera subplots lado a lado para cada coluna quantitativa especificada,
    usando boxplots (`modo='box'`) ou histogramas (`modo='hist`).

    Args:
        df (pd.DataFrame): Base de dados.
        colunas (List[str]): Lista de colunas numéricas para análise.
        modo (str, optional): 'box' para boxplots ou 'hist' para histogramas.
        mostrar_media (bool): Se True, anota a média no boxplot.
        mostrar_mediana (bool): Se True, desenha linha da mediana no boxplot.
        titulo (str, optional): Título principal do gráfico.
        sub_titulo (str, optional): Subtítulo (exibido abaixo do título principal).
        paleta (str): Nome da paleta de cores ('azul', 'vermelho', 'laranja', etc.).
        materia (str, optional): Nome da disciplina (usado no título).
        show_figure (bool): Se True, exibe a figura.
        salvar (bool): Se True, salva a figura em arquivo.

    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: Figura e lista de eixos.
    """

    n_plots = len(colunas)
    if n_plots == 0:
        print("Nenhuma coluna fornecida para plotar.")
        return None, None

    cor_offset = 0
    n_cores_total = n_plots
    if paleta != 'blue_to_green':
        cor_offset = 2
        n_cores_total += cor_offset

    cores = aplicar_estilo_visual(paleta, n=n_cores_total)

    fig, axes, _ = padronizar_figura(1, n_plots)
    if n_plots == 1:
        axes = [axes]

    for i, col in enumerate(colunas):
        ax = axes[i]
        cor_atual = cores[i + cor_offset]

        if modo == 'box':
            sns.boxplot(data=df, y=col, ax=ax, color=cor_atual,
                        linewidth=0.8,
                        boxprops={'edgecolor': 'black', 'linewidth': 0.8},
                        whiskerprops={'color': 'black', 'linewidth': 0.8},
                        capprops={'color': 'black', 'linewidth': 0.8},
                        medianprops={'color': 'black', 'linewidth': 0.8})

            if mostrar_media:
                media = df[col].mean()
                ax.annotate(f"{media:.2f}", xy=(0, media), xytext=(0.1, media),
                            textcoords='data', ha='left', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", lw=0.5),
                            fontsize=7)

            if mostrar_mediana:
                mediana = df[col].median()
                ax.axhline(mediana, linestyle='--', color='gray', linewidth=0.8)

            ax.set_xlabel("")
            ax.set_ylabel(formatar_titulo(col))

        elif modo == 'hist':
            ymax_hist = df.shape[0] / 3.33
            sns.histplot(data=df, x=col, kde=True, ax=ax, color=cor_atual, bins=20)
            ax.set_xlabel(formatar_titulo(col))
            ax.set_ylabel("Frequência")
            ax.set_ylim(0, ymax_hist)

    if titulo is None:
        if all('nota' in c for c in colunas):
            titulo = "Distribuição das Notas"
        elif all('falta' in c for c in colunas):
            titulo = "Distribuição das Faltas"
        elif 'idade' in colunas:
            titulo = "Distribuição de Idade e Faltas"
        else:
            titulo = "Distribuições Quantitativas"

    if materia:
        titulo += f" - {formatar_titulo(materia)}"

    fig.suptitle(titulo, fontsize=11, fontweight='bold')
    if sub_titulo:
        fig.text(0.5, 0.93, sub_titulo, ha='center', fontsize=9)

    fig.tight_layout()

    if salvar:
        nome_arquivo_final = f"{titulo_para_snake_case(titulo)}_{modo}"
        salvar_figura(nome_arquivo_final)

    if show_figure:
        plt.show()

    return fig, axes


# ------------------------------------------------------------------------------
# Seção: Mapas de Calor (Heatmaps)
# ------------------------------------------------------------------------------

def custom_heatmap(matriz_corr, cores, titulo, n_arq=None, disciplina=None, salvar = False):
    """Gera e opcionalmente salva um mapa de calor (heatmap).

    Cria um heatmap a partir de uma matriz de correlação (ou similar),
    usando uma paleta de cores customizada e formatação padronizada.

    Args:
        matriz_corr (pd.DataFrame): Matriz de correlação ou dados 2D.
        cores (Union[str, list]): Nome da paleta ('azul', 'verde', etc.) ou
            lista de cores hexadecimais.
        titulo (str): Título principal do gráfico.
        n_arq (Optional[str], optional): Nome base para o arquivo salvo. Se None,
            gerado a partir do `titulo`. Default None.
        disciplina (Optional[str], optional): Nome da disciplina/contexto, usado
            no subtítulo e no diretório de salvamento. Default None.
        salvar (bool, optional): Se True, salva a figura. Default False.

    Returns:
        None: A função exibe o plot e opcionalmente salva, mas não retorna objetos.
    """
    # Obtém colormap customizado
    cmap_custom = aplicar_estilo_visual(cores, retornar_cmap=True)
    # Cria figura e eixo padronizados
    fig, ax, font_sizes = padronizar_figura(1, 1)

    # Gera o heatmap
    sns.heatmap(matriz_corr,
                annot=True,          # Mostra valores nas células
                cmap=cmap_custom,    # Aplica colormap customizado
                fmt=".2f",           # Formato dos números (2 casas decimais)
                annot_kws={"size": 6}, # Tamanho da fonte das anotações
                cbar=False,          # Oculta a barra de cores
                ax=ax)

    # Formata e define o título principal (com quebra de linha)
    titulo_formatado = formatar_titulo(titulo)
    subtitulo = f"{titulo_formatado}" + (f" - {formatar_titulo(disciplina)}" if disciplina else "")
    ax.set_title(quebrar_rotulo(subtitulo, max_palavras=3), # Quebra título longo
                 fontsize=font_sizes.get('figure.titlesize', 11), # Usa tamanho de fonte padronizado
                 pad=8) # Espaçamento do título

    # Ajusta tamanho dos rótulos dos eixos
    ax.tick_params(axis='both', labelsize=font_sizes.get('xtick.labelsize', 4))

    # Define nome base do arquivo para salvamento
    nome_base_arquivo = titulo_para_snake_case(titulo) if n_arq is None else n_arq

    # Salva a figura se solicitado
    plt.tight_layout()
    if salvar:
        # Define diretório de salvamento (com subpasta para disciplina se fornecida)
        dir_salvar = os.path.join('correlacoes', titulo_para_snake_case(disciplina)) if disciplina else 'correlacoes'
        salvar_figura(f"mapa_calor_{nome_base_arquivo}",
                      diretorio=dir_salvar,
                      materia=titulo_para_snake_case(disciplina) if disciplina else '') # Passa matéria snake_case
    plt.show()

# ------------------------------------------------------------------------------
# Seção: Análise de Desempenho por Categoria
# ------------------------------------------------------------------------------





def selecao_impacto_variaveis_categoricas(df, variaveis_categoricas,
                                          paleta = 'azul',
                                          salvar=True,
                                          materia=None,
                                          coluna_avaliada='nota_final',
                                          plot_moderado=False):
    """Avalia e plota impacto de variáveis categóricas no desempenho.

    Analisa variáveis categóricas com base no "gap" de desempenho médio
    na `coluna_avaliada` entre suas categorias e no desequilíbrio da
    distribuição das categorias. Plota boxplots/countplots para variáveis
    consideradas de "impacto forte" segundo critérios
    heurísticos internos.

    A lógica difere se `materia` é None (compara POR vs MAT) ou especificada.

    Args:
        df (pd.DataFrame): Base de dados. Requer colunas de notas (com sufixo
            _por/_mat se `materia=None`) e as `variaveis_categoricas`.
        variaveis_categoricas (List[str]): Lista de colunas categóricas a avaliar.
        paleta (str, optional): Paleta de cores base. Default 'azul'.
        salvar (bool, optional): Se True, salva os gráficos gerados. Default True.
        materia (Optional[str], optional): Nome da disciplina ('portugues',
            'matematica'). Se None, compara desempenho entre Português e Matemática.
            Default None.
        coluna_avaliada (str, optional): Nome base da coluna de desempenho a
            ser avaliada (e.g., 'nota_final'). Sufixos _por/_mat são usados
            internamente se `materia=None`. Default 'nota_final'.
        plot_moderado (bool, optional): Se True, plota gráficos de impacto
            moderado. Default True.

    Returns:
        None: A função gera e exibe/salva gráficos, não retorna valores.

    Notes:
        - A função contém critérios heurísticos internos para definir "impacto forte"
          e "impacto fraco" que podem precisar de ajuste para outros contextos.
        - A lógica dupla baseada em `materia` pode ser complexa.
    """

    #  Comparação POR vs MAT (materia is None) ---
    if materia is None:
        gap_min = 1.0 # Limiar mínimo para diferença média entre POR e MAT
        frequencia_dominante_max = 70.0 # Limiar máximo para % da categoria mais frequente

        for col in variaveis_categoricas:
            # Pula coluna se tiver apenas uma categoria única
            if df[col].nunique() <= 1:
                continue

            # Calcula médias por categoria para Português e Matemática
            # Assume que as colunas existem no df
            try:
                medias_por = df.groupby(col)[f'{coluna_avaliada}_por'].mean()
                medias_mat = df.groupby(col)[f'{coluna_avaliada}_mat'].mean()
            except KeyError:
                print(f"Aviso: Colunas '{coluna_avaliada}_por' ou '{coluna_avaliada}_mat' não encontradas para a variável '{col}'. Pulando.")
                continue

            # Calcula diferença absoluta média entre as matérias para cada categoria
            gap_comportamento = abs(medias_por - medias_mat)

            # Calcula frequência relativa da categoria mais comum
            freq_dominante = df[col].value_counts(normalize=True).max() * 100

            # Critério para plotar: Gap máximo entre matérias >= limiar E frequência dominante <= limiar
            if (gap_comportamento.max() >= gap_min and
                freq_dominante <= frequencia_dominante_max):
                # Define nome do arquivo
                nome_do_arquivo = f'{titulo_para_snake_case(col)}_gap_por_mat' # Nome mais descritivo
                # Gera gráfico comparando as duas matérias para esta variável
                print(f"[PLOTANDO GAP POR/MAT] {col} → Gap Máx: {gap_comportamento.max():.2f} | Freq Dom: {freq_dominante:.1f}%")
                plot_boxplot_boxplot(df,
                                     x = col,
                                     materia=None, # Indica comparação entre matérias
                                     y1=f'{coluna_avaliada}_por',
                                     y2=f'{coluna_avaliada}_mat',
                                     paleta=paleta, # Usa paleta única
                                     hue=None, # Sem hue neste cenário
                                     nome_arquivo=nome_do_arquivo,
                                     salvar=salvar)

    #  Análise para uma Matéria Específica ---
    else:
        # Calcula desvio padrão da nota na matéria específica para definir limiares
        try:
            dp = df[coluna_avaliada].std()
        except KeyError:
            print(f"Aviso: Coluna avaliada '{coluna_avaliada}' não encontrada para matéria '{materia}'. Não é possível continuar.")
            return

        # Define limiares de gap (fraco e forte) baseados no desvio padrão
        limite_gap_fraco = 0.3 * dp
        limite_gap_forte = 0.9 * dp

        for col in variaveis_categoricas:
            n_cat = df[col].nunique()
            if n_cat <= 1: # Pula se não houver variação
                continue

            # Define limiar de desequilíbrio baseado no número de categorias
            if n_cat == 2:
                limiar_desequilibrio = 0.75 # Mais tolerante para binárias
            elif n_cat <= 4:
                limiar_desequilibrio = 0.60
            else:
                limiar_desequilibrio = 0.50 # Mais rigoroso para muitas categorias

            # Calcula desequilíbrio (frequência relativa da moda)
            desequilibrio = df[col].value_counts(normalize=True).max()
            # Calcula gap de desempenho (diferença entre média max e min da nota entre categorias)
            try:
                medias = df.groupby(col)[coluna_avaliada].mean()
                gap_media = medias.max() - medias.min()
            except KeyError:
                 print(f"Aviso: Coluna avaliada '{coluna_avaliada}' não encontrada ao calcular gap para '{col}'. Pulando.")
                 continue
            except Exception as e:
                 print(f"Aviso: Erro ao calcular gap para '{col}': {e}. Pulando.")
                 continue


            # Critério de impacto FRACO: Equilibrado E Gap pequeno
            if desequilibrio <= limiar_desequilibrio and gap_media <= limite_gap_fraco:
                print(f"[FRACO] {col} → equilíbrio: {desequilibrio:.2f} | gap: {gap_media:.2f}")
                nome_do_arquivo = f'{titulo_para_snake_case(col)}_impacto_fraco'

            # Critério de impacto FORTE: Desequilibrado E Gap grande
            elif desequilibrio >= limiar_desequilibrio and gap_media >= limite_gap_forte:
                print(f"[FORTE] {col} → desequilíbrio: {desequilibrio:.2f} | gap: {gap_media:.2f}")
                nome_do_arquivo = f'{titulo_para_snake_case(col)}_impacto_forte'
                # Plota boxplot + countplot (segmentado por aprovação)
                plot_boxplot_countplot(df,
                                       x = col,
                                       materia=materia,
                                       y=coluna_avaliada,
                                       paleta=paleta,
                                       hue='aprovacao', # Usa 'aprovacao' como hue
                                       nome_arquivo= nome_do_arquivo,
                                       salvar = salvar)
            
            # Critério de impacto MODERADO: Desequilibrado E Gap moderado
            elif limite_gap_fraco < gap_media < limite_gap_forte:
                print(f"[MODERADO] {col} → desequilíbrio: {desequilibrio:.2f} | gap: {gap_media:.2f}")
                nome_do_arquivo = f'{titulo_para_snake_case(col)}_impacto_moderado'
                # Plota boxplot + countplot (segmentado por aprovação)
                if plot_moderado:
                    plot_boxplot_countplot(df,
                                        x = col,
                                        materia=materia,
                                        y=coluna_avaliada,
                                        paleta=paleta,
                                        hue='aprovacao', # Usa 'aprovacao' como hue
                                        nome_arquivo= nome_do_arquivo,
                                        salvar = salvar)


      
    
            # Nota: Variáveis que não se encaixam em nenhum critério não são plotadas.




#opção alternativa para reduzir o número de gráficos

def diagnostico_impacto_variaveis_categoricas(df, variaveis_categoricas,
                                               materia=None,
                                               coluna_avaliada='nota_final',
                                               verbose=True):
    """
    Avalia o impacto de variáveis categóricas no desempenho sem gerar gráficos.
    
    Retorna um DataFrame com o gap de desempenho, desequilíbrio e rótulo de impacto
    (fraco, moderado, forte) para cada variável categórica, com base em critérios
    heurísticos internos.
    
    Args:
        df (pd.DataFrame): Base de dados.
        variaveis_categoricas (List[str]): Lista de colunas categóricas.
        materia (str or None): 'portugues', 'matematica' ou None para comparação POR x MAT.
        coluna_avaliada (str): Nome base da variável de desempenho.
        verbose (bool): Se True, imprime os diagnósticos.

    Returns:
        pd.DataFrame: Diagnóstico por variável categórica.
    """
    resultados = []

    if materia is None:
        gap_min = 1.0
        freq_dom_max = 70.0

        for col in variaveis_categoricas:
            if df[col].nunique() <= 1:
                continue
            try:
                medias_por = df.groupby(col)[f'{coluna_avaliada}_por'].mean()
                medias_mat = df.groupby(col)[f'{coluna_avaliada}_mat'].mean()
            except KeyError:
                continue

            gap = abs(medias_por - medias_mat).max()
            freq_dom = df[col].value_counts(normalize=True).max() * 100

            impacto = 'forte' if (gap >= gap_min and freq_dom <= freq_dom_max) else 'nenhum'

            if verbose and impacto == 'forte':
                print(f"[POR/MAT FORTE] {col} → gap: {gap:.2f} | freq_dom: {freq_dom:.1f}%")

            resultados.append({
                'variavel': col,
                'tipo_analise': 'comparacao_por_mat',
                'gap': gap,
                'freq_dominante(%)': freq_dom,
                'impacto': impacto
            })

    else:
        try:
            dp = df[coluna_avaliada].std()
        except KeyError:
            print(f"Aviso: Coluna '{coluna_avaliada}' não encontrada.")
            return pd.DataFrame()

        limite_gap_fraco = 0.3 * dp
        limite_gap_forte = 0.9 * dp

        for col in variaveis_categoricas:
            n_cat = df[col].nunique()
            if n_cat <= 1:
                continue

            limiar_deseq = 0.75 if n_cat == 2 else 0.6 if n_cat <= 4 else 0.5
            desequilibrio = df[col].value_counts(normalize=True).max()

            try:
                medias = df.groupby(col)[coluna_avaliada].mean()
                gap = medias.max() - medias.min()
            except:
                continue

            if desequilibrio <= limiar_deseq and gap <= limite_gap_fraco:
                impacto = 'fraco'
            elif desequilibrio >= limiar_deseq and gap >= limite_gap_forte:
                impacto = 'forte'
            else:
                impacto = 'moderado'

            if verbose:
                print(f"[{impacto.upper()}] {col} → desequilíbrio: {desequilibrio:.2f} | gap: {gap:.2f}")

            resultados.append({
                'variavel': col,
                'tipo_analise': f'analise_{materia}',
                'gap': gap,
                'desequilibrio': desequilibrio,
                'impacto': impacto
            })

    return pd.DataFrame(resultados).sort_values(by='gap', ascending=False).reset_index(drop=True)




# ------------------------------------------------------------------------------
# Seção: Gráficos Comparativos e de Dispersão
# ------------------------------------------------------------------------------

def comparar_notas_faltas(df, cor, dir, salvar = False):
    """Gera 6 gráficos comparando notas e faltas entre Português e Matemática.

    Assume que `df` contém colunas com sufixos _por e _mat (e.g.,
    'nota1_por', 'nota1_mat', 'faltas_por', 'faltas_mat').

    Args:
        df (pd.DataFrame): DataFrame com dados de ambas as disciplinas.
        cor (str): Nome da paleta de cores base (e.g., 'azul', 'verde').
        dir (str): Diretório onde salvar a figura resultante.
        salvar (bool, optional): Se True, salva a figura. Default False.


    Returns:
        None: A função exibe e salva o gráfico.
    """
    # Obtém paleta de cores
    paleta = aplicar_estilo_visual(cor)
    # Cria figura 2x3 padronizada
    fig, axes, font_sizes = padronizar_figura(2, 3)

    # Define os pares de variáveis e títulos para cada subplot
    comparacoes = [
        ('nota1_por', 'nota1_mat', 'POR vs MAT\n Nota 1 '),
        ('nota2_por', 'nota2_mat', 'POR vs MAT \n Nota 2 '),
        ('nota_final_por', 'nota_final_mat', 'POR vs MAT \n Nota Final '),
        ('faltas_por', 'faltas_mat', 'POR vs MAT \n Faltas'),
        ('faltas_por', 'nota_final_por', 'Nota Final vs Faltas \n Português'),
        ('faltas_mat', 'nota_final_mat', 'Nota Final vs Faltas \n Matemática'),
    ]

    # Itera e cria cada gráfico de dispersão com regressão
    for i, (x_col, y_col, titulo_sub) in enumerate(comparacoes):
        row, col = divmod(i, 3) # Calcula posição na grade 2x3
        ax = axes[row, col]
        # Plota regplot
        sns.regplot(data=df, x=x_col, y=y_col, ax=ax, color=paleta[2], # Usa a 3ª cor da paleta
                    scatter_kws={'alpha': 0.5, 's': 12}) # Configura pontos
        # Define título e rótulos formatados
        ax.set_title(titulo_sub, fontsize=font_sizes.get('axes.labelsize', 9)) # Usa tamanho de label para subtítulo
        ax.set_xlabel(formatar_titulo(x_col), fontsize=font_sizes.get('axes.labelsize', 9))
        ax.set_ylabel(formatar_titulo(y_col), fontsize=font_sizes.get('axes.labelsize', 9))
        # Ajusta tamanho dos ticks
        ax.tick_params(axis='both', labelsize=font_sizes.get('xtick.labelsize', 7))
        ax.grid(False) # Remove grid

    # Define título principal da figura
    fig.suptitle("Comparação entre Português e Matemática", fontsize=font_sizes.get('figure.titlesize', 12))
    plt.tight_layout()
    # Salva a figura (sem usar 'materia' pois compara ambas)
    if salvar:
        salvar_figura(diretorio=dir, nome_arquivo="comparativo_por_mat", materia=None)
    plt.show()


def plot_notas_faltas(df, cor, dir, mat,salvar = False):
    """Gera gráficos de dispersão para notas e faltas de UMA disciplina.

    Cria uma figura 2x2 mostrando relações entre:
    1. Faltas vs Nota Final
    2. Nota 1 vs Nota 2
    3. Nota 1 vs Nota Final
    4. Nota 2 vs Nota Final

    Args:
        df (pd.DataFrame): DataFrame com dados da disciplina. Requer colunas
            'faltas', 'nota1', 'nota2', 'nota_final'.
        cor (str): Nome da paleta de cores base.
        dir (str): Diretório onde salvar a figura.
        mat (str): Identificador da matéria ('portugues', 'matematica', ou outro)
            usado no título e nome do arquivo.
        salvar (bool, optional): Se True, salva a figura. Default False.


    Returns:
        None: A função exibe e salva o gráfico.
    """
    # Obtém paleta de cores
    paleta = aplicar_estilo_visual(cor)
    # Cria figura 2x2 padronizada
    fig, axes, font_sizes = padronizar_figura(2, 2)

    # Formata nome da matéria para o título
    if mat == 'portugues':
        materia_titulo = 'Português'
    elif mat == 'matematica':
        materia_titulo = 'Matemática'
    elif mat is None:
        materia_titulo = '' # Sem matéria especificada
    else:
        materia_titulo = formatar_titulo(mat) # Usa formatador geral

    # Define título principal
    fig.suptitle(f'Visualização de Atributos Quantitativos - {materia_titulo}',
                 fontsize=font_sizes.get('figure.titlesize', 12))

    # --- Cria os 4 subplots de dispersão com regressão ---

    # 1. Faltas vs Nota Final (axes[0, 0])
    sns.regplot(data=df, x='faltas', y='nota_final', ax=axes[0, 0], color=paleta[3], # Usa 4ª cor
                scatter_kws={'alpha': 0.5, 's': 12})
    axes[0, 0].set_title('Faltas vs Nota Final', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[0, 0].set_xlabel('Faltas', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[0, 0].set_ylabel('Nota Final', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[0, 0].tick_params(axis='both', labelsize=font_sizes.get('xtick.labelsize', 7))
    axes[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) # Limita número de ticks X
    axes[0, 0].grid(False)

    # 2. Nota 1 vs Nota 2 (axes[0, 1])
    sns.regplot(data=df, x='nota1', y='nota2', ax=axes[0, 1], color=paleta[3],
                scatter_kws={'alpha': 0.5, 's': 12})
    axes[0, 1].set_title('Nota 1 vs Nota 2', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[0, 1].set_xlabel('Nota 1', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[0, 1].set_ylabel('Nota 2', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[0, 1].tick_params(axis='both', labelsize=font_sizes.get('xtick.labelsize', 7))
    axes[0, 1].grid(False)

    # 3. Nota 1 vs Nota Final (axes[1, 0])
    sns.regplot(data=df, x='nota1', y='nota_final', ax=axes[1, 0], color=paleta[3],
                scatter_kws={'alpha': 0.5, 's': 12})
    axes[1, 0].set_title('Nota 1 vs Nota Final', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[1, 0].set_xlabel('Nota 1', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[1, 0].set_ylabel('Nota Final', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[1, 0].tick_params(axis='both', labelsize=font_sizes.get('xtick.labelsize', 7))
    # Formata eixo Y para 1 casa decimal
    axes[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
    axes[1, 0].grid(False)

    # 4. Nota 2 vs Nota Final (axes[1, 1])
    sns.regplot(data=df, x='nota2', y='nota_final', ax=axes[1, 1], color=paleta[3],
                scatter_kws={'alpha': 0.5, 's': 12})
    axes[1, 1].set_title('Nota 2 vs Nota Final', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[1, 1].set_xlabel('Nota 2', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[1, 1].set_ylabel('Nota Final', fontsize=font_sizes.get('axes.labelsize', 9))
    axes[1, 1].tick_params(axis='both', labelsize=font_sizes.get('xtick.labelsize', 7))
    axes[1, 1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) # Limita número de ticks X
    axes[1, 1].grid(False)

    plt.tight_layout()

    if salvar:
        # Salva a figura usando o identificador da matéria
        salvar_figura(diretorio=dir, nome_arquivo="plot_notas_faltas", materia=mat)
    plt.show()

# ==============================================================================
# ========================= SUMÁRIO ESTATÍSTICO E OUTLIERS =====================
# ==============================================================================

def resumir_outliers(df):
    """Calcula e exibe estatísticas de outliers para colunas numéricas.

    Utiliza o método IQR (Interquartile Range) para identificar limites
    inferior (Q1 - 1.5*IQR) e superior (Q3 + 1.5*IQR). Calcula a contagem
    total de outliers e quantos estão abaixo do limite inferior e acima do
    limite superior para cada coluna numérica no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        pd.DataFrame: Tabela (DataFrame) com as estatísticas de outliers por
                      coluna, ordenada por contagem total de outliers decrescente.
                      Os valores numéricos são formatados para exibição.
    """
    resumo = {}
    # Seleciona apenas colunas numéricas
    colunas_numericas = df.select_dtypes(include=np.number).columns

    for col in colunas_numericas:
        # Calcula quartis e IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define limites para outliers
        lim_inferior = Q1 - 1.5 * IQR
        lim_superior = Q3 + 1.5 * IQR

        # Identifica outliers
        outliers_baixo = df[df[col] < lim_inferior]
        outliers_cima = df[df[col] > lim_superior]

        # Armazena estatísticas no dicionário
        resumo[col] = {
            'Q1 (1º Quartil)': Q1,
            'Q3 (3º Quartil)': Q3,
            'Limite Inferior (L1)': lim_inferior,
            'Limite Superior (L3)': lim_superior,
            'Outliers Totais': len(outliers_baixo) + len(outliers_cima),
            'Outliers < L1': len(outliers_baixo),
            'Outliers > L3': len(outliers_cima)
        }

    # Converte dicionário para DataFrame
    resumo_df = pd.DataFrame.from_dict(resumo, orient='index')
    # Ordena por número total de outliers
    resumo_df = resumo_df.sort_values(by='Outliers Totais', ascending=False)

    # Formata números para exibição (notação científica para muito pequenos/grandes)
    resumo_df_formatado = resumo_df.applymap(
        lambda x: (
            0 if isinstance(x, (float, int)) and abs(x) < 1e-10 # Trata quase zero como 0
            else f"{x:.2e}" if isinstance(x, (float, int)) and (abs(x) < 1e-4 or abs(x) > 1e4) # Notação científica
            else round(x, 3) if isinstance(x, (float, int)) # Arredonda outros floats/ints
            else x # Mantém não numéricos (nomes das colunas no índice)
        )
    )
    # Exibe o DataFrame formatado (para uso interativo em notebooks)
    display(resumo_df_formatado)

    # Retorna o DataFrame original (não formatado) para possível uso programático
    return resumo_df # Retorna o DF com números, não o formatado

def perfil_categorico_outliers(df_outliers, df_total, variaveis_categoricas):
    """Compara a distribuição de categorias entre outliers e a base total.

    Para cada variável categórica especificada, calcula e compara a
    distribuição percentual de suas categorias no subconjunto de outliers
    (`df_outliers`) versus no conjunto de dados completo (`df_total`).

    Args:
        df_outliers (pd.DataFrame): Subconjunto de dados contendo apenas outliers.
        df_total (pd.DataFrame): DataFrame completo original.
        variaveis_categoricas (List[str]): Lista de nomes das colunas
            categóricas a serem analisadas.

    Returns:
        Dict[str, pd.DataFrame]: Dicionário onde cada chave é o nome de uma
        variável categórica e o valor é um DataFrame contendo:
            - 'Frequência Outlier': Contagem absoluta no grupo outlier.
            - '% Outlier': Percentual da categoria no grupo outlier.
            - '% Total': Percentual da categoria na base total.
            - 'Diferença (%)': Diferença percentual (% Outlier - % Total).
        Os DataFrames são ordenados pela 'Diferença (%)'.
    """
    perfis = {} # Dicionário para armazenar os resultados por variável

    for var in variaveis_categoricas:
        # Calcula distribuição percentual na base total
        dist_total = df_total[var].value_counts(normalize=True)
        # Calcula distribuição percentual nos outliers
        dist_out = df_outliers[var].value_counts(normalize=True)
        # Calcula contagem absoluta nos outliers
        contagem_out = df_outliers[var].value_counts()

        # Cria DataFrame combinando as informações, preenchendo com 0 se uma categoria não aparece em um grupo
        df_result = pd.DataFrame({
            'Frequência Outlier': contagem_out,
            '% Outlier': dist_out,
            '% Total': dist_total
        }).fillna(0)

        # Calcula a diferença percentual
        df_result['Diferença (%)'] = (df_result['% Outlier'] - df_result['% Total']) * 100

        # Formata percentuais para exibição (como strings com '%')
        df_result['% Outlier'] = (df_result['% Outlier'] * 100).round(1).astype(str) + '%'
        df_result['% Total'] = (df_result['% Total'] * 100).round(1).astype(str) + '%'
        # Arredonda a diferença
        df_result['Diferença (%)'] = df_result['Diferença (%)'].round(1)

        # Ordena pela diferença para destacar as maiores discrepâncias
        perfis[var] = df_result.sort_values(by='Diferença (%)', ascending=False)

    return perfis

# ==============================================================================
# ==================== ANÁLISE DE GRUPOS EXTREMOS ==============================
# ==============================================================================

def comparar_grupos_extremos(df, variavel_numerica, variaveis_categoricas, Q1, Q3, min_diferenca=0.15):
    """Compara distribuição de categorias entre grupos de desempenho extremo.

    Identifica dois grupos no DataFrame `df` com base nos quantis `Q1` e `Q3`
    da `variavel_numerica` (e.g., nota final):
    - Grupo Baixo: `variavel_numerica` <= Q1
    - Grupo Alto: `variavel_numerica` >= Q3
    Para cada `variavel_categorica`, compara a proporção de suas categorias
    entre esses dois grupos. Retorna as categorias onde a diferença absoluta
    de proporção excede `min_diferenca`.

    Args:
        df (pd.DataFrame): Base de dados original.
        variavel_numerica (str): Nome da variável usada para definir os grupos.
        variaveis_categoricas (List[str]): Lista de variáveis categóricas a comparar.
        Q1 (float): Limite do 1º quartil (ou limite inferior) para o grupo baixo.
        Q3 (float): Limite do 3º quartil (ou limite superior) para o grupo alto.
        min_diferenca (float, optional): Diferença mínima absoluta de proporção
            (0 a 1) para considerar uma categoria relevante. Default 0.15.

    Returns:
        Tuple[pd.DataFrame, int, int]:
            - df_resultado (pd.DataFrame): Tabela com categorias relevantes,
              mostrando a variável, a categoria, os percentuais em cada grupo
              (baixo e alto) e a diferença absoluta percentual. Ordenado pela
              diferença.
            - n_baixo (int): Número de observações no grupo baixo.
            - n_alto (int): Número de observações no grupo alto.
    """
    # Define os grupos extremos
    grupo_baixo = df[df[variavel_numerica] <= Q1]
    grupo_alto = df[df[variavel_numerica] >= Q3]
    n_baixo, n_alto = len(grupo_baixo), len(grupo_alto)
    resultados = [] # Lista para armazenar dicionários de resultados

    # Itera sobre as variáveis categóricas a serem analisadas
    for var in variaveis_categoricas:
        # Calcula distribuições e contagens absolutas em cada grupo
        dist_baixo = grupo_baixo[var].value_counts(normalize=True)
        dist_alto = grupo_alto[var].value_counts(normalize=True)
        abs_baixo = grupo_baixo[var].value_counts()
        abs_alto = grupo_alto[var].value_counts()

        # Compara cada categoria presente em qualquer um dos grupos
        todas_categorias = set(dist_baixo.index).union(dist_alto.index)
        for categoria in todas_categorias:
            # Obtém percentuais (ou 0 se categoria não existe no grupo)
            perc_baixo = dist_baixo.get(categoria, 0)
            perc_alto = dist_alto.get(categoria, 0)
            # Calcula diferença absoluta de proporção
            dif = abs(perc_baixo - perc_alto)

            # Se a diferença for maior que o mínimo, armazena o resultado
            if dif >= min_diferenca:
                resultados.append({
                    'Variável': var,
                    'Categoria': categoria,
                    # Formata percentuais e contagens para exibição clara
                    f'% Grupo Nota Baixa (≤{Q1:.1f})': f"{round(perc_baixo * 100, 1)}% ({abs_baixo.get(categoria, 0)}/{n_baixo})",
                    f'% Grupo Nota Alta (≥{Q3:.1f})': f"{round(perc_alto * 100, 1)}% ({abs_alto.get(categoria, 0)}/{n_alto})",
                    'Diferença Absoluta (%)': round(dif * 100, 1) # Diferença em formato percentual
                })

    # Cria DataFrame final e ordena pela diferença
    df_resultado = pd.DataFrame(resultados)
    if not df_resultado.empty: # Só ordena se houver resultados
        df_resultado = df_resultado.sort_values(by='Diferença Absoluta (%)', ascending=False)
        # Formata coluna de diferença para exibição (opcional, pode ser feito depois)
        df_resultado['Diferença Absoluta (%)'] = df_resultado['Diferença Absoluta (%)'].apply(
            lambda x: f"{x:.2e}" if (abs(x) < 1e-4 or abs(x) > 1e4) else round(x, 3)
        )

    # Retorna o DataFrame e os tamanhos dos grupos
    return df_resultado, n_baixo, n_alto


def identificar_extremos_comparaveis(
    df,
    variavel_numerica,
    variaveis_categoricas,
    min_diferenca=None,
    q_limite=None,
    entrada=None,
    otimizar=True
):
    """Identifica categorias com diferenças significativas entre grupos extremos.

    Define grupos de desempenho baixo e alto com base em quantis da
    `variavel_numerica`. Pode usar limites de quantis fixos (`q_limite`),
    limites manuais (`entrada=(Q1, Q3)`), ou tentar otimizar (`otimizar=True`)
    os quantis para encontrar grupos com tamanhos similares e que atendam a
    critérios específicos (tamanho mínimo, diferença máxima de tamanho,
    limites de nota). Chama `comparar_grupos_extremos` para obter os resultados.

    Args:
        df (pd.DataFrame): Base de dados.
        variavel_numerica (str): Nome da variável contínua para definir grupos.
        variaveis_categoricas (List[str]): Variáveis categóricas a comparar.
        min_diferenca (float, optional): Diferença mínima de proporção para
            `comparar_grupos_extremos`. Default 0.15.
        q_limite (float, optional): Quantil base (0 a 0.5) para definir Q1=q
            e Q3=1-q. Usado se `otimizar=False` e `entrada=None`. Default 0.25.
        entrada (Optional[Tuple[float, float]], optional): Limites manuais
            explícitos (Q1, Q3) para os grupos. Sobrescreve `q_limite` e
            `otimizar`. Default None.
        otimizar (bool, optional): Se True, tenta encontrar os melhores quantis
            que satisfaçam critérios internos de tamanho e equilíbrio dos grupos.
            Ignorado se `entrada` for fornecido. Default True.

    Returns:
        Tuple[Optional[pd.DataFrame], int, int, Optional[float], Optional[float]]:
            - df_diferencas (pd.DataFrame ou None): Resultado de
              `comparar_grupos_extremos` com as categorias relevantes, ou None
              se nenhum grupo válido for encontrado na otimização.
            - n_baixo (int): Tamanho do grupo baixo encontrado.
            - n_alto (int): Tamanho do grupo alto encontrado.
            - Q1 (float ou None): Limite inferior (quantil) usado.
            - Q3 (float ou None): Limite superior (quantil) usado.
    """
    # Define defaults internos se não fornecidos
    min_diferenca_final = 0.15 if min_diferenca is None else min_diferenca
    q_limite_final = 0.25 if q_limite is None else q_limite

    # --- Modo 1: Entrada Manual Explícita ---
    if entrada:
        Q1, Q3 = entrada
        print(f"Usando limites manuais: Q1={Q1:.2f}, Q3={Q3:.2f}")
        df_dif, n_baixo, n_alto = comparar_grupos_extremos(
            df, variavel_numerica, variaveis_categoricas, Q1, Q3, min_diferenca_final
        )
        display(df_dif) # Exibe resultado para o usuário
        return df_dif, n_baixo, n_alto, Q1, Q3

    # --- Modo 2: Otimização Automática ---
    if otimizar:
        melhores_resultados_tupla = None # Armazena a melhor tupla (df_dif, n_b, n_a, q1, q3)
        melhor_diff_abs_tamanho = float('inf') # Critério: minimizar diferença de tamanho dos grupos

        # Critérios internos para otimização (poderiam ser parâmetros)
        nota_minima_grupo_alto = 14.0 # Q3 deve ser >= que isso
        nota_maxima_grupo_baixo = 10.0 # Q1 deve ser <= que isso
        tamanho_minimo_grupo = 30     # n_baixo e n_alto devem ser >= que isso
        max_diff_relativa_tamanho = 0.2 # Diferença percentual máxima entre n_baixo e n_alto

        # Quantis a serem testados na otimização
        quantis_teste = [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]

        print("Tentando otimizar limites Q1 e Q3...")
        for q_atual in quantis_teste:
            Q1_teste = df[variavel_numerica].quantile(q_atual)
            Q3_teste = df[variavel_numerica].quantile(1 - q_atual)

            # Verifica critérios de nota mínima/máxima
            if Q3_teste < nota_minima_grupo_alto or Q1_teste > nota_maxima_grupo_baixo:
                continue # Pula este quantil se não atender aos limites de nota

            # Compara grupos com estes quantis
            df_dif_teste, n_baixo_teste, n_alto_teste = comparar_grupos_extremos(
                df, variavel_numerica, variaveis_categoricas, Q1_teste, Q3_teste, min_diferenca_final
            )

            # Verifica critérios de tamanho mínimo
            if n_baixo_teste < tamanho_minimo_grupo or n_alto_teste < tamanho_minimo_grupo:
                continue # Pula se algum grupo for muito pequeno

            # Verifica critério de diferença relativa de tamanho
            diff_tamanho_abs = abs(n_baixo_teste - n_alto_teste)
            if diff_tamanho_abs / max(n_baixo_teste, n_alto_teste) > max_diff_relativa_tamanho:
                continue # Pula se os grupos forem muito desiguais em tamanho

            # Se passou em todos os critérios, verifica se é a melhor solução até agora
            # (minimiza a diferença absoluta de tamanho)
            if diff_tamanho_abs < melhor_diff_abs_tamanho:
                melhor_diff_abs_tamanho = diff_tamanho_abs
                melhores_resultados_tupla = (df_dif_teste, n_baixo_teste, n_alto_teste, Q1_teste, Q3_teste)
                print(f"  Nova melhor configuração encontrada: q={q_atual:.3f} "
                      f"(Q1={Q1_teste:.2f}, Q3={Q3_teste:.2f}, "
                      f"N_b={n_baixo_teste}, N_a={n_alto_teste}, "
                      f"Diff={diff_tamanho_abs})")


        # Após testar todos os quantis, retorna a melhor configuração encontrada
        if melhores_resultados_tupla:
            print("Melhor configuração encontrada:")
            display(melhores_resultados_tupla[0]) # Exibe o DataFrame de diferenças
            return melhores_resultados_tupla # Retorna (df_dif, n_b, n_a, Q1, Q3)
        else:
            print("Nenhuma configuração de quantil satisfez todos os critérios de otimização.")
            return None, 0, 0, None, None # Retorna indicando falha

    # --- Modo 3: Quantil Fixo (sem otimização, sem entrada manual) ---
    else: # otimizar=False e entrada=None
        Q1 = df[variavel_numerica].quantile(q_limite_final)
        Q3 = df[variavel_numerica].quantile(1 - q_limite_final)
        print(f"Usando quantil fixo: q={q_limite_final:.2f} (Q1={Q1:.2f}, Q3={Q3:.2f})")
        df_dif, n_baixo, n_alto = comparar_grupos_extremos(
            df, variavel_numerica, variaveis_categoricas, Q1, Q3, min_diferenca_final
        )
        display(df_dif) # Exibe resultado
        return df_dif, n_baixo, n_alto, Q1, Q3


def plot_top_diferencas_extremos(df_diferencas, materia, q1_lim, q3_lim, n_baixo, n_alto,
                                  top_n=10, diretorio='graficos_diferencas_perfil', salvar= False):
    """Plota as categorias com maior diferença percentual entre grupos extremos.

    Recebe o DataFrame de resultados de `identificar_extremos_comparaveis`
    e cria um gráfico de barras horizontais mostrando as `top_n` categorias
    com a maior 'Diferença Absoluta (%)'.

    Args:
        df_diferencas (pd.DataFrame): DataFrame retornado por
            `identificar_extremos_comparaveis` (ou `comparar_grupos_extremos`).
            Deve conter colunas 'Variável', 'Categoria', 'Diferença Absoluta (%)'.
        materia (Optional[str]): Nome da disciplina/contexto ('portugues',
            'matematica', etc.) para definir a paleta e título.
        q1_lim (float): Limite inferior (Q1) usado para definir o grupo baixo.
        q3_lim (float): Limite superior (Q3) usado para definir o grupo alto.
        n_baixo (int): Tamanho do grupo baixo.
        n_alto (int): Tamanho do grupo alto.
        top_n (int, optional): Número de categorias a exibir no gráfico. Default 10.
        diretorio (str, optional): Pasta onde salvar a imagem.
            Default 'graficos_diferencas_perfil'.
        salvar (bool, optional): Se True, salva o gráfico. Default True.

    Returns:
        None: A função exibe e opcionalmente salva o gráfico.
    """
    # Verifica se há dados para plotar
    if df_diferencas is None or df_diferencas.empty:
        print("DataFrame de diferenças está vazio ou é None. Nenhum gráfico será gerado.")
        return

    # Seleciona e ordena as top N categorias
    # Garante que 'Diferença Absoluta (%)' seja numérica para ordenação
    df_top = df_diferencas.copy()
    # Tenta converter para numérico, tratando erros (caso já esteja formatado como string)
    df_top['Diff_Num'] = pd.to_numeric(df_top['Diferença Absoluta (%)'], errors='coerce')
    df_top.dropna(subset=['Diff_Num'], inplace=True) # Remove linhas onde a conversão falhou
    df_top = df_top.nlargest(top_n, 'Diff_Num') # Pega top N baseado no valor numérico

    # Cria rótulo combinado para o eixo Y
    df_top['rótulo_plot'] = df_top['Variável'] + ' = ' + df_top['Categoria'].astype(str)
    df_top = df_top.sort_values(by='Diff_Num', ascending=False) 

    # Define estilo visual baseado na matéria
    # Paleta azul para português, verde para matemática, cinza para outros/None
    if materia == 'portugues':
        paleta_nome = 'azul'
    elif materia == 'matematica':
        paleta_nome = 'verde'
    else: # Se materia for None ou outra string
        paleta_nome = 'cinza' # Ou uma paleta padrão neutra
        # Definir paleta 'cinza' em aplicar_estilo_visual ou usar uma do seaborn
        # Ex: cores = sns.color_palette("Greys", n_colors=len(df_top))
        
    # Obtém cores (precisa garantir que 'cinza' esteja definido ou usar fallback)
    try:
        cores = aplicar_estilo_visual(paleta_nome, n=len(df_top))
    except ValueError: # Fallback se paleta_nome não existir
        print(f"Aviso: Paleta '{paleta_nome}' não definida. Usando paleta padrão.")
        cores = sns.color_palette(n_colors=len(df_top))


    # Cria a figura e o eixo
    fig, ax = plt.subplots(figsize=(8, max(4.2, len(df_top) * 0.5))) # Altura dinâmica

    # Cria o gráfico de barras horizontais
    bars = sns.barplot(
        y=df_top['rótulo_plot'],
        x=df_top['Diff_Num'], # Usa a coluna numérica para plotar
        palette=cores, # Aplica cores
        ax=ax
    )

    # Adiciona anotações de texto nas barras com contraste ajustado
    for i, (valor_diff, patch) in enumerate(zip(df_top['Diff_Num'], bars.patches)):
        # Determina cor do texto baseada no brilho da cor da barra
        try:
            cor_barra_rgb = mcolors.to_rgb(patch.get_facecolor())
            brilho = mcolors.rgb_to_hsv(cor_barra_rgb)[2] # Componente V (Value/Brightness)
            # Define fator de contraste (ajustado empiricamente)
            fator_contraste = 0.45 if paleta_nome == 'verde' else 0.255 # Exemplo
            # Escolhe cor do texto (preto claro ou branco)
            cor_texto = mcolors.to_hex((brilho * fator_contraste,) * 3) if brilho > 0.72 else 'white'
        except Exception: # Fallback em caso de erro na conversão de cor
            cor_texto = 'black'

        # Adiciona o texto formatado (com %) na barra
        ax.text(valor_diff / 2, i, f"{valor_diff:.1f}%", color=cor_texto,
                va='center', ha='center', fontsize=10, weight='bold') # Ajuste fontsize/weight

    # --- Títulos e Legendas ---
    materia_titulo_sufixo = f' - {formatar_titulo(materia)}' if materia else ''
    plt.title(
        f'Top {len(df_top)} Categorias com Maior Diferença entre Grupos Extremos{materia_titulo_sufixo}',
        fontsize=14, weight='bold', pad=15
    )
    # Adiciona subtítulo com informações dos grupos
    plt.figtext(
        0.5, 0.01, # Posição ajustada (mais para baixo)
        f'Critério: Diferença Absoluta (%) entre Grupos Baixo (Nota ≤ {q1_lim:.1f}) e Alto (Nota ≥ {q3_lim:.1f})\n'
        f'Tamanhos dos Grupos: N_baixo = {n_baixo} | N_alto = {n_alto}',
        wrap=True, horizontalalignment='center', fontsize=8, style='italic'
    )

    # --- Ajustes Finais de Estética ---
    ax.yaxis.tick_right() # Move ticks Y para a direita (opcional)
    ax.set_ylabel(None) # Remove rótulo do eixo Y
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.5) # Adiciona grid vertical leve
    # Ajusta posição para dar espaço ao figtext e título
    fig.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.15) # Ajuste manual
    # plt.tight_layout(rect=[0, 0.05, 1, 0.9]) # Alternativa ao ajuste manual

    # Salva figura se solicitado
    if salvar:
        # Cria nome de arquivo
        nome_arquivo_salvo = f'top{len(df_top)}_diferencas_perfil'
        # Usa função de salvamento (garantir que titulo_para_snake_case esteja disponível)
        try:
             materia_snake = titulo_para_snake_case(materia) if materia else ''
             salvar_figura(nome_arquivo=nome_arquivo_salvo, diretorio=diretorio, materia=materia_snake)
        except NameError:
             print("Aviso: Funções 'salvar_figura' ou 'titulo_para_snake_case' não encontradas. Figura não salva automaticamente.")

    plt.show()

# ==============================================================================
# ==================== FUNÇÕES DE COMPARAÇÃO GERAL =============================
# ==============================================================================


def gerar_resumo_categoricas(df, variaveis, target='aprovacao'):
    """
    Gera tabelas de resumo para variáveis categóricas:
    - Frequência absoluta (Total)
    - Número de aprovados (1)
    - Taxa de aprovação (%)

    Args:
        df (pd.DataFrame): Base de dados.
        variaveis (list): Lista de nomes de colunas categóricas.
        target (str): Nome da variável-alvo binária (default = 'aprovacao').

    Returns:
        dict: Dicionário com uma tabela (DataFrame) por variável.
    """
    resultados = {}

    for var in variaveis:
        tabela = df.groupby(var)[target].agg(
            Total='count',
            Aprovados=lambda x: (x == 1).sum(),
            Taxa_Aprovacao=lambda x: round((x == 1).mean() * 100, 2)
        ).reset_index()
        resultados[var] = tabela

    return resultados
