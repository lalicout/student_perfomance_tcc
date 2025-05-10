# ======================================
# Módulo: documentar_resultados.py
# ======================================
"""Exportação e salvamento padronizado de resultados de análise.

Este módulo fornece funções utilitárias para salvar gráficos Matplotlib
e exportar DataFrames Pandas para tabelas formatadas em LaTeX, facilitando
a documentação e a criação de relatórios consistentes.

Principais Funcionalidades:
  - Salvamento de figuras Matplotlib em um sistema de diretórios organizado.
  - Exportação de DataFrames para arquivos .tex como tabelas LaTeX, com
    ajustes opcionais de cabeçalho e controle de largura.
"""
# Nota: As funções de estilização visual (aplicar_estilo_visual, etc.)
# foram movidas para eda_functions.py e removidas desta docstring.

# ==============================================================================
# ========================== IMPORTAÇÃO DE BIBLIOTECAS =========================
# ==============================================================================

import pandas as pd
# import numpy as np # Não utilizado diretamente no código fornecido
import os
# import seaborn as sns # Não utilizado diretamente no código fornecido
# from matplotlib.colors import LinearSegmentedColormap, to_hex # Não utilizado
import matplotlib.pyplot as plt # Necessário para plt.savefig

# Assume que quebrar_nomes_latex vem de outro módulo (ex: eda_functions)
# Se não estiver disponível, a funcionalidade de ajustar cabeçalhos falhará.
# from eda_functions import quebrar_nomes_latex # Exemplo de importação

# ==============================================================================
# ==================== SEÇÃO: SALVAMENTO E EXPORTAÇÃO ==========================
# ==============================================================================

def salvar_figura(nome_arquivo,
                  materia=None,
                  diretorio = None,
                  formato = 'png',
                  pasta_raiz_imagens = 'imagens'):
    """Salva a figura Matplotlib atual em um diretório organizado.

    Cria uma estrutura de pastas (`pasta_raiz_imagens`/`diretorio`) se não
    existir e salva a figura ativa do Matplotlib (`plt.gcf()`) nela. O nome
    do arquivo final pode incluir um identificador de `materia`.

    Args:
        nome_arquivo (str): O nome base para o arquivo da figura
            (sem extensão).
        materia (Optional[str]): Um identificador opcional (e.g., nome da
            disciplina ou categoria) a ser anexado ao `nome_arquivo`. Se None
            ou vazio, não é adicionado.
        diretorio (str, optional): O nome da subpasta dentro de
            `pasta_raiz_imagens` onde a figura será salva.
            Padrão é 'figuras'.
        formato (str, optional): A extensão/formato do arquivo da imagem
            (e.g., 'png', 'pdf', 'svg', 'jpg'). Padrão é 'png'.
        pasta_raiz_imagens (str, optional): O nome da pasta raiz onde a
            subpasta `diretorio` será criada. Padrão é 'imagens'.

    Returns:
        None
    """
    # Diretório centralizado para as imagens coletadas
    try:
        os.makedirs(pasta_raiz_imagens, exist_ok=True)
    except OSError as e:
        print(f"Erro ao criar diretório raiz '{pasta_raiz_imagens}': {e}")
        return

    if not diretorio:
        caminho_pasta=''
    else:
        # Define caminho da subpasta dentro da pasta raiz
        caminho_pasta = os.path.join(pasta_raiz_imagens, diretorio)
    try:
        os.makedirs(caminho_pasta, exist_ok=True)
    except OSError as e:
        print(f"Erro ao criar subdiretório '{caminho_pasta}': {e}")
        return


    # Define caminho completo do arquivo
    if materia: # Anexa materia se não for None ou vazia
        nome_completo = f"{nome_arquivo}_{materia}.{formato}"
    else:
        nome_completo = f"{nome_arquivo}.{formato}"

    caminho_completo = os.path.join(caminho_pasta, nome_completo)

    # Figura salva com configurações de alta resolução e ajuste de bounding box
    try:
        # Pega a figura atual para salvar
        fig = plt.gcf()
        if not fig.get_axes(): # Verifica se a figura tem algum conteúdo
             print("Aviso: Tentando salvar uma figura vazia.")
             # return # Pode optar por não salvar figuras vazias

        fig.savefig(caminho_completo, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {caminho_completo}")
    except Exception as e:
        print(f"Erro ao salvar figura em '{caminho_completo}': {e}")

#------------------------------------------------------------------------------

# Placeholder para a função que quebra nomes de colunas (deve ser importada ou definida)
def quebrar_nomes_latex(col_dict, limite):
     """(Placeholder) Quebra nomes de colunas para LaTeX."""
     # Implementação real necessária aqui ou via importação
     print("AVISO: Usando placeholder para quebrar_nomes_latex. Nomes de colunas não serão quebrados.")
     # Retorna o dicionário original sem modificação como fallback
     return {v: v for k, v in col_dict.items()}

#------------------------------------------------------------------------------

from typing import Optional # Para anotação de tipo

def exportar_df_para_latex(df: pd.DataFrame,
                           nome_tabela: str = "tabela",
                           caminho_pasta: str = "./tables",
                           index: bool = False,
                           caption: Optional[str] = None,
                           label: Optional[str] = None,
                           ajustar_cabecalhos: bool = True,
                           limite_quebra_cabecalho: int = 25,
                           margem_adjustwidth: str = "1in" 
                          ) -> None:
    """Exporta um DataFrame Pandas para um arquivo .tex como uma tabela LaTeX.

    Gera um arquivo .tex contendo o código LaTeX para uma tabela, baseada no
    DataFrame fornecido. A tabela é formatada para inclusão em documentos
    LaTeX, com opções para gerar automaticamente legendas (`caption`) e
    rótulos (`label`), ajustar cabeçalhos longos e controlar a largura
    usando o ambiente `adjustwidth` (do pacote `changepage`).

    Args:
        df (pd.DataFrame): O DataFrame a ser exportado como tabela.
        nome_tabela (str, optional): O nome base para o arquivo .tex
            (sem extensão) e usado para gerar caption/label padrão.
            Padrão é "tabela".
        caminho_pasta (str, optional): O caminho para o diretório onde o
            arquivo .tex será salvo. Será criado se não existir.
            Padrão é "./tables".
        index (bool, optional): Se True, inclui o índice do DataFrame como
            uma coluna na tabela LaTeX. Padrão é False.
        caption (Optional[str], optional): A legenda (`\\caption{}`) para a
            tabela. Se None, uma legenda padrão é gerada a partir de
            `nome_tabela`. Padrão é None.
        label (Optional[str], optional): O rótulo (`\\label{}`) para
            referenciar a tabela (e.g., `tab:meunome`). Se None, um label
            padrão é gerado a partir de `nome_tabela`. Padrão é None.
        ajustar_cabecalhos (bool, optional): Se True, tenta quebrar nomes de
            colunas longos em múltiplas linhas usando a função auxiliar
            `quebrar_nomes_latex`. Padrão é True.
        limite_quebra_cabecalho (int, optional): O número aproximado de
            caracteres por linha a ser usado por `quebrar_nomes_latex` ao
            ajustar cabeçalhos (relevante apenas se `ajustar_cabecalhos`
            for True). Padrão é 25.
        margem_adjustwidth (str, optional): A margem a ser aplicada em ambos
            os lados pelo ambiente `adjustwidth` (e.g., "1in", "2cm").
            Define o quanto a tabela pode exceder as margens normais.
            Padrão é "1in".

    Returns:
        None

    Raises:
        NameError: Se `ajustar_cabecalhos` for True e a função
            `quebrar_nomes_latex` não estiver definida ou acessível.
        IOError: Se ocorrer um erro ao escrever o arquivo .tex.

    Notes:
        - O LaTeX gerado requer os pacotes `float` (para `[H]`), `booktabs`
          (implícito por `df.to_latex`) e `changepage` (para `adjustwidth`).
        - A função `quebrar_nomes_latex` precisa estar definida ou importada.
    """
    # Cria a pasta se necessário
    try:
        os.makedirs(caminho_pasta, exist_ok=True)
    except OSError as e:
        print(f"Erro ao criar diretório '{caminho_pasta}': {e}")
        return # Aborta se não conseguir criar a pasta

    # Legenda e label automáticos se não fornecidos
    final_caption = caption if caption is not None else f"Tabela: {nome_tabela.replace('_', ' ').capitalize()}"
    final_label = label if label is not None else f"tab:{nome_tabela.lower()}"

    caminho_arquivo = os.path.join(caminho_pasta, f"{nome_tabela}.tex")

    # Faz uma cópia para evitar modificar o DataFrame original
    df_copy = df.copy()

    # Aplica quebra nos nomes das colunas na cópia, se desejado
    if ajustar_cabecalhos:
        try:
            # Tenta chamar a função que deveria estar disponível
            colunas_ajustadas = quebrar_nomes_latex({col: col for col in df_copy.columns}, limite_quebra_cabecalho)
            df_copy.rename(columns=colunas_ajustadas, inplace=True)
        except NameError:
            print("Aviso: Função 'quebrar_nomes_latex' não encontrada. "
                  "Cabeçalhos não serão ajustados.")
        except Exception as e_break:
             print(f"Aviso: Erro ao tentar quebrar cabeçalhos com 'quebrar_nomes_latex': {e_break}. "
                   "Cabeçalhos não serão ajustados.")

    # Gera o conteúdo LaTeX da tabela a partir da cópia (modificada ou não)
    try:
        # Gera apenas o conteúdo tabular interno
        conteudo_tabela_latex = df_copy.to_latex(index=index,
                                                 escape=False, # Importante se cabeçalhos têm comandos LaTeX
                                                 # multicolumn=True, # Padrões do Pandas
                                                 # multicolumn_format='c',
                                                 header=True, # Garante que o cabeçalho seja incluído
                                                 # booktabs=True # Pandas usa booktabs por padrão
                                                 )
    except Exception as e_tolatex:
        print(f"Erro ao converter DataFrame para LaTeX com df.to_latex(): {e_tolatex}")
        return

    # Cria o conteúdo completo do arquivo .tex usando adjustwidth
    # Usa \small para tentar reduzir o tamanho da fonte da tabela
    # Usa \renewcommand{\arraystretch}{1.25} para aumentar espaçamento vertical
    latex_string = f"""\\begin{{table}}[H] % Requer float package
    \\centering
    \\caption{{{final_caption}}}
    \\label{{{final_label}}}
    \\renewcommand{{\\arraystretch}}{{1.25}} % Aumenta espaçamento vertical
    \\begin{{adjustwidth}}{{ -{margem_adjustwidth} }}{{ -{margem_adjustwidth} }} % Requer changepage package
    \\centering % Centraliza a tabela dentro do adjustwidth
    \\small % Reduz tamanho da fonte
    {conteudo_tabela_latex.strip()}
    \\end{{adjustwidth}}
    \\renewcommand{{\\arraystretch}}{{1.0}} % Restaura espaçamento padrão
\\end{{table}}
% Para inserir esta tabela no texto principal do LaTeX:
% ----------------------------------------------------
% \\usepackage{{float}} % No preâmbulo
% \\usepackage[strict]{{changepage}} % No preâmbulo (ou apenas \\usepackage{{changepage}})
% \\usepackage{{booktabs}} % Geralmente necessário para tabelas do Pandas
% \\input{{{os.path.relpath(caminho_arquivo).replace(os.sep, '/')}}} % Gera caminho relativo
"""

    # Salva o arquivo .tex
    try:
        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write(latex_string)
        print(f"Tabela LaTeX salva com sucesso em: {caminho_arquivo}")
    except IOError as e:
        print(f"Erro de I/O ao salvar a tabela LaTeX em '{caminho_arquivo}': {e}")
    except Exception as e_write:
         print(f"Erro inesperado ao salvar a tabela LaTeX em '{caminho_arquivo}': {e_write}")

# ======================================
# Fim do módulo documentar_resultados.py
# ======================================
