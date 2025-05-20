# ======================================
# Módulo: pre_modelagem.py
# ======================================
"""Pré-processamento e preparação de dados de desempenho escolar.

Este módulo fornece um conjunto de funções para as etapas iniciais do pipeline
de análise de dados, com foco na importação, padronização, transformação e
preparação de dados brutos de estudantes para posterior análise exploratória
e modelagem preditiva.

As funcionalidades incluem o carregamento de dados a partir de arquivos CSV
por disciplina, a padronização de nomes de colunas e a tradução de valores
categóricos, além da criação de uma variável alvo binária. Também são realizadas
a codificação de variáveis categóricas (Label Encoding e One-Hot Encoding),
o tratamento de valores ausentes por meio de imputação, o escalonamento de
atributos numéricos (opcional) e o balanceamento de classes utilizando
a técnica SMOTE-Tomek.

O objetivo principal é transformar os dados em um formato limpo, estruturado
e apropriado para as fases seguintes do processo de mineração de dados,
em especial para a etapa de preparação de dados conforme o CRISP-DM.
"""



#--IMPORTAÇÃO DE BIBLIOTECAS ----------------------------------


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek



#-- SEÇÃO: IMPORTAÇÃO E LIMPEZA INICIAL ------------------------

def importar_base(materia, caminho_completo=None):
    """Lê, traduz e realiza padronização inicial de dados de desempenho escolar.

    Carrega dados de um arquivo CSV específico da disciplina (Matemática ou
    Português), renomeia as colunas para português, traduz os valores de
    variáveis categóricas para termos em português e adiciona uma coluna
    'aprovacao' baseada na nota final do aluno (>=10 para Aprovado).

    Args:
        materia (str): Código da disciplina. Aceita 'mat' para Matemática ou
            'por' para Português. Também aceita 'matematica' e 'portugues'
            como aliases que são convertidos internamente.
        caminho_completo(str): localização dos arquivos CSV. Default None.

    Returns:
        pd.DataFrame: Um DataFrame do Pandas contendo os dados carregados,
        com colunas renomeadas, valores traduzidos e a nova coluna 'aprovacao'
        (com valores 'Aprovado' ou 'Reprovado').

    Raises:
        ValueError: Se o parâmetro `materia` não for um dos valores válidos.
        FileNotFoundError: Se o arquivo CSV correspondente à `materia` não
            for encontrado no caminho padrão ('~/student_performance_tcc/data/').
    """
    # Validação e normalização do parâmetro 'materia'
    if materia not in ['mat', 'por']:
        if materia == 'portugues':
            materia = 'por'
        elif materia == 'matematica':
            materia = 'mat'
        else:
            raise ValueError("O parâmetro 'materia' deve ser 'mat' para Matemática ou 'por' para Português.")

    # Define o caminho base onde os arquivos CSV estão localizados
    # ATENÇÃO: Este caminho é fixo e depende da estrutura de diretórios do usuário.
    if not caminho_completo:

        base_path = os.path.join(os.path.expanduser("~"), "student_performance_pipeline", "data")
        arquivo = f"student-{materia}.csv"
    
    caminho_completo = os.path.join(base_path, arquivo)

    # Leitura do arquivo CSV correspondente à matéria
    # Levanta FileNotFoundError se o arquivo não existir.
    df = pd.read_csv(caminho_completo, sep=';')

    # Dicionário para renomear colunas (Inglês -> Português)
    colunas_renomeadas = {
        'school': 'escola', 'sex': 'genero', 'age': 'idade', 'address': 'endereco',
        'famsize': 'tamanho_familia', 'Pstatus': 'status_parental',
        'Medu': 'escolaridade_mae', 'Fedu': 'escolaridade_pai',
        'Mjob': 'profissao_mae', 'Fjob': 'profissao_pai',
        'reason': 'motivo_escolha_escola', 'guardian': 'responsavel_legal',
        'traveltime': 'tempo_transporte', 'studytime': 'tempo_estudo',
        'failures': 'reprovacoes', 'schoolsup': 'apoio_escolar',
        'famsup': 'apoio_familiar', 'paid': 'aulas_particulares',
        'activities': 'atividades_extracurriculares', 'nursery': 'frequentou_creche',
        'higher': 'interesse_ensino_superior', 'internet': 'acesso_internet',
        'romantic': 'relacionamento_romantico', 'famrel': 'relacao_familiar',
        'freetime': 'tempo_livre', 'goout': 'frequencia_saidas',
        'Dalc': 'alcool_dias_uteis', 'Walc': 'alcool_fim_semana',
        'health': 'saude', 'absences': 'faltas',
        'G1': 'nota1', 'G2': 'nota2', 'G3': 'nota_final'
    }
    df.rename(columns=colunas_renomeadas, inplace=True)

    # Dicionário para traduzir valores categóricos
    substituicoes = {
        'escola': {'GP': 'Gabriel Pereira', 'MS': 'Mousinho da Silveira'},
        'genero': {'F': 'Mulher', 'M': 'Homem'},
        'endereco': {'U': 'Urbano', 'R': 'Rural'},
        'tamanho_familia': {'GT3': 'Mais de 3 membros', 'LE3': '3 membros ou menos'},
        'status_parental': {'A': 'Separados', 'T': 'Juntos'},
        'profissao_mae': {'at_home': 'Dona de casa', 'health': 'Área da saúde', 'other': 'Outra profissão', 'services': 'Serviços', 'teacher': 'Professor(a)'},
        'profissao_pai': {'at_home': 'Dono de casa', 'health': 'Área da saúde', 'other': 'Outra profissão', 'services': 'Serviços', 'teacher': 'Professor(a)'},
        'motivo_escolha_escola': {'course': 'Curso específico', 'other': 'Outro motivo', 'home': 'Próximo de casa', 'reputation': 'Reputação da escola'},
        'responsavel_legal': {'mother': 'Mãe', 'father': 'Pai', 'other': 'Outro responsável'},
        'apoio_escolar': {'yes': 'Sim', 'no': 'Não'},
        'apoio_familiar': {'yes': 'Sim', 'no': 'Não'},
        'aulas_particulares': {'yes': 'Sim', 'no': 'Não'},
        'atividades_extracurriculares': {'yes': 'Sim', 'no': 'Não'},
        'frequentou_creche': {'yes': 'Sim', 'no': 'Não'},
        'interesse_ensino_superior': {'yes': 'Sim', 'no': 'Não'},
        'acesso_internet': {'yes': 'Sim', 'no': 'Não'},
        'relacionamento_romantico': {'yes': 'Sim', 'no': 'Não'}
    }
    # Aplica as substituições de valores
    for coluna, mapa_valores in substituicoes.items():
        # Verifica se a coluna existe antes de tentar substituir
        if coluna in df.columns:
            df[coluna].replace(mapa_valores, inplace=True)

    # Cria a coluna 'aprovacao' baseada na 'nota_final'
    # Define 'Aprovado' se nota >= 10, caso contrário 'Reprovado'
    df['aprovacao'] = df['nota_final'].apply(lambda x: 'Aprovado' if x >= 10 else 'Reprovado')

    return df


# -- PREPARAÇÃO PARA MODELAGEM -----------------------




def preparar_treino_e_teste(
    df_train,
    df_test,
    target='aprovacao',
    drop_notas=True,
    scaling=True
):
    """
    Prepara os conjuntos de treino e teste para modelagem supervisionada.

    Inclui codificação de variáveis categóricas, mapeamentos binários e ordinais, imputação de valores ausentes, escalonamento de variáveis numéricas e separação em X/y.

    Se `scaling=True`, os conjuntos retornados já estarão com as variáveis numéricas escalonadas (padrão z-score), utilizando `StandardScaler` treinado no conjunto de treino.

    Args:
        df_train (pd.DataFrame): Conjunto de treino original.
        df_test (pd.DataFrame): Conjunto de teste original.
        target (str): Nome da variável alvo. Default é 'aprovacao'.
        drop_notas (bool): Se True, remove as colunas nota1, nota2 e nota_final. Default é True.
        scaling (bool): Se True, aplica imputação e escalonamento às variáveis numéricas. Default é True.

    Returns:
        Tuple:
            X_train (pd.DataFrame): Conjunto de treino com variáveis preditoras (possivelmente escalonadas).
            X_test (pd.DataFrame): Conjunto de teste com variáveis preditoras (possivelmente escalonadas).
            y_train (pd.Series): Variável-alvo do treino.
            y_test (pd.Series): Variável-alvo do teste.
            scaler (StandardScaler or None): Objeto de escalonamento treinado, ou None se scaling=False.
            imputer (SimpleImputer or None): Objeto de imputação treinado, ou None se scaling=False.
    """



    # Cópias para não alterar os DataFrames originais
    df_train = df_train.copy()
    df_test = df_test.copy()

    # 1. Drop das colunas de nota (se solicitado)
    if drop_notas:
        col_notas = ['nota1', 'nota2', 'nota_final']
        col_notas_existentes_train = [col for col in col_notas if col in df_train.columns]
        col_notas_existentes_test = [col for col in col_notas if col in df_test.columns]

        df_train.drop(columns=col_notas_existentes_train, inplace=True)
        df_test.drop(columns=col_notas_existentes_test, inplace=True)

    # 2. Mapeamentos manuais (binários e ordinais)
    mappings = {
        'tamanho_familia': {'Mais de 3 membros': 1, '3 membros ou menos': 0},
        'aprovacao': {'Aprovado': 1, 'Reprovado': 0}
    }

    for col, mapa in mappings.items():
        for df_ in [df_train, df_test]:
            if col in df_.columns:
                df_[col] = df_[col].map(mapa)
                if df_[col].isnull().any():
                    print(f"Aviso: valores não mapeados ou ausentes na coluna '{col}'.")

    # 3. Mapeamento das colunas binárias Sim/Não
    bin_cols_sim_nao = [
        'apoio_escolar', 'apoio_familiar', 'aulas_particulares',
        'atividades_extracurriculares', 'frequentou_creche',
        'interesse_ensino_superior', 'acesso_internet', 'relacionamento_romantico'
    ]

    for col in bin_cols_sim_nao:
        for df_ in [df_train, df_test]:
            if col in df_.columns:
                df_[col] = df_[col].map({'Sim': 1, 'Não': 0})
                if df_[col].isnull().any():
                    valores_invalidos = df_.loc[df_[col].isnull(), col].unique()
                    print(f"Aviso: valores não mapeados na coluna '{col}': {valores_invalidos}")



    # One-Hot Encoding
    ohe_cols = [
        'escola', 'genero', 'endereco', 'status_parental',
        'profissao_mae', 'profissao_pai', 'motivo_escolha_escola',
        'responsavel_legal'
    ]
    ohe_cols_exist = [c for c in ohe_cols if c in df_train.columns and c in df_test.columns]

    if ohe_cols_exist:
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        ohe.fit(df_train[ohe_cols_exist])

        df_train_ohe = pd.DataFrame(
            ohe.transform(df_train[ohe_cols_exist]),
            columns=ohe.get_feature_names_out(ohe_cols_exist),
            index=df_train.index
        )
        df_test_ohe = pd.DataFrame(
            ohe.transform(df_test[ohe_cols_exist]),
            columns=ohe.get_feature_names_out(ohe_cols_exist),
            index=df_test.index
        )

        df_train.drop(columns=ohe_cols_exist, inplace=True)
        df_test.drop(columns=ohe_cols_exist, inplace=True)

        df_train = pd.concat([df_train, df_train_ohe], axis=1)
        df_test = pd.concat([df_test, df_test_ohe], axis=1)
    else:
        print("Nenhuma coluna categórica nominal encontrada para aplicar One-Hot Encoding.")

    # Separar X e y
    y_train = df_train[target]
    X_train = df_train.drop(columns=[target])

    y_test = df_test[target]
    X_test = df_test.drop(columns=[target])

    # Escalonamento (Imputer + StandardScaler)
    scaler = None
    imputer = None
    if scaling:
        num_cols = X_train.select_dtypes(include='number').columns.tolist()

        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = imputer.transform(X_test[num_cols])

        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler, imputer




def preparar_dados(
    df,
    target_column= None,
    scaling=False,
    columns_to_drop=None
):
    """
    Aplica transformações completas para preparar o DataFrame para modelagem.

    Realiza remoção de colunas, codificação manual (binária e ordinal), codificação one-hot de variáveis categóricas nominais, e, se solicitado, imputação e escalonamento das variáveis numéricas não binárias.

    Se `scaling=True`, as variáveis numéricas não-binárias são escalonadas com `StandardScaler`, e o resultado final já estará transformado.

    Args:
        df (pd.DataFrame): DataFrame original.
        target_column (str, optional): Nome da variável alvo (excluída do escalonamento). Default é None.
        scaling (bool, optional): Se True, aplica imputação (mean) e escalonamento (z-score). Default é False.
        columns_to_drop (list, optional): Lista de colunas a remover antes do processamento. Default é None.

    Returns:
        pd.DataFrame: DataFrame processado, com codificações e transformações aplicadas. Se `scaling=True`, já retornado com variáveis numéricas escalonadas.

    Raises:
        ValueError: Se `scaling=True` e `target_column` não estiver presente no DataFrame.
        TypeError: Se `columns_to_drop` não for uma lista ou None.
    """


    # 0) Validações iniciais

    if columns_to_drop is not None:
        if not isinstance(columns_to_drop, list):
            raise TypeError("'columns_to_drop' deve ser uma lista ou None.")
        # Verifica se as colunas a serem removidas existem
        cols_existentes_drop = [c for c in columns_to_drop if c in df.columns]
        if len(cols_existentes_drop) != len(columns_to_drop):
            cols_faltantes = set(columns_to_drop) - set(cols_existentes_drop)
            print(f"Aviso: Colunas para remover não encontradas e serão ignoradas: {list(cols_faltantes)}")
        columns_to_drop = cols_existentes_drop # Usa apenas as colunas existentes

    df_proc = df.copy()

    # 1) Remover colunas indesejadas (se houver alguma válida para remover)
    if columns_to_drop:
        print(f"Removendo colunas: {columns_to_drop}")
        df_proc.drop(columns=columns_to_drop, inplace=True)

    # 2) Label Encoding manual
    mappings = {
        'tamanho_familia': {'Mais de 3 membros': 1, '3 membros ou menos': 0},
        'aprovacao':       {'Aprovado': 1, 'Reprovado': 0}, # Mapeia mesmo se for target
    }
    bin_cols_sim_nao = [
        'apoio_escolar', 'apoio_familiar', 'aulas_particulares',
        'atividades_extracurriculares', 'frequentou_creche',
        'interesse_ensino_superior', 'acesso_internet', 'relacionamento_romantico'
    ]
    # Guarda nomes das colunas que se tornaram binárias 0/1 após mapeamento
    binary_encoded_cols = []

    for col, m in mappings.items():
        if col in df_proc:
            original_nan_count = df_proc[col].isnull().sum()
            df_proc[col] = df_proc[col].map(m)
            if df_proc[col].isnull().sum() > original_nan_count:
                 print(f"Aviso: NaNs gerados em '{col}' por valores não mapeados.")
            # Verifica se a coluna resultante é binária (ignorando NaNs)
            unique_vals = df_proc[col].dropna().unique()
            if set(unique_vals) <= {0, 1}:
                 binary_encoded_cols.append(col)

    for col in bin_cols_sim_nao:
        if col in df_proc:
            original_nan_count = df_proc[col].isnull().sum()
            df_proc[col] = df_proc[col].map({'Sim': 1, 'Não': 0})
            if df_proc[col].isnull().sum() > original_nan_count:
                 print(f"Aviso: NaNs gerados em '{col}' por valores não mapeados.")
            unique_vals = df_proc[col].dropna().unique()
            if set(unique_vals) <= {0, 1}:
                 binary_encoded_cols.append(col)

    # 3) One-Hot Encoding
    ohe_cols_cat = [ # Colunas categóricas nominais a serem codificadas
        'escola', 'genero', 'endereco', 'status_parental',
        'profissao_mae', 'profissao_pai', 'motivo_escolha_escola',
        'responsavel_legal'
    ]
    exist_ohe_cols = [c for c in ohe_cols_cat if c in df.columns]

    if exist_ohe_cols:
        # Trata NaNs antes do OHE (preenche com placeholder)
        for col in exist_ohe_cols:
             if df_proc[col].isnull().any():
                  print(f"Aviso: Preenchendo NaNs em '{col}' com 'DESCONHECIDO' antes do OHE.")
                  # Converte para string para garantir tipo consistente antes de fillna
                  df_proc[col] = df_proc[col].astype(str).fillna('DESCONHECIDO')

        # handle_unknown='error' é mais seguro para consistência entre treino/teste
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='error')
        try:
            # Aplica OHE
            arr = ohe.fit_transform(df_proc[exist_ohe_cols])
            cols_created_by_ohe = ohe.get_feature_names_out(exist_ohe_cols)
            # Cria DataFrame com as dummies
            df_ohe = pd.DataFrame(arr, columns=cols_created_by_ohe, index=df_proc.index)
            # Remove colunas originais e concatena as dummies
            df_proc.drop(columns=exist_ohe_cols, inplace=True)
            df_proc = pd.concat([df_proc, df_ohe], axis=1)
            # Adiciona as novas colunas dummy à lista de binárias
            binary_encoded_cols.extend(cols_created_by_ohe)
        except ValueError as ve:
             print(f"ERRO durante One-Hot Encoding: {ve}. Verifique categorias ou NaNs.")
             raise ve # Re-levanta o erro

    # 4) Escalonamento (opcional)
    if scaling:

        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"Coluna alvo '{target_column}' não encontrada no DataFrame.")

        # Detecta todas as colunas numéricas APÓS as codificações
        numeric_cols = df_proc.select_dtypes(include='number').columns.tolist()

        # Identifica o conjunto final de colunas binárias (as mapeadas + as do OHE)
        # Garante que sejam únicas e existam no df atual
        final_binary_cols = set(col for col in binary_encoded_cols if col in df_proc.columns)

        # Seleciona para escalar: numéricas que NÃO são o alvo e NÃO são binárias
        cols_to_scale = [
            c for c in numeric_cols
            if c != target_column and c not in final_binary_cols
        ]

        if cols_to_scale:
            print(f"Aplicando Imputação(mean) e Scaling(StandardScaler) a: {cols_to_scale}")
            imputer = SimpleImputer(strategy='mean')
            scaler  = StandardScaler()
            # Aplica imputer e scaler usando .loc para evitar SettingWithCopyWarning
            # Trata possível erro se houver apenas NaNs em uma coluna
            try:
                df_proc.loc[:, cols_to_scale] = imputer.fit_transform(df_proc[cols_to_scale])
                df_proc.loc[:, cols_to_scale] = scaler.fit_transform(df_proc[cols_to_scale])
            except ValueError as ve_scale:
                print(f"ERRO durante imputação/escalonamento: {ve_scale}. Verifique NaNs ou colunas com variância zero.")
                raise ve_scale
        else:
             print("Nenhuma coluna numérica não-binária (e não-alvo) encontrada para escalar.")

    # 5) Limpeza final de NaNs e Infinitos
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True) # Trata infinitos
    rows_before_drop = len(df_proc)
    df_proc.dropna(inplace=True) # Remove linhas com QUALQUER NaN restante
    rows_after_drop = len(df_proc)
    if rows_before_drop > rows_after_drop:
        print(f"INFO: {rows_before_drop - rows_after_drop} linhas removidas devido a NaNs.")

    print(f"Shape final do DataFrame preparado: {df_proc.shape}")
    return df_proc


# -- SEÇÃO: BALANCEAMENTO DE CLASSES ---------------------

def balancear_dados(X, y, r_state = 42):
    """Aplica a técnica SMOTE-Tomek para balancear a distribuição de classes.

    Utiliza SMOTE para oversampling da classe minoritária e Tomek Links para
    undersampling/limpeza. Preserva o tipo de dado original (DataFrame/Series
    ou NumPy array).

    Args:
        X (Union[pd.DataFrame, np.ndarray]): Matriz de features (preditoras).
            Espera-se que contenha apenas dados numéricos.
        y (Union[pd.Series, np.ndarray]): Vetor de labels da classe alvo.
        r_state (int): Define o random_state. Default é 42

    Returns:
        Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
            Uma tupla contendo:
            - X_resampled: As features reamostradas, no mesmo formato de `X`.
            - y_resampled: Os labels reamostrados, no mesmo formato de `y`.

    Notes:
        - Requer a biblioteca `imbalanced-learn` instalada.
        - SMOTE-Tomek pode alterar o tamanho do conjunto de dados.
        - Deve ser aplicado apenas aos dados de TREINAMENTO.
    """

    # Guarda informações para reconstrução do tipo original
    is_df = isinstance(X, pd.DataFrame)
    cols = X.columns if is_df else None
    y_name = y.name if hasattr(y, 'name') else None # Preserva nome da Series y

    # Converte para arrays NumPy para SMOTETomek
    X_np = X.values if is_df else np.asarray(X)
    y_np = y.values if isinstance(y, pd.Series) else np.asarray(y) # Usa .values para Series

    # Instancia SMOTETomek com random_state para reprodutibilidade
    smt = SMOTETomek(random_state=r_state)
    # Aplica reamostragem
    # ATENÇÃO: Isso pode falhar se a classe minoritária for muito pequena (< k_neighbors do SMOTE).
    try:
        X_res_np, y_res_np = smt.fit_resample(X_np, y_np)
    except ValueError as e:
         print(f"ERRO durante SMOTETomek: {e}. "
               "Verifique se a classe minoritária tem amostras suficientes (>= k_neighbors+1, padrão k=5). "
               "Retornando dados originais.")
         return X, y # Retorna original em caso de erro

    # Reconstrói para o formato original, se necessário
    if is_df:
        X_res = pd.DataFrame(X_res_np, columns=cols)
    else:
        X_res = X_res_np

    if isinstance(y, pd.Series): # Verifica se y original era Series
        y_res = pd.Series(y_res_np, name=y_name)
    else:
        y_res = y_res_np

    print(f"Shape após balanceamento: X={X_res.shape}, y={y_res.shape}")
    if isinstance(y_res, pd.Series):
        print("Contagem de classes após balanceamento:")
        print(y_res.value_counts())
    else:
        unique, counts = np.unique(y_res, return_counts=True)
        print("Contagem de classes após balanceamento:")
        print(dict(zip(unique, counts)))


    return X_res, y_res

# ======================================
# Fim do módulo
# ======================================
