import pandas as pd
from sklearn.model_selection import train_test_split
from modulos.pre_modelagem import importar_base

def dividir_e_salvar(df, nome_base, target='aprovacao', test_size=0.3, random_state=42):
    """
    Realiza a divisão estratificada da base em conjuntos de treino e teste, e salva os resultados em arquivos CSV.

    A função separa a base de dados original em conjuntos de treino e teste, utilizando a variável de destino
    especificada. A divisão é estratificada para manter a proporção das classes, garantindo consistência
    em análises futuras. Os dados divididos são salvos como arquivos CSV na pasta 'data/', nomeados com
    a base e o random_state.

    Args:
        df (pd.DataFrame): Base de dados completa.
        nome_base (str): Nome identificador da base (ex: 'portugues', 'matematica').
        target (str, optional): Nome da variável alvo (default 'aprovacao').
        test_size (float, optional): Proporção dos dados reservados para teste (default 0.3).
        random_state (int, optional): Semente para reprodutibilidade da divisão (default 42).

    Salva:
        - data/dados_treino_<nome_base>_rs<random_state>.csv
        - data/dados_teste_<nome_base>_rs<random_state>.csv

    Exemplo:
        >>> dividir_e_salvar(df, nome_base='portugues')
        [✓] Portugues: treino e teste salvos com random_state=42
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    df_treino = df.loc[X_train.index].copy()
    df_teste = df.loc[X_test.index].copy()

    df_treino.to_csv(f'data/dados_treino_{nome_base}_rs{random_state}.csv', index=False)
    df_teste.to_csv(f'data/dados_teste_{nome_base}_rs{random_state}.csv', index=False)
    print(f"[✓] {nome_base.title()}: treino e teste salvos com random_state={random_state}")

if __name__ == "__main__":
    df_por = importar_base('portugues')
    df_mat = importar_base('matematica')

    dividir_e_salvar(df_por, nome_base='portugues')
    dividir_e_salvar(df_mat, nome_base='matematica')
