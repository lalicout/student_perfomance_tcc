# ajustar_path.py
import sys
import pathlib

def adicionar_modulos_ao_path(nome_raiz='student_perfomance_tcc', nome_pasta='modulos'):
    """Adiciona a pasta de módulos ao sys.path a partir de qualquer local"""
    path = pathlib.Path().resolve()

    while path.name != nome_raiz and path != path.parent:
        path = path.parent

    if path.name != nome_raiz:
        raise RuntimeError(f"Pasta raiz '{nome_raiz}' não encontrada.")

    mod_path = path / nome_pasta
    if not mod_path.exists():
        raise FileNotFoundError(f"Pasta de módulos '{mod_path}' não encontrada.")

    if str(mod_path) not in sys.path:
        sys.path.append(str(mod_path))