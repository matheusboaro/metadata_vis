import os, shutil
'''
Arquivos com funções de auxlio no desenvolvimento do projeto
'''


def clean_folder(folder):
    '''
    Descrição:
    - Realiza a exclusão de todos os documentos de uma pasta fornecida pelo usuário.

    Parâmetros:
    - folder: caminho para uma pasta

    Retorno:
    - None
    '''

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))