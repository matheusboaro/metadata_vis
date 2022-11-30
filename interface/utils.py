import os 

'''
Arquivos com funções de auxlio para a interface
'''


def get_files(path):
    '''
    Descrição:
    - Busca a lista de todos os arquivos com extensão .jpg em um diretorio 

    Parâmetros:
    - path (str): Caminho para um diretório

    Retorno:
    - files (list): Lista com o caminho dos arquivos dentro do diretorio passado
    '''

    files=[]
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            files.append(os.path.join(os.getcwd(),path+ file))

    return files