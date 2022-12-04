from data.dert import Dert
from data.vit_gpt2 import VitGpt2
from data.VilT import VilT

class Analyzer:
    def __init__(self, setup_dict:dict):
        '''
        Classe Analyzer
        - Classe que implementa o núcleo do analyzador.

        Parâmetros:
        - setup_dict : dict: um dicionário com as configurações escolhidas pelo usuário através da interface gráfica
        '''
        self.object_detection_model = setup_dict.get('object_detection_model')
        self.image_captioning_model = setup_dict.get('image_captioning')
        self.question_answering_model = setup_dict.get('question_answering_model')
        self.question = setup_dict.get('question')
        self.image_path = setup_dict.get('image_path')
        self.results = {}

    def detect_object(self):
        '''
        Método que realiza a detecção de objetos em cena baseado no modelo escolhido pelo usuário

        Parâmetros:
        - None

        Retorno
        - None
        '''

        if self.object_detection_model == 'Dert':
            model = Dert()
        else:
            return []
        self.results['object_detection'] = model.predict(self.image_path)
        model.show(self.image_path,self.results['object_detection'])
        model.crop_objects(self.image_path)

    def image_captioning(self):
        '''
        Método que cria a legenda da imagem baseada no modelo escolhido pelo usuário

        Parametros:
        - None

        Retorno:
        - None
        '''


        if self.image_captioning_model == 'VitGPT2':
            print(self.image_captioning)
            model = VitGpt2()
        else:
            self.results['image_captioning'] = [None]
            return 
        self.results['image_captioning'] = model.predict(self.image_path)[0]

    def question_answering(self):

        '''
        Método que responde uma pergunta feita pelo usuário baseado no modelo escolhido pelo usuário

        Parametro:
        - None

        Retorno:
        - None
        '''
        if self.question_answering_model =='VilT':
            model = VilT()
        else:
            self.results['qea'] = None
            return
        
        self.results['qea']=model.predict(self.question, self.image_path)