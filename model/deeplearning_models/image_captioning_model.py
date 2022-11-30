from model.deeplearning_models.abstract_model import AbstractModel

class ImageCaptioningModel(AbstractModel):

    '''
    ImageCaptioningModel
    - Classe abstrata dos modelos de descrição de imagem utilizados
    '''
    def __init__(self, model):
        super().__init__(model)