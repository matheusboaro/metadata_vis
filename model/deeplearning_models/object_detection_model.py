from model.deeplearning_models.abstract_model import AbstractModel

class ObjectDetectionModel(AbstractModel):
    '''
    ObjectDetectionModel
    - Classe abstrata dos modelos de detecção de objetos utilizados
    '''

    def __init__(self, model):
        super().__init__(model)

    def predict(self, image):
        pass