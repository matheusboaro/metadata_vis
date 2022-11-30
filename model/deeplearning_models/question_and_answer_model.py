from model.deeplearning_models.abstract_model import AbstractModel
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


class VisualQuestionAnswering(AbstractModel):
    '''
    VisualQuestionAnswering
    - Classe abstrata dos modelos de produção de respostas baseadas em imagem utilizados
    '''

    
    def __init__(self, model):
        super().__init__(model)

    def predict(self, image):
        pass