from model.deeplearning_models.question_and_answer_model import VisualQuestionAnswering
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


class VilT(VisualQuestionAnswering):

    '''
    Classe que implementa o modelo VitT de question and answer
    '''

    def __init__(self):
        '''
        Construtor que carrega diretamente o modelo da biblioteca Vilt
        '''

        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


    def predict(self,question, image):
        '''
        Descrição:
        - Método que implementa a predição do modelo

        Parametos:
        - question (str): Questão a ser perguntada ao modelo
        - image (path): Caminho da imagem
        '''
        image = Image.open(image)
        text = question

        encoding = self.processor(image, text, return_tensors="pt")

        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        print("Predicted answer:", self.model.config.id2label[idx])
        return self.model.config.id2label[idx]