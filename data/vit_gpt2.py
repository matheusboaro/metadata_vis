from PIL import Image
import torch
from torchvision import transforms
from model.deeplearning_models.image_captioning_model import ImageCaptioningModel
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers.models.ofa.generate import sequence_generator

class VitGpt2(ImageCaptioningModel):
    '''
    Classe que implementa o modelo VitGPT2 para a geração de descrição de imagem
    
    '''

    def __init__(self):

        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        

    def predict(self,image):
        '''
        Descrição:
        - Método que realiza a predição do modelo

        Parametros:
        - image (str): Caminho para uma imagem

        Retorno
        - preds (str): Predição do modelo
        '''


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        print(preds)
        return preds
