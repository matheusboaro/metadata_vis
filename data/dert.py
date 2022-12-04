
import os
import numpy as np
from data.utils import clean_folder
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import cv2
class Dert:
    '''
    Descrição:
    - Classe que implementa o modelo de detecção de objetos YOLO

    Métodos
    - set_model: Carrega o modelo YOLO
    - predict: Realiza a predição do modelo com a imagem escolhida pelo usuário
    - show: Mostra a predição do modelo
    - get_labels: Devolve as labels preditas pelo modelo para imagem fornecida
    - crop_objects: Corta da imagem fornecida os objetos detectados
    '''


    def __init__(self):
        self.model = self.set_model()
        clean_folder('metadata/')

    def set_model(self):
        '''
        Descrição:
        - Método que inicializa o modelo FastRCNN

        Parametro:
        - None

        Retorno:
        - model (FastRCNN()): Modelo FastRCNN
        '''
        self.feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        return DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


    def predict(self, image):
        '''
        Descrição:
        - Método que realiza a predição do modelo em uma imagem

        Parametros:
        - image (str): Caminho para uma imagem

        Retorno:
        - predictions (list(nd_array)): Predições do modelo
        '''
        image = Image.open(image)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        self.results = self.feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

        self.objects= {}
        for score, label, box in zip(self.results["scores"], self.results["labels"], self.results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # let's only keep detections with score > 0.9
            if score > 0.9:
                self.objects[self.model.config.id2label[label.item()]] = self.objects.get(self.model.config.id2label[label.item()],[])
                self.objects[self.model.config.id2label[label.item()]].append(box)

        self. results = self.objects
        return self.results

    def show(self,image,result,save=True):
        '''
        Descrição:
        - Método que mostra as predições realizadas pela rede

        Parametros:
        - image (str): Caminho para a imagem
        - result (list(nd_array)): Predições do modelo
        - save (bool): Flag que indica se o resultado deverá ser salvo ou não

        retorno:
        - Imagem salva, caso optado pelo usuário.
        '''
        image = cv2.imread(image)  
        for key, values in result.items():
            for point in values:
                x,y,w,h = int(point[0]),int(point[1]),int(point[2]),int(point[3])
                cv2.rectangle(image, (x, y), (w,h), (255,0,0), 4)
                cv2.putText(image,key,(x,h+10),0,2,(0,255,0),5)
        cv2.imwrite("metadata/teste.png",image)
        #return self.model.show_result(image, result, out_file = 'metadata/teste.png')


    def crop_objects(self, image):
        '''
        Descrição:
        - Método que corta cada objeto detectado da imagem.

        Parametros
        - image (str): Caminho para uma imagem

        Retono
        - None
        '''

        # if instance segmentation
        # bbox_result, segm_result = result
        

        im = Image.open(image)

        # Size of the image in pixels (size of original image)
        # (This is not mandatory)
        width, height = im.size
        count = 0
        if not os.path.exists("metadata/"):
            os.mkdir("metadata/")
            
        for key, values in self.results.items():
            for point in values:
        
        # Setting the points for cropped image
                left = int(point[0])
                top = int(point[1])
                right = int(point[2])
                bottom =int(point[3])
    
                # Cropped image of above dimension
                # (It will not change original image)
                im1 = im.crop((left, top, right, bottom))

            # Shows the image in image viewer
            

                im1.convert('RGB').save("metadata/{}_{}.jpg".format(count,key))
                count+=1

if __name__=='__main__':

    obj = Dert()

    print(obj.predict(r'J:\metadata_vis\IMG_3881.JPG'))
    #obj.show(r'J:\metadata_vis\IMG_3881.JPG',obj.results)
    obj.crop_objects(r'J:\metadata_vis\IMG_3881.JPG')