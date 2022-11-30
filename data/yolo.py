from mmdet.apis import init_detector, inference_detector
import mmcv
from model.deeplearning_models.object_detection_model import ObjectDetectionModel
import os
import numpy as np
from mmdet.core import get_classes
from PIL import Image
from data.utils import clean_folder
class Yolo(ObjectDetectionModel):
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


    def __init__(self, config_file=r'J:\metadata_visualizer\data\config\yolo\yolov3_d53_mstrain-608_273e_coco.py', checkpoint_file=r'J:\metadata_visualizer\data\config\yolo\yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
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

        model = init_detector(self.config_file, self.checkpoint_file, device='cuda:0')
        return model

    def predict(self, image):
        '''
        Descrição:
        - Método que realiza a predição do modelo em uma imagem

        Parametros:
        - image (str): Caminho para uma imagem

        Retorno:
        - predictions (list(nd_array)): Predições do modelo
        '''

        self.predictions = inference_detector(self.model, image)
        return self.predictions

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

        return self.model.show_result(image, result, out_file = 'metadata/teste.png')

    def get_labels(self):
        '''
        Descrição:
        - Método que identifica o nome das classes preditas na imagem. i.e: carro, casa, oculos...

        Prametros:
        - None

        Retorno:
        - labels_impt (list): Index das labels detectadas
        - bboxex (list(nd_array)): Região em que os objetos fora detectados ((x1,y1),(x2,y2))
        - label_classes (list(str)): Nome das classes.
        '''

        bbox_result = self.predictions
        # if instance segmentation
        # bbox_result, segm_result = result
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)\
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)
        labels_impt = np.where(bboxes[:, -1] > 0.3)[0]

        classes = get_classes("coco")
        labels_impt_list = [labels[i] for i in labels_impt]
        labels_class = [classes[i] for i in labels_impt_list]

        return labels_impt,bboxes, labels_class

    def crop_objects(self, image):
        '''
        Descrição:
        - Método que corta cada objeto detectado da imagem.

        Parametros
        - image (str): Caminho para uma imagem

        Retono
        - None
        '''
        bbox_result = self.predictions
        # if instance segmentation
        # bbox_result, segm_result = result
        labels_impt,bboxes, labels_class = self.get_labels()

        im = Image.open(image)

        # Size of the image in pixels (size of original image)
        # (This is not mandatory)
        width, height = im.size
        count = 0
        if not os.path.exists("metadata/"):
            os.mkdir("metadata/")
            
        for box in range(labels_impt.size):
            
        # Setting the points for cropped image
            left = bboxes[labels_impt][box][0]
            top = bboxes[labels_impt][box][1]
            right = bboxes[labels_impt][box][2]
            bottom = bboxes[labels_impt][box][3]
            print(left)
            # Cropped image of above dimension
            # (It will not change original image)
            im1 = im.crop((left, top, right, bottom))

            # Shows the image in image viewer
            

            im1.convert('RGB').save("metadata/{}_{}.jpg".format(box,labels_class[box]))
            count+=1
