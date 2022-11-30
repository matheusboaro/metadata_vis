class AbstractModel:

    '''
    AbstractModel
    - Classe abstrata dos modelos de deeplearning utilizados
    '''
    
    def __init__(self,model):
        self.model = model

    def  predict(self, image):
        pass

    
        