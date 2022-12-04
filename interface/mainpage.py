import json
import sys, os
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont, QFontDatabase, QColor, QIcon,QPixmap
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QTableWidgetItem,QFileDialog, QDialog,QListWidgetItem, QListView
from data.analyzer import Analyzer
from interface.utils import get_files
from PIL import Image


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("interface/mainpage.ui", self)
        self.setWindowTitle("MetadataAnalyzer-0.0")

        self.buttom_connections()
        self.set_object_detection_model()
        self.set_image_captioning_model()
        self.set_question_and_answering_model()
        
        self.analyzer_item={}


    def buttom_connections(self):
        self.search_file_buttom.clicked.connect(self.file_window)
        self.analyze_button.clicked.connect(self.analyze)
        self.export_buttom.clicked.connect(self.export)


    def set_object_detection_model(self):
        model_names = ['None','Dert']
        self.object_detection_combo.addItems(model_names)

    def set_image_captioning_model(self):
        model_names = ['None','VitGPT2']
        self.image_captioning_combo.addItems(model_names)

    def set_question_and_answering_model(self):
        model_names = ['None','VilT']
        self.qanda_combo.addItems(model_names)

    def analyze(self):
        self.setup_dict = {
            'image_path':self.file_name.text(),
            'object_detection_model' : self.object_detection_combo.currentText(),
            'image_captioning' : self.image_captioning_combo.currentText(),
            'question': self.question_line.text(),
            'question_answering_model': self.qanda_combo.currentText()
        }
        self.an_pipeline = Analyzer(self.setup_dict)
        self.an_pipeline.detect_object()
        self.an_pipeline.image_captioning()
        self.an_pipeline.question_answering()

        self.show_results()

    def show_results(self):
        try:
            _fromUtf8 = QtCore.QString.fromUtf8
        except AttributeError:
            _fromUtf8 = lambda s: s

        files = get_files("metadata/")
        print(files)
        self.object_list.clear()
        self.object_list.setViewMode(QListView.IconMode)
        for x in files:
            im = Image.open(x)

            # Size of the image in pixels (size of original image)
            # (This is not mandatory)
            width, height = im.size
            item = QListWidgetItem()
            icon = QIcon()
            icon.addPixmap(QPixmap(_fromUtf8(x)), QIcon.Normal, QIcon.Off)
            item.setIcon(icon)
            self.object_list.addItem(item)

        size = QSize(256, 256)
        
        # setting icon size
        self.object_list.setIconSize(size)

        pixmap = QPixmap('metadata/teste.png').scaled(661, 401, QtCore.Qt.KeepAspectRatio)
        self.image.resize(pixmap.width(), pixmap.height())
        self.image.setPixmap(pixmap)
        
        #self.image.setCentralWidget(self.image)
        self.caption_label.setText(self.an_pipeline.results['image_captioning'])
        self.answer_label.setText(self.answer_label.text()+self.an_pipeline.results['qea'])
        self.question_label.setText(self.question_label.text()+self.question_line.text())

        
  


    def file_window(self,directory='', forOpen=True, fmt='', isFolder=False):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.DontUseCustomDirectoryIcons
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)
        # ARE WE TALKING ABOUT FILES OR FOLDERS
        if isFolder:
            dialog.setFileMode(QFileDialog.DirectoryOnly)
        else:
            dialog.setFileMode(QFileDialog.AnyFile)
        # OPENING OR SAVING
        dialog.setAcceptMode(QFileDialog.AcceptOpen) if forOpen else dialog.setAcceptMode(QFileDialog.AcceptSave)

        # SET FORMAT, IF SPECIFIED
        if fmt != '' and isFolder is False:
            dialog.setDefaultSuffix(fmt)
            dialog.setNameFilters([f'{fmt} (*.{fmt})'])

        # SET THE STARTING DIRECTORY
        if directory != '':
            dialog.setDirectory(str(directory))


        if dialog.exec_() == QDialog.Accepted:
            path = dialog.selectedFiles()[0]  # returns a list
            self.file_name.setText(path)
            return path
        else:
            return ''

    def export(self,directory='', forOpen=True, fmt='', isFolder=False):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.DontUseCustomDirectoryIcons
        dialog = QFileDialog()
        dialog.setOptions(options)

        dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)
        # ARE WE TALKING ABOUT FILES OR FOLDERS
        if isFolder:
            dialog.setFileMode(QFileDialog.DirectoryOnly)
        else:
            dialog.setFileMode(QFileDialog.AnyFile)
        # OPENING OR SAVING
        dialog.setAcceptMode(QFileDialog.AcceptOpen) if forOpen else dialog.setAcceptMode(QFileDialog.AcceptSave)

        # SET FORMAT, IF SPECIFIED
        if fmt != '' and isFolder is False:
            dialog.setDefaultSuffix(fmt)
            dialog.setNameFilters([f'{fmt} (*.{fmt})'])

        # SET THE STARTING DIRECTORY
        if directory != '':
            dialog.setDirectory(str(directory))


        if dialog.exec_() == QDialog.Accepted:
            path = dialog.selectedFiles()[0]  # returns a list
            self.save_metadata(path)
            return path
        else:
            return ''

    def save_metadata(self,path):
        self.an_pipeline.results['object_detection'] = [item.tolist() for item in self.an_pipeline.results['object_detection']]

        json_object = json.dumps(self.an_pipeline.results,indent=4)   
 
        # Writing to sample.json
        with open(path, "w") as outfile:
            outfile.write(json_object)



    