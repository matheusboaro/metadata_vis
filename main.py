import sys, os
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont, QFontDatabase, QColor, QIcon
from PyQt5.QtWidgets import QGraphicsDropShadowEffect, QTableWidgetItem
from interface.mainpage import MainWindow

sys.path.append("..\\data\\")






if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
    sys.exit()