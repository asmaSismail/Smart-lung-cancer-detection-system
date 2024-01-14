import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from PyQt5.QtWidgets import  QDialog, QFileDialog
import classification_cnn

class Classification_win(QtWidgets.QMainWindow):
     
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gs = gridspec.GridSpec(2, 2)
        self.figure = Figure(figsize=(6, 6))
        self.figure.suptitle('Classification des nodules', fontsize=16 , color="blue")
        self.canvas = FigureCanvas(self.figure)
        self.ax  = self.figure.add_subplot(self.gs[0])
        self.ax3 = self.figure.add_subplot(self.gs[2])
        self.buploadlabel = self.figure.add_axes([0.63, 0.30, 0.18, 0.070])
        self.buploadmsk = self.figure.add_axes([0.63, 0.42, 0.18, 0.07])
        self.bclasif = self.figure.add_axes([0.63, 0.52, 0.18, 0.07])
        self.bclasif = Button(self.bclasif, 'classifier')
        self.buploadlabel = Button(self.buploadlabel, 'upload_label')
        self.buploadmsk = Button(self.buploadmsk, 'upload_CT_mask')
        self.ax.set_title("Mask")
        self.ax3.set_title("Prediction")
        self.figure.set_figheight(20)
        self.figure.set_figwidth(20)
        self.gs.tight_layout(self.figure,rect=[3, 5, 3, 4])
        self.setCentralWidget(self.canvas)
        self.buploadlabel.on_clicked(self.load_label)
        self.buploadmsk.on_clicked(self.load_nodule)
        self.bclasif.on_clicked(self.classifier)
    def load_nodule(self,event):
        fname=QFileDialog.getOpenFileName(self, 'Open file', './params/classifier/data/testimg', 'Images (*.npy)')
        self.nodule= np.load("{}".format(fname[0]))
        self.ax.imshow(self.nodule, cmap='gray')
        self.ax3.imshow(self.nodule, cmap='gray')
        
    def load_label(self,event):
        self.ax3.set_ylabel('')
        fname=fname=QFileDialog.getOpenFileName(self, 'Open file', './params/classifier/data/testlabl', 'Images (*.npy)')
        print(fname)
        self.label= np.load("{}".format(fname[0])).astype(int)
        if self.label == 0 :
            self.ax.set_ylabel('benign (not cancer)',color='green',fontsize=15)
        else :
            self.ax.set_ylabel('malignant (cancer)',color='red',fontsize=15)
        
    def classifier(self,event):
        self.ax3.set_xlabel('')
        self.classifieur = classification_cnn.create_model()
        self.mythreshold = classification_cnn.mythreshold
        self.classifieur.load_weights("./params/classifier/my_model_weights_classifier_update_last.h5")
        self.label_pred = (self.classifieur.predict(np.expand_dims(self.nodule, axis=0))>= self.mythreshold).astype(int)
        print(self.label_pred)
        if self.label_pred[0] == 0 :
            self.ax3.set_ylabel('benign (not cancer)',color='green',fontsize=15)
        else :
            self.ax3.set_ylabel('malignant (cancer)',color='red',fontsize=15)