import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from PyQt5.QtWidgets import  QDialog, QFileDialog
import segmentation_unet2D
import classification_cnn
class segmentation_win(QtWidgets.QMainWindow):
     
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gs = gridspec.GridSpec(2, 2)
        self.figure = Figure(figsize=(6, 6))
        self.figure.suptitle('Detection des nodules', fontsize=16 , color="blue")
        self.canvas = FigureCanvas(self.figure)
        self.ax  = self.figure.add_subplot(self.gs[0])
        self.ax2 = self.figure.add_subplot(self.gs[1])
        self.ax3 = self.figure.add_subplot(self.gs[2])
        self.buploadim = self.figure.add_axes([0.63, 0.20, 0.18, 0.070])
        self.buploadmsk = self.figure.add_axes([0.63, 0.32, 0.18, 0.07])
        self.bpred = self.figure.add_axes([0.63, 0.07, 0.18, 0.07])
        self.bpred = Button(self.bpred, 'Predict')
        self.buploadim = Button(self.buploadim, 'upload_CT_image')
        self.buploadmsk = Button(self.buploadmsk, 'upload_CT_mask')
        self.ax.set_title("Image")
        self.ax2.set_title("Mask")
        self.ax3.set_title("Prediction")
        self.figure.set_figheight(20)
        self.figure.set_figwidth(20)
        self.gs.tight_layout(self.figure,rect=[3, 5, 3, 4])
        self.setCentralWidget(self.canvas)
        self.buploadmsk.on_clicked(self.uploadmsk)
        self.buploadim.on_clicked(self.uploadim)
        self.bpred.on_clicked(self.predire)
        
    def uploadmsk(self,event):
        self.ax2.set_visible(True)
        fname=QFileDialog.getOpenFileName(self, 'Open file', './params/segmentation/data/testseg', 'Images (*.npy)')
        self.mask=  np.load("{}".format(fname[0]))
        self.ax2.imshow(self.mask, cmap='gray')
        
    def uploadim(self,event):
        fname=QFileDialog.getOpenFileName(self, 'Open file', './params/segmentation/data/testimg', 'Images (*.npy)')
        print(fname)
        self.image= np.load("{}".format(fname[0]))
        self.ax.imshow(self.image, cmap='gray')
            
    def predire(self,event):
        self.segmentation = segmentation_unet2D.create_model()
        self.segmentation.load_weights("./params/segmentation/my_model_weights_update1.h5")
        self.classifieur = classification_cnn.create_model()
        self.mythreshold = classification_cnn.mythreshold
        self.classifieur.load_weights("./params/classifier/my_model_weights_classifier_update_last.h5")
        self.pred= self.segmentation.predict(np.expand_dims(self.image, axis=0))
        self.label_pred = (self.classifieur.predict(np.expand_dims(self.pred[0], axis=0))>= self.mythreshold).astype(int)
        self.ax3.imshow(self.pred[0], cmap='gray')
        if np.all(( self.mask == 0)) == True:
            self.ax3.set_ylabel('Clean',color='grey',fontsize=20)
        else:
            if self.label_pred[0] == 0 :
                self.ax3.set_ylabel('benign (not cancer)',color='green',fontsize=15)
            else :
                self.ax3.set_ylabel('malignant (cancer)',color='red',fontsize=15)
