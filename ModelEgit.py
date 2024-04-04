
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox,QTableWidgetItem
from PyQt5 import QtWidgets
from matplotlib import pyplot as plt
from pandas.io import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale, LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle
import seaborn as sn
from tasarimMakine import Ui_MainWindow
import sys
import pandas as pd

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.confision=[]
        self.confisionFold1=[]
        self.confisionFold2=[]
        self.confisionFold3=[]
        self.confisionFold4=[]
        self.setupUi(self)

        self.dosyaAdi = ""
        self.btnveri.clicked.connect(self.veriSetiSec)
        self.hsVeriSetiBoyutu.valueChanged.connect(self.hs_valueChanged)
        self.btnModelEgit.clicked.connect(self.model_Egit)
        self.btnConfusion.clicked.connect(self.confusionMatrix)
        self.btnModelEgitKFold.clicked.connect(self.KFold_Model_Egit)
        self.btnverKFold.clicked.connect(self.veriSetiSecFold)
        self.btnConfusionKFold1.clicked.connect(self.confusionMatrixKFold1)
        self.btnConfusionKFold2.clicked.connect(self.confusionMatrixKFold2)
        self.btnConfusionKFold3.clicked.connect(self.confusionMatrixKFold3)
        self.btnConfusionKFold4.clicked.connect(self.confusionMatrixKFold4)
        self.btnTahminEt.clicked.connect(self.ModelTahmin)
        self.btnveriHoldOut.clicked.connect(self.veriSetiSecHoldOut)
        self.hsVeriSetiBoyutuHoldOut.valueChanged.connect(self.hs_valueChangedHoldOut)
        self.hsVeriSetiBoyutuHoldOutVal.valueChanged.connect(self.hs_valueChangedHoldOutVal)
        self.btnModelEgitHoldOut.clicked.connect(self.ModelEgitHoldOut)
        self.btnConfusionHoldOut.clicked.connect(self.confusionMatrixHoldOut)
        self.btnGirdiTemizle.clicked.connect(self.girdiTemizle)
    def confusionMatrix(self):
        if len(self.confision)!=0:
            df_cm = pd.DataFrame(self.confision, index=[i for i in "6543210"],
                                 columns=[i for i in "6543210"])
            plt.figure(figsize=(14, 9))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixHoldOut(self):
        if len(self.confision) != 0:
            df_cm = pd.DataFrame(self.confision, index=[i for i in "6543210"],
                                 columns=[i for i in "6543210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixKFold1(self):
        if len(self.confisionFold1) != 0:
            df_cm = pd.DataFrame(self.confisionFold1, index=[i for i in "6543210"],
                                 columns=[i for i in "6543210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixKFold2(self):
        if len(self.confisionFold2) != 0:
            df_cm = pd.DataFrame(self.confisionFold2, index=[i for i in "6543210"],
                                 columns=[i for i in "6543210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixKFold3(self):
        if len(self.confisionFold3) != 0:
            df_cm = pd.DataFrame(self.confisionFold3, index=[i for i in "6543210"],
                                 columns=[i for i in "6543210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def confusionMatrixKFold4(self):
        if len(self.confisionFold3) != 0:
            df_cm = pd.DataFrame(self.confisionFold4, index=[i for i in "6543210"],
                                 columns=[i for i in "6543210"])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True,fmt='d')
            plt.show()
        else:
            QMessageBox.about(self, "Hata", "Önce model eğitimini tamamlamalısınız!")
            QMessageBox.setStyleSheet(self, "")
    def hs_valueChanged(self):
        val = "% " + str(self.hsVeriSetiBoyutu.value())
        self.lblBoyut.setText(val)
    def hs_valueChangedHoldOut(self):
        val = "% " + str(self.hsVeriSetiBoyutuHoldOut.value())
        self.lblBoyutHoldOut.setText(val)
    def hs_valueChangedHoldOutVal(self):
        val = "% " + str(self.hsVeriSetiBoyutuHoldOutVal.value())
        self.lblBoyutHoldOutVal.setText(val)
    def veriSetiSec(self):
        dosya_yolu = QFileDialog.getOpenFileName(parent=self, caption="Dosya seç",directory="C:\\Users\\HilalSeyhan\\Desktop\\Makine\\Dataset")
        dosyaUzanti = dosya_yolu[0].split("/")
        dosyaAdi = dosyaUzanti[len(dosyaUzanti) - 1]
        self.dosyaAdi = dosyaAdi
        self.lblVeriSeti.setText("Veri seti adı: " + dosyaAdi)
    def veriSetiSecFold(self):
        dosya_yolu = QFileDialog.getOpenFileName(parent=self, caption="Dosya seç",directory="C:\\Users\\HilalSeyhan\\Desktop\\Makine\\Dataset")
        dosyaUzanti = dosya_yolu[0].split("/")
        dosyaAdi = dosyaUzanti[len(dosyaUzanti) - 1]
        self.dosyaAdi = dosyaAdi
        self.lblVeriSetiKFold.setText("Veri Seti Adı: " + dosyaAdi)
    def veriSetiSecHoldOut(self):
        dosya_yolu = QFileDialog.getOpenFileName(parent=self, caption="Dosya seç",directory="C:\\Users\\HilalSeyhan\\Desktop\\Makine\\Dataset")
        dosyaUzanti = dosya_yolu[0].split("/")
        dosyaAdi = dosyaUzanti[len(dosyaUzanti) - 1]
        self.dosyaAdi = dosyaAdi
        self.lblVeriSetiHoldOut.setText("Veri Seti Adı: " + dosyaAdi)
    def readData(self,dosyaAdi):
        anaKlasor = "./"
        data = pd.read_csv(anaKlasor + "Dataset/" + dosyaAdi,sep=";")
        data.info()
        labels=data.columns
        ozNitelikSayisi = data.shape[1]
        Xveri=data.values
        Yveri=Xveri[:, ozNitelikSayisi - 1]

        LE = LabelEncoder()
        data["Class"] = LE.fit_transform(data["Class"])
        data["Perimeter"] = [float(str(i).replace(",", "")) for i in data["Perimeter"]]
        data["MajorAxisLength"] = [float(str(i).replace(",", "")) for i in data["MajorAxisLength"]]
        data["MinorAxisLength"] = [float(str(i).replace(",", "")) for i in data["MinorAxisLength"]]
        data["AspectRation"] = [float(str(i).replace(",", "")) for i in data["AspectRation"]]
        data["Eccentricity"] = [float(str(i).replace(",", "")) for i in data["Eccentricity"]]
        data["ConvexArea"] = [float(str(i).replace(",", "")) for i in data["ConvexArea"]]
        data["EquivDiameter"] = [float(str(i).replace(",", "")) for i in data["EquivDiameter"]]
        data["Extent"] = [float(str(i).replace(",", "")) for i in data["Extent"]]
        data["Solidity"] = [float(str(i).replace(",", "")) for i in data["Solidity"]]
        data["roundness"] = [float(str(i).replace(",", "")) for i in data["roundness"]]
        data["Compactness"] = [float(str(i).replace(",", "")) for i in data["Compactness"]]
        data["ShapeFactor1"] = [float(str(i).replace(",", "")) for i in data["ShapeFactor1"]]
        data["ShapeFactor2"] = [float(str(i).replace(",", "")) for i in data["ShapeFactor2"]]
        data["ShapeFactor3"] = [float(str(i).replace(",", "")) for i in data["ShapeFactor3"]]
        data["ShapeFactor4"] = [float(str(i).replace(",", "")) for i in data["ShapeFactor4"]]
        veriler = data.values

        y = veriler[:, ozNitelikSayisi - 1]
        X = veriler[:, 0:ozNitelikSayisi - 1]

        column_number = len(labels)-1
        y=np.array(y)

        self.tblX.setRowCount(Xveri.shape[0])
        self.tblX.setColumnCount(column_number)
        self.tblX.setHorizontalHeaderLabels(labels)
        self.tblY.setRowCount(y.shape[0])
        self.tblY.setColumnCount(1)
        labels2 = ['Class']
        self.tblY.setHorizontalHeaderLabels(labels2)
        for i, data in enumerate(Xveri):
            for j in range(0, len(data)):
                self.tblX.setItem(i, j, QTableWidgetItem(str(data[j])))
            self.tblY.setItem(i, 0, QTableWidgetItem(str(Yveri[i])))
        return X, y,labels,labels2
    def func_performansHesapla(self,cm):
        FP=cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        print("TP:", TP)
        print("TN:", TN)
        print("FP:", FP)
        print("FN:", FN)
        accuracy=0
        sensitivity=0
        specificity=0
        for i in range(0,7):
            accuracy += (TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i]) * 100
            sensitivity += TP[i] / (TP[i] + FN[i]) * 100
            specificity += TN[i] / (TN[i] + FP[i]) * 100
        accuracy=accuracy/7
        sensitivity=sensitivity/7
        specificity=specificity/7
        print("Accuracy:", accuracy, " Sen:", sensitivity, " Spe", specificity)
        return accuracy,sensitivity,specificity
    def getIndex(self):
        index = self.cmbModel.currentIndex()
        return  index
    def getIndexTahminModel(self):
        index = self.cmbModelTahmin.currentIndex()
        return  index
    def getIndexFold(self):
        index = self.cmbModelKFold.currentIndex()
        return  index
    def selectedMin(self):
        select = self.rbMinMax.isChecked()
        return select
    def selectedStan(self):
        select = self.rbStan.isChecked()
        return select
    def selectedMinKFold(self):
        select = self.rbMinMaxKFold.isChecked()
        return select
    def selectedStanKFold(self):
        select = self.rbStanKFold.isChecked()
        return select
    def model_Egit(self):
        if self.dosyaAdi != "":
            X, y,labels,labels2 = self.readData(self.dosyaAdi)
            #self.selected()
            tutMin =self.selectedMin()
            tutStan =self.selectedStan()

            if tutMin == True:
                X=minmax_scale(X,feature_range=(0,15))
            if tutStan == True:
                scalar=StandardScaler()
                X=scalar.fit_transform(X)
            sizeValue=float(self.hsVeriSetiBoyutu.value())
            sizeValue=sizeValue/100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizeValue, random_state=42)
            print("xshape",X_train.shape)
            print("yshape",y_train.shape)
            column_number=X_train.shape[1]
            self.tblXTrain.setRowCount(X_train.shape[0])
            self.tblXTrain.setColumnCount(column_number)
            self.tblXTrain.setHorizontalHeaderLabels(labels)
            self.tblYTrain.setRowCount(y_train.shape[0])
            self.tblYTrain.setColumnCount(1)
            self.tblYTrain.setHorizontalHeaderLabels(labels2)
            self.tblXTest.setRowCount(X_test.shape[0])
            self.tblXTest.setColumnCount(column_number)
            self.tblXTest.setHorizontalHeaderLabels(labels)
            self.tblYTest.setRowCount(y_test.shape[0])
            self.tblYTest.setColumnCount(1)
            self.tblYTest.setHorizontalHeaderLabels(labels2)
            print("X:",X_train)
            print("Y:",y_train)

            for i, data in enumerate(X_train):
                for j in range(0, len(data)):
                    self.tblXTrain.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblYTrain.setItem(i, 0, QTableWidgetItem(str(y_train[i])))
            for i, data in enumerate(X_test):
                for j in range(0, len(data)):
                    self.tblXTest.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblYTest.setItem(i, 0, QTableWidgetItem(str(y_test[i])))
            cb = self.getIndex()
            model = RandomForestClassifier()
            if cb == 0:
                model =RandomForestClassifier(n_estimators=40)
            elif cb == 1:
                model = DecisionTreeClassifier()
            elif cb == 2:
                model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            print("Gerçekler")
            print(y_test)
            print("Tahminler")
            print(predict)

            cm = confusion_matrix(y_test, predict)
            self.confision=cm
            print(cm)
            basari = accuracy_score(y_test, predict)
            print("Başarı: ", basari)
            acc,sens,spe = self.func_performansHesapla(cm)
            self.lblACC.setText("Accuracy: {0}".format(acc))
            self.lblSen.setText("Sensitivity: {0}".format(sens))
            self.lblSpe.setText("Specificity: {0}".format(spe))
            if cb == 0:
                mdl=open("./Models/RandomForest.pkl","wb")
                pickle.dump(model,mdl)
            elif cb == 1:
                mdl = open("./Models/DecisionTree.pkl", "wb")
                pickle.dump(model, mdl)
            elif cb == 2:
                mdl = open("./Models/KNN-Classifier.pkl", "wb")
                pickle.dump(model, mdl)
        else:
            QMessageBox.about(self,"Hata","Veri seti seçiniz!")
            QMessageBox.setStyleSheet(self,"")
    def get_score_KFold(self,model,X_train,X_test,y_train,y_test):
        model.fit(X_train,y_train)
        return model.score(X_test,y_test)

    def KFold_Model_Egit(self):
        if self.dosyaAdi != "":
            X, y,labels,labels2 = self.readData(self.dosyaAdi)


            tutMin =self.selectedMinKFold()
            tutStan =self.selectedStanKFold()

            if tutMin == True:
                X=minmax_scale(X,feature_range=(0,15))
            if tutStan == True:
                scalar=StandardScaler()
                X=scalar.fit_transform(X)

            X_train1=[]
            X_test1=[]
            y_train1=[]
            y_test1=[]
            X_train2 = []
            X_test2 = []
            y_train2 = []
            y_test2 = []
            X_train3 = []
            X_test3 = []
            y_train3 = []
            y_test3 = []
            X_train4 = []
            X_test4 = []
            y_train4 = []
            y_test4 = []
            score_Model=[]
            KFold1XTrain=[]
            KFold1XTest=[]
            KFold1YTrain=[]
            KFold1YTest=[]
            KFold2XTrain = []
            KFold2XTest = []
            KFold2YTrain = []
            KFold2YTest = []
            KFold3XTrain = []
            KFold3XTest = []
            KFold3YTrain = []
            KFold3YTest = []
            KFold4XTrain = []
            KFold4XTest = []
            KFold4YTrain = []
            KFold4YTest = []
            model=RandomForestClassifier()
            cb = self.getIndexFold()
            kf = KFold(n_splits=4,shuffle=True)
            kf.get_n_splits(X)
            model=RandomForestClassifier()
            if cb==0:
                model=RandomForestClassifier()
            elif cb==0:
                model=DecisionTreeClassifier()
            elif cb == 1:
                model=KNeighborsClassifier()
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                if i==0:
                    X_train1, X_test1 = X[train_index], X[test_index]
                    y_train1, y_test1 = y[train_index], y[test_index]
                    KFold1XTrain.append(X_train1)
                    KFold1XTest.append(X_test1)
                    KFold1YTrain.append(y_train1)
                    KFold1YTest.append(y_test1)
                    model.fit(X_train1,y_train1)
                    predict = model.predict(X_test1)
                    basari = accuracy_score(y_test1, predict)
                    cm = confusion_matrix(y_test1, predict)
                    self.confisionFold1 = cm
                    score_Model.append(basari)
                elif i ==1:
                    X_train2, X_test2 = X[train_index], X[test_index]
                    y_train2, y_test2 = y[train_index], y[test_index]
                    KFold2XTrain.append(X_train2)
                    KFold2XTest.append(X_test2)
                    KFold2YTrain.append(y_train2)
                    KFold2YTest.append(y_test2)
                    model.fit(X_train2, y_train2)
                    predict = model.predict(X_test2)
                    basari = accuracy_score(y_test2, predict)
                    cm = confusion_matrix(y_test2, predict)
                    self.confisionFold2 = cm
                    score_Model.append(basari)
                elif i ==2:
                    X_train3, X_test3 = X[train_index], X[test_index]
                    y_train3, y_test3 = y[train_index], y[test_index]
                    KFold3XTrain.append(X_train3)
                    KFold3XTest.append(X_test3)
                    KFold3YTrain.append(y_train3)
                    KFold3YTest.append(y_test3)
                    model.fit(X_train3, y_train3)
                    predict = model.predict(X_test3)
                    basari = accuracy_score(y_test3, predict)
                    cm = confusion_matrix(y_test3, predict)
                    self.confisionFold3=cm
                    score_Model.append(basari)
                elif i ==3:
                    X_train4, X_test4 = X[train_index], X[test_index]
                    y_train4, y_test4 = y[train_index], y[test_index]
                    KFold4XTrain.append(X_train4)
                    KFold4XTest.append(X_test4)
                    KFold4YTrain.append(y_train4)
                    KFold4YTest.append(y_test4)
                    model.fit(X_train4, y_train4)
                    predict = model.predict(X_test4)
                    basari = accuracy_score(y_test4, predict)
                    cm = confusion_matrix(y_test4, predict)
                    self.confisionFold4 = cm
                    score_Model.append(basari)

            print(score_Model)
            basariOrt=0
            for i in range(0,len(score_Model)):
                basariOrt += score_Model[i]
            print("Başarı Ortalama",basariOrt/4)
            self.lblFold1.setText("Acc: {0}".format(score_Model[0]))
            self.lblFold2.setText("Acc: {0}".format(score_Model[1]))
            self.lblFold3.setText("Acc: {0}".format(score_Model[2]))
            self.lblFold4.setText("Acc: {0}".format(score_Model[3]))

            KFold1XTrain=np.array(KFold1XTrain)
            KFold1XTest=np.array(KFold1XTest)
            KFold1YTrain=np.array(KFold1YTrain)
            KFold1YTest=np.array(KFold1YTest)
            KFold2XTrain = np.array(KFold2XTrain)
            KFold2XTest = np.array(KFold2XTest)
            KFold2YTrain = np.array(KFold2YTrain)
            KFold2YTest = np.array(KFold2YTest)
            KFold3XTrain = np.array(KFold3XTrain)
            KFold3XTest = np.array(KFold3XTest)
            KFold3YTrain = np.array(KFold3YTrain)
            KFold3YTest = np.array(KFold3YTest)
            KFold4XTrain = np.array(KFold4XTrain)
            KFold4XTest = np.array(KFold4XTest)
            KFold4YTrain = np.array(KFold4YTrain)
            KFold4YTest = np.array(KFold4YTest)
            column_number=X_train1.shape[1]
            self.tblKFold1XTrain.setRowCount(KFold1XTrain.shape[1])
            self.tblKFold1XTrain.setColumnCount(column_number)
            self.tblKFold1XTrain.setHorizontalHeaderLabels(labels)
            self.tblKFold1YTrain.setRowCount(KFold1YTrain.shape[1])
            self.tblKFold1YTrain.setColumnCount(1)
            self.tblKFold1YTrain.setHorizontalHeaderLabels(labels2)
            self.tblKFold1XTest.setRowCount(KFold1XTest.shape[1])
            self.tblKFold1XTest.setColumnCount(column_number)
            self.tblKFold1XTest.setHorizontalHeaderLabels(labels)
            self.tblKFold1YTest.setRowCount(KFold1YTest.shape[1])
            self.tblKFold1YTest.setColumnCount(1)
            self.tblKFold1YTest.setHorizontalHeaderLabels(labels2)
            self.tblKFold2XTrain.setRowCount(KFold2XTrain.shape[1])
            self.tblKFold2XTrain.setColumnCount(column_number)
            self.tblKFold2XTrain.setHorizontalHeaderLabels(labels)
            self.tblKFold2XTest.setRowCount(KFold2XTest.shape[1])
            self.tblKFold2XTest.setColumnCount(column_number)
            self.tblKFold2XTest.setHorizontalHeaderLabels(labels)
            self.tblKFold2YTest.setRowCount(KFold2YTest.shape[1])
            self.tblKFold2YTest.setColumnCount(column_number)
            self.tblKFold2YTest.setHorizontalHeaderLabels(labels)
            self.tblKFold2YTrain.setRowCount(KFold2YTrain.shape[1])
            self.tblKFold2YTrain.setColumnCount(column_number)
            self.tblKFold2YTrain.setHorizontalHeaderLabels(labels)
            self.tblKFold3XTrain.setRowCount(KFold3XTrain.shape[1])
            self.tblKFold3XTrain.setColumnCount(column_number)
            self.tblKFold3XTrain.setHorizontalHeaderLabels(labels)
            self.tblKFold3XTest.setRowCount(KFold3XTest.shape[1])
            self.tblKFold3XTest.setColumnCount(column_number)
            self.tblKFold3XTest.setHorizontalHeaderLabels(labels)
            self.tblKFold3YTest.setRowCount(KFold3YTest.shape[1])
            self.tblKFold3YTest.setColumnCount(column_number)
            self.tblKFold3YTest.setHorizontalHeaderLabels(labels)
            self.tblKFold3YTrain.setRowCount(KFold3YTrain.shape[1])
            self.tblKFold3YTrain.setColumnCount(column_number)
            self.tblKFold3YTrain.setHorizontalHeaderLabels(labels)
            self.tblKFold4XTrain.setRowCount(KFold4XTrain.shape[1])
            self.tblKFold4XTrain.setColumnCount(column_number)
            self.tblKFold4XTrain.setHorizontalHeaderLabels(labels)
            self.tblKFold4XTest.setRowCount(KFold4XTest.shape[1])
            self.tblKFold4XTest.setColumnCount(column_number)
            self.tblKFold4XTest.setHorizontalHeaderLabels(labels)
            self.tblKFold4YTest.setRowCount(KFold4YTest.shape[1])
            self.tblKFold4YTest.setColumnCount(column_number)
            self.tblKFold4YTest.setHorizontalHeaderLabels(labels)
            self.tblKFold4YTrain.setRowCount(KFold4YTrain.shape[1])
            self.tblKFold4YTrain.setColumnCount(column_number)
            self.tblKFold4YTrain.setHorizontalHeaderLabels(labels)
            print("X:",KFold1XTrain)
            print("Y:",KFold1YTrain)
            print(X_train1)
            for i, data in enumerate(X_train1):
                for j in range(0, len(data)):
                    self.tblKFold1XTrain.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblKFold1YTrain.setItem(i, 0, QTableWidgetItem(str(y_train1[i])))
            for i, data in enumerate(X_test1):
                for j in range(0, len(data)):
                    self.tblKFold1XTest.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblKFold1YTest.setItem(i, 0, QTableWidgetItem(str(y_test1[i])))
            for i, data in enumerate(X_train2):
                for j in range(0, len(data)):
                    self.tblKFold2XTrain.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblKFold2YTrain.setItem(i, 0, QTableWidgetItem(str(y_train2[i])))
            for i, data in enumerate(X_test2):
                for j in range(0, len(data)):
                    self.tblKFold2XTest.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblKFold2YTest.setItem(i, 0, QTableWidgetItem(str(y_test2[i])))
            for i, data in enumerate(X_train3):
                for j in range(0, len(data)):
                    self.tblKFold3XTrain.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblKFold3YTrain.setItem(i, 0, QTableWidgetItem(str(y_train3[i])))
            for i, data in enumerate(X_test3):
                for j in range(0, len(data)):
                    self.tblKFold3XTest.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblKFold3YTest.setItem(i, 0, QTableWidgetItem(str(y_test3[i])))
            for i, data in enumerate(X_train4):
                for j in range(0, len(data)):
                    self.tblKFold4XTrain.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblKFold4YTrain.setItem(i, 0, QTableWidgetItem(str(y_train4[i])))
            for i, data in enumerate(X_test4):
                for j in range(0, len(data)):
                    self.tblKFold4XTest.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblKFold4YTest.setItem(i, 0, QTableWidgetItem(str(y_test4[i])))

            acc1, sens1, spe1 = self.func_performansHesapla(self.confisionFold1)
            acc2, sens2, spe2 = self.func_performansHesapla(self.confisionFold2)
            acc3, sens3, spe3 = self.func_performansHesapla(self.confisionFold3)
            acc4, sens4, spe4 = self.func_performansHesapla(self.confisionFold4)
            acc=(acc1+acc2+acc3+acc4)/4
            sens=(sens1+sens2+sens3+sens4)/4
            spe=(spe1+spe2+spe3+spe4)/4

            self.lblACCKFold.setText("Ortalama Accuracy: {0}".format(acc))
            self.lblSenKFold.setText("Ortalama Sensitivity: {0}".format(sens))
            self.lblSpeKFold.setText("Ortalama Specificity: {0}".format(spe))
            if cb == 0:
                mdl = open("./Models/RandomForest.pkl", "wb")
                pickle.dump(model, mdl)
            elif cb == 1:
                mdl = open("./Models/DecisionTree.pkl", "wb")
                pickle.dump(model, mdl)
            elif cb == 2:
                mdl = open("./Models/KNN-Classifier.pkl", "wb")
                pickle.dump(model, mdl)
        else:
            QMessageBox.about(self,"Hata","Veri seti seçiniz!")
            QMessageBox.setStyleSheet(self,"")
    def getIndexHoldOut(self):
        index = self.cmbModelHoldOut.currentIndex()
        return index
    def selectedMinHoldOut(self):
        select = self.rbMinMaxHoldOut.isChecked()
        return select
    def selectedStanHoldOut(self):
        select = self.rbStanHoldOut.isChecked()
        return select
    def ModelEgitHoldOut(self):
        if self.dosyaAdi != "":
            X, y, labels, labels2 = self.readData(self.dosyaAdi)
            tutMin = self.selectedMinHoldOut()
            tutStan = self.selectedStanHoldOut()
            if tutMin == True:
                X = minmax_scale(X, feature_range=(0, 15))
            if tutStan == True:
                scalar = StandardScaler()
                X = scalar.fit_transform(X)
            sizeValue = float(self.hsVeriSetiBoyutuHoldOut.value())
            sizeValue = sizeValue / 100
            sizeValueVal=float(self.hsVeriSetiBoyutuHoldOutVal.value())
            sizeValueVal=sizeValueVal/100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=sizeValue, random_state=42)
            X_val,X_hold,y_val,y_hold=train_test_split(X_test,y_test,test_size=sizeValueVal,random_state=42)
            print("xshape", X_val.shape)
            print("yshape", X_hold.shape)
            column_number = X_train.shape[1]
            self.tblXTrainHoldOut.setRowCount(X_train.shape[0])
            self.tblXTrainHoldOut.setColumnCount(column_number)
            self.tblXTrainHoldOut.setHorizontalHeaderLabels(labels)
            self.tblYTrainHoldOut.setRowCount(y_train.shape[0])
            self.tblYTrainHoldOut.setColumnCount(1)
            self.tblYTrainHoldOut.setHorizontalHeaderLabels(labels2)
            self.tblXValHoldOut.setRowCount(X_val.shape[0])
            self.tblXValHoldOut.setColumnCount(column_number)
            self.tblXValHoldOut.setHorizontalHeaderLabels(labels)
            self.tblYValHoldOut.setRowCount(y_val.shape[0])
            self.tblYValHoldOut.setColumnCount(1)
            self.tblYValHoldOut.setHorizontalHeaderLabels(labels2)
            self.tblXHoldOut.setRowCount(X_hold.shape[0])
            self.tblXHoldOut.setColumnCount(column_number)
            self.tblXHoldOut.setHorizontalHeaderLabels(labels)
            self.tblYHoldOut.setRowCount(y_hold.shape[0])
            self.tblYHoldOut.setColumnCount(1)
            self.tblYHoldOut.setHorizontalHeaderLabels(labels2)
            for i, data in enumerate(X_train):
                for j in range(0, len(data)):
                    self.tblXTrainHoldOut.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblYTrainHoldOut.setItem(i, 0, QTableWidgetItem(str(y_train[i])))
            for i, data in enumerate(X_val):
                for j in range(0, len(data)):
                    self.tblXValHoldOut.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblYValHoldOut.setItem(i, 0, QTableWidgetItem(str(y_val[i])))
            for i, data in enumerate(X_hold):
                for j in range(0, len(data)):
                    self.tblXHoldOut.setItem(i, j, QTableWidgetItem(str(data[j])))
                self.tblYHoldOut.setItem(i, 0, QTableWidgetItem(str(y_hold[i])))
            cb = self.getIndexHoldOut()
            model = RandomForestClassifier()
            if cb == 0:
                model = RandomForestClassifier(n_estimators=40)
            elif cb == 1:
                model = DecisionTreeClassifier()
            elif cb == 2:
                model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            predict = model.predict(X_hold)
            print("Gerçekler")
            print(y_hold)
            print("Tahminler")
            print(predict)

            cm = confusion_matrix(y_hold, predict)
            self.confision = cm
            print(cm)
            basari = accuracy_score(y_hold, predict)
            print("Başarı: ", basari)
            acc, sens, spe = self.func_performansHesapla(cm)
            self.lblACCHoldOut.setText("Accuracy: {0}".format(acc))
            self.lblSenHoldOut.setText("Sensitivity: {0}".format(sens))
            self.lblSpeHoldOut.setText("Specificity: {0}".format(spe))
            if cb == 0:
                mdl = open("./Models/RandomForest.pkl", "wb")
                pickle.dump(model, mdl)
            elif cb == 1:
                mdl = open("./Models/DecisionTree.pkl", "wb")
                pickle.dump(model, mdl)
            elif cb == 2:
                mdl = open("./Models/KNN-Classifier.pkl", "wb")
                pickle.dump(model, mdl)
        else:
            QMessageBox.about(self, "Hata", "Veri seti seçiniz!")
            QMessageBox.setStyleSheet(self, "")
    def getIndexAgri(self):
        index = self.cmbGogusAgriTuru.currentIndex()
        return  index
    def getIndexSTegimi(self):
        index = self.cmbSTegimi.currentIndex()
        return  index
    def getIndexkalpECG(self):
        index = self.cmbKalpECG.currentIndex()
        return  index
    def girdiTemizle(self):
        self.leAlan.setText("")
        self.leCevre.setText("")
        self.leAnaEksen.setText("")
        self.leKucukEksen.setText("")
        self.leEnBoyOrani.setText("")
        self.leEksantriklik.setText("")
        self.leDisBukeyAlan.setText("")
        self.leEsdegerCap.setText("")
        self.leKapsam.setText("")
        self.leSaglamlik.setText("")
        self.leYuvarlaklik.setText("")
        self.leKompaktlk.setText("")
        self.leSekilFaktoru1.setText("")
        self.leSekilFaktoru2.setText("")
        self.leSekilFaktoru3.setText("")
        self.leSekilFaktoru4.setText("")
    def ModelTahmin(self):
        try:
            mdl=self.getIndexTahminModel()
            model=None
            sonuc=None
            if mdl == 0:
                model = pickle.load(open("./Models/RandomForest.pkl","rb"))
            elif mdl == 1:
                model = pickle.load(open("./Models/DecisionTree.pkl","rb"))
            elif mdl == 2:
                model = pickle.load(open("./Models/KNN-Classifier.pkl","rb"))
            if self.leTahminDegerleri.text()=="":
                alan=float(self.leAlan.text().replace(",",""))
                cevre=float(self.leCevre.text().replace(",",""))
                anaEksen=float(self.leAnaEksen.text().replace(",",""))
                kucukEksen=float(self.leKucukEksen.text().replace(",",""))
                enBoyOrani=float(self.leEnBoyOrani.text().replace(",",""))
                eksantriklik=float(self.leEksantriklik.text().replace(",",""))
                disBukey=float(self.leDisBukeyAlan.text().replace(",",""))
                esDegerCap=float(self.leEsdegerCap.text().replace(",",""))
                kapsam=float(self.leKapsam.text().replace(",",""))
                saglamlik=float(self.leSaglamlik.text().replace(",",""))
                yuvarlak=float(self.leYuvarlaklik.text().replace(",",""))
                kompaktlik=float(self.leKompaktlk.text().replace(",",""))
                sekilFaktoru1=float(self.leSekilFaktoru1.text().replace(",",""))
                sekilFaktoru2=float(self.leSekilFaktoru2.text().replace(",",""))
                sekilFaktoru3=float(self.leSekilFaktoru3.text().replace(",",""))
                sekilFaktoru4=float(self.leSekilFaktoru4.text().replace(",",""))
                tahmin=[]
                tahmin.append(alan)
                tahmin.append(cevre)
                tahmin.append(anaEksen)
                tahmin.append(kucukEksen)
                tahmin.append(enBoyOrani)
                tahmin.append(eksantriklik)
                tahmin.append(disBukey)
                tahmin.append(esDegerCap)
                tahmin.append(kapsam)
                tahmin.append(saglamlik)
                tahmin.append(yuvarlak)
                tahmin.append(kompaktlik)
                tahmin.append(sekilFaktoru1)
                tahmin.append(sekilFaktoru2)
                tahmin.append(sekilFaktoru3)
                tahmin.append(sekilFaktoru4)
                tahmin=np.array(tahmin)
                sonDizi=[]
                sonDizi.append(tahmin)
                sonuc = model.predict(sonDizi)
            else:
                degerler=self.leTahminDegerleri.text()
                degerlerDizi=degerler.split(" ")
                degerlerDizi = [float(str(i).replace(",", "")) for i in degerlerDizi]
                sonDizi = []
                sonDizi.append(degerlerDizi)
                sonuc=model.predict(sonDizi)
            if sonuc == 0:
                self.lblSonuc.setText("Barbunya Fasülyesi")
                self.lblSonuc.setStyleSheet("background-color: rgb(255, 0, 255);")
            elif sonuc == 1:
                self.lblSonuc.setText("Bombaya Fasülyesi")
                self.lblSonuc.setStyleSheet("background-color: rgb(255, 0, 255);")
            elif sonuc == 2:
                self.lblSonuc.setText("Çalı Fasülyesi")
                self.lblSonuc.setStyleSheet("background-color: rgb(255, 0, 255);")
            elif sonuc == 3:
                self.lblSonuc.setText("Dermason Fasülyesi")
                self.lblSonuc.setStyleSheet("background-color: rgb(255, 0, 255);")
            elif sonuc == 4:
                self.lblSonuc.setText("Horoz Fasülyesi")
                self.lblSonuc.setStyleSheet("background-color: rgb(255, 0, 255);")
            elif sonuc == 5:
                self.lblSonuc.setText("Şeker Fasülyesi")
                self.lblSonuc.setStyleSheet("background-color: rgb(255, 0, 255);")
            elif sonuc == 6:
                self.lblSonuc.setText("Sıra Fasülyesi")
                self.lblSonuc.setStyleSheet("background-color: rgb(255, 0, 255);")
        except:
            QMessageBox.about(self, "Hata", "Girdiğiniz verilerin doğruluğundan emin olun!")
            QMessageBox.setStyleSheet(self, "")



app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()