# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import venv.Osah.model_view_pandas as pandas_view_model
import math
import numpy as np
import venv.Osah.residuals_model as res_model


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(657, 622)
        MainWindow.setMaximumSize(QtCore.QSize(657, 622))
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout_7.addWidget(self.label, 0, 0, 1, 2)
        self.csv_file_path = QtWidgets.QLabel(self.centralwidget)
        self.csv_file_path.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.csv_file_path.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.csv_file_path.setObjectName("csv_file_path")
        self.gridLayout_7.addWidget(self.csv_file_path, 0, 2, 1, 1)
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tt_num_of_iterations = QtWidgets.QLineEdit(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tt_num_of_iterations.sizePolicy().hasHeightForWidth())
        self.tt_num_of_iterations.setSizePolicy(sizePolicy)
        self.tt_num_of_iterations.setObjectName("tt_num_of_iterations")
        self.gridLayout_5.addWidget(self.tt_num_of_iterations, 0, 0, 1, 1)
        self.btn_train_model = QtWidgets.QPushButton(self.groupBox_5)
        self.btn_train_model.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_train_model.setObjectName("btn_train_model")
        self.gridLayout_5.addWidget(self.btn_train_model, 0, 1, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_5, 5, 0, 1, 4)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_7.addWidget(self.progressBar, 6, 0, 1, 4)
        self.lbl_wait_period = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_wait_period.sizePolicy().hasHeightForWidth())
        self.lbl_wait_period.setSizePolicy(sizePolicy)
        self.lbl_wait_period.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_wait_period.setObjectName("lbl_wait_period")
        self.gridLayout_7.addWidget(self.lbl_wait_period, 7, 0, 1, 4)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lat = QtWidgets.QCheckBox(self.groupBox_2)
        self.lat.setObjectName("lat")
        self.gridLayout_2.addWidget(self.lat, 0, 0, 1, 1)
        self.lon = QtWidgets.QCheckBox(self.groupBox_2)
        self.lon.setObjectName("lon")
        self.gridLayout_2.addWidget(self.lon, 1, 0, 1, 1)
        self.height = QtWidgets.QCheckBox(self.groupBox_2)
        self.height.setObjectName("height")
        self.gridLayout_2.addWidget(self.height, 2, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_2, 1, 2, 2, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.rad_zwd = QtWidgets.QRadioButton(self.groupBox_3)
        self.rad_zwd.setObjectName("rad_zwd")
        self.gridLayout_3.addWidget(self.rad_zwd, 2, 0, 1, 1)
        self.rad_zhd = QtWidgets.QRadioButton(self.groupBox_3)
        self.rad_zhd.setObjectName("rad_zhd")
        self.gridLayout_3.addWidget(self.rad_zhd, 1, 0, 1, 1)
        self.rad_ztd = QtWidgets.QRadioButton(self.groupBox_3)
        self.rad_ztd.setObjectName("rad_ztd")
        self.gridLayout_3.addWidget(self.rad_ztd, 0, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_3, 1, 3, 2, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.linear_act_fun = QtWidgets.QRadioButton(self.groupBox_6)
        self.linear_act_fun.setObjectName("linear_act_fun")
        self.gridLayout_4.addWidget(self.linear_act_fun, 0, 1, 1, 1)
        self.relu_act_fun = QtWidgets.QRadioButton(self.groupBox_6)
        self.relu_act_fun.setObjectName("relu_act_fun")
        self.gridLayout_4.addWidget(self.relu_act_fun, 0, 0, 1, 1)
        self.softmax_act_fun = QtWidgets.QRadioButton(self.groupBox_6)
        self.softmax_act_fun.setObjectName("softmax_act_fun")
        self.gridLayout_4.addWidget(self.softmax_act_fun, 1, 0, 1, 1)
        self.sigmoid_act_fun = QtWidgets.QRadioButton(self.groupBox_6)
        self.sigmoid_act_fun.setObjectName("sigmoid_act_fun")
        self.gridLayout_4.addWidget(self.sigmoid_act_fun, 2, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_6, 3, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.chk_temp = QtWidgets.QCheckBox(self.groupBox)
        self.chk_temp.setObjectName("chk_temp")
        self.gridLayout.addWidget(self.chk_temp, 1, 0, 1, 1)
        self.chk_pressure = QtWidgets.QCheckBox(self.groupBox)
        self.chk_pressure.setObjectName("chk_pressure")
        self.gridLayout.addWidget(self.chk_pressure, 0, 0, 1, 1)
        self.rad_water_vapor = QtWidgets.QRadioButton(self.groupBox)
        self.rad_water_vapor.setObjectName("rad_water_vapor")
        self.gridLayout.addWidget(self.rad_water_vapor, 3, 0, 1, 1)
        self.chk_mean_remp = QtWidgets.QCheckBox(self.groupBox)
        self.chk_mean_remp.setObjectName("chk_mean_remp")
        self.gridLayout.addWidget(self.chk_mean_remp, 4, 0, 1, 1)
        self.chk_water_vapour_lapse = QtWidgets.QCheckBox(self.groupBox)
        self.chk_water_vapour_lapse.setObjectName("chk_water_vapour_lapse")
        self.gridLayout.addWidget(self.chk_water_vapour_lapse, 5, 0, 1, 1)
        self.rad_rhd = QtWidgets.QRadioButton(self.groupBox)
        self.rad_rhd.setObjectName("rad_rhd")
        self.gridLayout.addWidget(self.rad_rhd, 2, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox, 1, 0, 2, 2)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.pred_ztd = QtWidgets.QRadioButton(self.groupBox_4)
        self.pred_ztd.setObjectName("pred_ztd")
        self.gridLayout_6.addWidget(self.pred_ztd, 0, 0, 1, 1)
        self.pred_zhd = QtWidgets.QRadioButton(self.groupBox_4)
        self.pred_zhd.setObjectName("pred_zhd")
        self.gridLayout_6.addWidget(self.pred_zhd, 1, 0, 1, 1)
        self.pred_zwd = QtWidgets.QRadioButton(self.groupBox_4)
        self.pred_zwd.setObjectName("pred_zwd")
        self.gridLayout_6.addWidget(self.pred_zwd, 2, 0, 1, 1)
        self.pred_ztd_res = QtWidgets.QRadioButton(self.groupBox_4)
        self.pred_ztd_res.setObjectName("pred_ztd_res")
        self.gridLayout_6.addWidget(self.pred_ztd_res, 3, 0, 1, 1)
        self.pred_zhd_res = QtWidgets.QRadioButton(self.groupBox_4)
        self.pred_zhd_res.setObjectName("pred_zhd_res")
        self.gridLayout_6.addWidget(self.pred_zhd_res, 4, 0, 1, 1)
        self.pred_zwd_res = QtWidgets.QRadioButton(self.groupBox_4)
        self.pred_zwd_res.setObjectName("pred_zwd_res")
        self.gridLayout_6.addWidget(self.pred_zwd_res, 5, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_4, 3, 2, 1, 1)
        self.pred_dataset = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pred_dataset.sizePolicy().hasHeightForWidth())
        self.pred_dataset.setSizePolicy(sizePolicy)
        self.pred_dataset.setObjectName("pred_dataset")
        self.gridLayout_7.addWidget(self.pred_dataset, 0, 3, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 657, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionImport_File_csv = QtWidgets.QAction(MainWindow)
        self.actionImport_File_csv.setObjectName("actionImport_File_csv")
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionContact_developer = QtWidgets.QAction(MainWindow)
        self.actionContact_developer.setObjectName("actionContact_developer")
        self.actionLoad_Model = QtWidgets.QAction(MainWindow)
        self.actionLoad_Model.setObjectName("actionLoad_Model")
        self.actionImport_predicting_data_csv = QtWidgets.QAction(MainWindow)
        self.actionImport_predicting_data_csv.setObjectName("actionImport_predicting_data_csv")
        self.menuFile.addAction(self.actionImport_File_csv)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionImport_predicting_data_csv)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad_Model)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionClose)
        self.menuHelp.addAction(self.actionContact_developer)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.lbl_wait_period.setVisible(False)
        self.progressBar.setVisible(False)
        self.chk_mean_remp.setDisabled(True)
        self.chk_water_vapour_lapse.setDisabled(True)
        self.pred_dataset.setVisible(False)

        self.actionImport_File_csv.triggered.connect(self.openCSVfile)
        self.actionLoad_Model.triggered.connect(self.loadModel)
        self.actionImport_predicting_data_csv.triggered.connect(self.loadDataset)
        self.btn_train_model.clicked.connect(self.featureSelection)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Deep Neural Network Using Tensorflow"))
        self.label.setText(_translate("MainWindow", "Select important parameters for Model training"))
        self.csv_file_path.setText(_translate("MainWindow", ""))
        self.groupBox_5.setTitle(_translate("MainWindow", "Enter required number of Iterations / Epochs"))
        self.btn_train_model.setText(_translate("MainWindow", "Train Model"))
        self.lbl_wait_period.setText(_translate("MainWindow", "Wait period"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2 Positional Data"))
        self.lat.setText(_translate("MainWindow", "Latitude [in degrees]"))
        self.lon.setText(_translate("MainWindow", "Longitude [in dgrees]"))
        self.height.setText(_translate("MainWindow", "Ellipsoidal Height [in meters]"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Zenith Delays"))
        self.rad_zwd.setText(_translate("MainWindow", "ZWD [m]"))
        self.rad_zhd.setText(_translate("MainWindow", "ZHD [m]"))
        self.rad_ztd.setText(_translate("MainWindow", "ZTD [m]"))
        self.groupBox_6.setTitle(_translate("MainWindow", "4. Select required Activation Function"))
        self.linear_act_fun.setText(_translate("MainWindow", "Linear"))
        self.relu_act_fun.setText(_translate("MainWindow", "Relu"))
        self.softmax_act_fun.setText(_translate("MainWindow", "Softmax"))
        self.sigmoid_act_fun.setText(_translate("MainWindow", "Sigmoid"))
        self.groupBox.setTitle(_translate("MainWindow", "1. ATMOSPHERIC PARAMETERS"))
        self.chk_temp.setText(_translate("MainWindow", "Temperature"))
        self.chk_pressure.setText(_translate("MainWindow", "Pressure [hpa]"))
        self.rad_water_vapor.setText(_translate("MainWindow", "Water Vapour partial pressure (e) in [hpa]"))
        self.chk_mean_remp.setText(_translate("MainWindow", "Mean Temperature (Tm) in [K]"))
        self.chk_water_vapour_lapse.setText(_translate("MainWindow", "Water vapour Lapse Rate (Lambda)"))
        self.rad_rhd.setText(_translate("MainWindow", "Relative Humidity"))
        self.groupBox_4.setTitle(_translate("MainWindow", "5. What are you predicting ?"))
        self.pred_ztd.setText(_translate("MainWindow", "ZTD "))
        self.pred_zhd.setText(_translate("MainWindow", "ZHD"))
        self.pred_zwd.setText(_translate("MainWindow", "ZWD"))
        self.pred_ztd_res.setText(_translate("MainWindow", "ZTD residuals"))
        self.pred_zhd_res.setText(_translate("MainWindow", "ZHD residuals"))
        self.pred_zwd_res.setText(_translate("MainWindow", "ZWD residuals"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionImport_File_csv.setText(_translate("MainWindow", "Import File (.csv)"))
        self.actionImport_predicting_data_csv.setText(_translate("MainWindow", "Import Predicting dataset (.csv)"))
        self.actionLoad_Model.setText(_translate("MainWindow", "Load Model"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionContact_developer.setText(_translate("MainWindow", "Contact developer"))

    def loadDataset(self):

        dataset = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "prediction dataset","c\\", "*.csv")
        with open(dataset[0], "r") as d:
            d.read()
        predicting_dataset_path = dataset[0]
        self.pred_dataset.setText(predicting_dataset_path)

    def loadModel(self):
        model = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, 'model', 'c\\', '*.meta')
        with open(model[0], 'r') as m:
            self.model_name = m.read()


    def listConv(self, s):
        string = " "
        return string.join(s)

    # function open csv file
    def openCSVfile(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, 'Open .csv file only', 'c\\',
                                                         'CSV file (*.csv)')
        with open(filename[0], 'r') as f:
            filepath = f.read()
            self.csv_file_path.setText(filename[0])
            self.csv_file_path.setVisible(False)
            self.read_csv(filename[0])

    def read_csv(self, path):
        self.data = pd.read_csv(path)

        self.data.dropna()
        self.csv_dialog = QtWidgets.QDialog(self.centralwidget)
        self.csv_dialog.setWindowTitle("CSV Reader")
        g_layout = QtWidgets.QGridLayout(self.csv_dialog)
        csv_table = QtWidgets.QTableView(self.csv_dialog)
        csv_table.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        model = pandas_view_model.pandasModel(self.data)
        csv_table.setModel(model)
        g_layout.addWidget(csv_table)

        csv_table.show()
        self.csv_dialog.exec_()

    # Training Functions
    # Training parameters (Pressure, Temperature, Water Vopour,Relative Humidity, Mean Temperature, Water vapor lapse,
    # latitude, logitude, ellip_height,ztd,zhd,zwd, acti_funcs( relu,softmax,linear,sigmoid)
    def featureSelection(self):

        if self.csv_file_path.text() is "" or self.pred_dataset.text() is "":
            QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                           "Import CSV File (Training and predicting datasets) ",
                                           QtWidgets.QMessageBox.Ok)

        else:

            if (
                    not self.chk_temp.isChecked() and not self.chk_water_vapour_lapse.isChecked() and not self.chk_mean_remp.isChecked() and not
            self.chk_pressure.isChecked() and not self.lat.isChecked() and not self.lon.isChecked() and not self.height.isChecked() and not
            self.rad_ztd.isChecked() and not self.rad_rhd.isChecked() and not self.rad_zwd.isChecked() and not self.rad_water_vapor.isChecked()
                    and not self.relu_act_fun.isChecked() and not self.linear_act_fun.isChecked() and not self.softmax_act_fun.isChecked() and not self.sigmoid_act_fun.isChecked()
                    and not self.pred_zwd_res.isChecked() and not self.pred_zwd.isChecked() and not self.pred_zhd_res.isChecked() and not self.pred_zhd.isChecked() and not
            self.pred_ztd_res.isChecked() and not self.pred_zwd_res.isChecked()):


                dt = pd.read_csv(self.csv_file_path.text())
                df = dt.iloc[:, 0:int(len(dt.columns) - 1)]
                data = res_model.model(dt)
                desc_column = dt.columns.values
                data.display_plot(desc_column)
                data.data_description(df.columns.values)
                data.build_model().summary()

                self.history, self.eval, self.pred = data.train(pred_data=self.pred_dataset.text())
                self.display_history()

               # self.write_pred()

               # self.display_pred_history()


            else:

                if self.tt_num_of_iterations.text() is "":
                    QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                   "Number of Iterations cannot be empty ",
                                                   QtWidgets.QMessageBox.Ok)
                else:

                    self.train_features = []
                    self.train_labels = []

                    # latitude feature
                    if self.lat.isChecked():
                        try:
                            if 'Lat' in self.data.columns:
                                self.latitude = self.data['Lat']  # Latitude
                                self.train_features.append(self.latitude)
                                self.train_labels.append(['Lat'])


                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "Lat Column is not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.latitude)
                                self.train_labels.remove(['Lat'])
                        except RuntimeError as err:
                            print(err)

                    #         # Longitude feature
                    if self.lon.isChecked():
                        try:
                            if 'Lon' in self.data.columns:
                                self.longitude = self.data['Lon']  # Longitude
                                self.train_features.append(self.longitude)
                                self.train_labels.append(['Lon'])

                            else:
                                self.lon.setChecked(False)
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "Lon Column is not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.longitude)
                                self.train_labels.remove(['Lat'])
                        except RuntimeError as err:
                            print(err)

                    # Height feature
                    if self.height.isChecked():
                        try:
                            if 'h' in self.data.columns:
                                self.h = self.data['h']
                                self.train_features.append(self.h)
                                self.train_labels.append(['h'])

                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "h Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.h)
                                self.train_labels.remove(['h'])
                        except RuntimeError as err:
                            print(err)

                    # Pressure feature
                    if self.chk_pressure.isChecked():
                        try:
                            if 'P' in self.data.columns:
                                self.P = self.data['P']
                                self.train_features.append(self.P)
                                self.train_labels.append(['P'])

                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "P Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.P)
                                self.train_labels.remove(['P'])
                        except RuntimeError as err:
                            print(err)

                    # Temperature Feature
                    if self.chk_temp.isChecked():
                        try:
                            if 'T' in self.data.columns:
                                self.T = self.data['T']
                                self.train_features.append(self.T)
                                self.train_labels.append(['T'])

                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "T Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.T)
                                self.train_labels.remove(['T'])
                        except RuntimeError as err:
                            print(err)

                    # Water Vapour Feature
                    if self.rad_water_vapor.isChecked():
                        try:
                            if 'e' in self.data.columns:
                                self.e = np.array(self.data['e'])
                                self.train_features.append(self.e)
                                self.train_labels.append(['e'])


                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "e Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.e)
                                self.train_labels.remove(['e'])
                        except RuntimeError as err:
                            print(err)

                    # Calculate Humidity (e) if relative humidity checkbox is selected
                    if self.rad_rhd.isChecked():
                        try:
                            if 'e' in self.data.columns:
                                self.e = self.data['e']
                                self.T = np.array(self.data['T'])
                                exp_num = []
                                temp = self.T + 273.15
                                for i in range(len(temp)):
                                    num = 0.01 * self.e[i] * math.exp(
                                        -37.2465 + 0.213166 * temp[i] - 0.000256908 * math.pow(temp[i], 2))
                                    exp_num.append(num)

                                self.e_humidity = exp_num
                                # print(self.e_humidity)
                                self.train_features.append(self.e_humidity)
                                self.train_labels.append(['e_humid'])
                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "e Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.e_humidity)
                                self.train_labels.remove(['e_humid'])
                        except RuntimeError as err:
                            print(err)

                    # Zenith Delays
                    # ZTD_VMF feature
                    if self.rad_ztd.isChecked():
                        try:
                            if 'ZTD_vmf' in self.data.columns:
                                self.ZTD_vmf = self.data['ZTD_vmf']
                                self.train_features.append(self.ZTD_vmf)
                                self.train_labels.append(['ZTD_vmf'])

                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "ZTD_vmf Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.ZTD_vmf)
                                self.train_labels.remove(['ZTD_vmf'])
                        except RuntimeError as err:
                            print(err)
                    if self.rad_zhd.isChecked():
                        try:
                            if 'ZHD_vmf' in self.data.columns:
                                self.ZHD_vmf = self.data['ZHD_vmf']
                                self.train_features.append(self.ZHD_vmf)
                                self.train_labels.append(['ZHD_vmf'])

                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "ZHD_vmf Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.ZTD_vmf)
                                self.train_labels.remove(['ZHD_vmf'])
                        except RuntimeError as err:
                            print(err)
                    if self.rad_zwd.isChecked():
                        try:
                            if 'ZWD_vmf' in self.data.columns:
                                self.ZWD_vmf = self.data['ZWD_vmf']
                                self.train_features.append(self.ZWD_vmf)
                                self.train_labels.append(['ZWD_vmf'])

                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "ZWD_vmf Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.ZWD_vmf)
                                self.train_labels.remove(['ZWD_vmf'])
                        except RuntimeError as err:
                            print(err)
                    # End of Zenith Delays

                    # Predicting/Target Features
                    # Residuals
                    # RES Feature
                    if self.pred_ztd_res.isChecked():
                        try:
                            if 'RES' in self.data.columns:
                                self.RES = self.data['RES']  # Longitude
                                self.train_features.append(self.RES)
                                self.train_labels.append(['RES'])
                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "RES Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.RES)
                                self.train_labels.remove(['RES'])
                        except RuntimeError as err:
                            print(err)
                    if self.pred_zhd_res.isChecked():
                        try:
                            if 'RES' in self.data.columns:
                                self.RES = self.data['RES']
                                self.train_features.append(self.RES)
                                self.train_labels.append(['RES'])

                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "RES Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.RES)
                                self.train_labels.remove(['RES'])
                        except RuntimeError as err:
                            print(err)
                    if self.pred_zwd_res.isChecked():
                        try:
                            if 'RES' in self.data.columns:
                                self.RES = self.data['RES']
                                self.train_features.append(self.RES)
                                self.train_labels.append(['RES'])


                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "RES Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.RES)
                                self.train_labels.remove(['RES'])
                        except RuntimeError as err:
                            print(err)
                    # End of Residuals

                    # Non Residuals
                    # ZTD_igs Feature
                    if self.pred_ztd.isChecked():
                        try:
                            if 'ZTD_igs' in self.data.columns:
                                self.ZTD_igs = self.data['ZTD_igs']
                                self.train_features.append(self.ZTD_igs)
                                self.train_labels.append(['ZTD_igs'])

                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "ZTD_igs Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.ZTD_igs)
                                self.train_labels.remove(['ZTD_igs'])
                        except RuntimeError as err:
                            print(err)
                    if self.pred_zhd.isChecked():
                        try:
                            if 'ZHD_igs' in self.data.columns:
                                self.ZHD_igs = self.data['ZHD_igs']
                                self.train_features.append(self.ZHD_igs)
                                self.train_labels.append(['ZHD_igs'])
                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "ZHD_igs Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.ZHD_igs)
                                self.train_labels.remove(['ZHD_igs'])
                        except RuntimeError as err:
                            print(err)
                    if self.pred_zwd.isChecked():
                        try:
                            if 'ZWD_igs' in self.data.columns:
                                self.ZWD_igs = self.data['ZWD_igs']
                                self.train_features.append(self.ZWD_igs)
                                self.train_labels.append(['ZWD_igs'])
                            else:
                                QtWidgets.QMessageBox.question(self.centralwidget, "Message",
                                                               "ZWD_igs Column not available",
                                                               QtWidgets.QMessageBox.Ok)
                                self.train_features.remove(self.ZWD_igs)
                                self.train_labels.remove(['ZWD_igs'])
                        except RuntimeError as err:
                            print(err)

                    self.np_train_features = np.array(self.train_features)

                    self.df_data = []
                    self.col_labels = []

                    for i in range(len(self.np_train_features)):
                        columns = self.listConv(self.train_labels[i])
                        self.col_labels.append(columns)

                    self.np_train_features = self.np_train_features.transpose()
                    self.df = pd.DataFrame(data=self.np_train_features, columns=self.col_labels)
                    print("New Dataframe: ", self.df)
                    print(self.col_labels)
                    if self.pred_ztd_res.isChecked() or self.pred_zhd_res.isChecked() or self.pred_zwd_res.isChecked():
                        print
                        self.modelTraning(self.df, self.col_labels, "RES", "RES", epochs=int(self.tt_num_of_iterations.text()))

                    if self.pred_ztd.isChecked() or self.pred_zhd_.isChecked() or self.pred_zwd_res.isChecked():
                        print
                        self.modelTraning(self.df, self.col_labels, "ZTD_igs", "ZTD_igs",
                                          epochs=int(self.tt_num_of_iterations.text()))



    def modelTraning(self, df, plot_label, desc,target, epochs):
        # check activation functions
        # relu
        self.model = res_model.model(df)
        # for i in range(len(self.train_labels)):
        self.model.display_plot(plot_label)
        self.model.data_description(desc)
        if self.relu_act_fun.isChecked():
            self.relu = "relu"
            self.model.data_description(desc)
            self.model.build_model(self.relu).summary()
            self.history, self.eval, self.pred = self.model.train(act_fun=self.relu, epochs=epochs, pred_data=self.pred_dataset.text())
            self.display_history()
            self.display_pred_history()
        # softmax
        if self.softmax_act_fun.isChecked():
            self.softmax = "softmax"
            self.model.data_description(desc)
            self.model.build_model(self.softmax).summary()
            self.history, self.eval, self.pred = self.model.train(act_fun=self.softmax, epochs=epochs,
                                                                  pred_data=self.pred_dataset.text())
            self.display_history()
            self.display_pred_history()
        # sigmoid
        if self.sigmoid_act_fun.isChecked():
            self.sigmoid = "sigmoid"
            self.model.data_description(desc)
            self.model.build_model(self.sigmoid).summary()
            self.history, self.eval, self.pred = self.model.train(act_fun=self.sigmoid, epochs=epochs,
                                                                  pred_data=self.pred_dataset.text())
            self.display_history()
            self.display_pred_history()
        # linear
        if self.linear_act_fun.isChecked():
            self.linear = "linear"
            self.model.data_description(desc)
            self.model.build_model(self.linear).summary()
            self.history, self.eval, self.pred = self.model.train(act_fun=self.linear, epochs=epochs,
                                                                  pred_data=self.pred_dataset.text())
            self.display_history()
            self.display_pred_history()

    def display_history(self):
        self.hist_diag = QtWidgets.QDialog(self.centralwidget)
        self.hist_diag.resize(452, 516)
        self.hist_diag.setWindowTitle("Training Summary")
        self.gridLayout = QtWidgets.QGridLayout(self.hist_diag)
        self.txt_display_records = QtWidgets.QTableView(self.hist_diag)
        self.gridLayout.addWidget(self.txt_display_records, 0, 0, 1, 1)
        self.btn_save_model = QtWidgets.QPushButton(self.hist_diag)
        self.btn_save_model.setText("Save Model")
        self.gridLayout.addWidget(self.btn_save_model, 1, 0, 1, 2)
        model = pandas_view_model.pandasModel(self.history)
        self.txt_display_records.setModel(model)

        self.btn_save_model.clicked.connect(self.saveModel)

        self.hist_diag.exec_()

    def display_pred_history(self):
        self.pred_hist_diag = QtWidgets.QDialog(self.centralwidget)
        self.pred_hist_diag.resize(452, 516)
        self.pred_hist_diag.setWindowTitle("Prediction History")
        self.gridLayout = QtWidgets.QGridLayout(self.pred_hist_diag)
        self.table_display_pred = QtWidgets.QTableView(self.pred_hist_diag)
        self.gridLayout.addWidget(self.table_display_pred, 0, 0, 1, 1)
        self.pred_summary = QtWidgets.Qlabel(self.pred_hist_diag)
        self.gridLayout.addWidget(self.pred_summary, 1, 0, 1, 2)
        self.pred_summary.setText("Loss, mean_square_error, mean_absolute_error: ",self.eval)
        pred = np.asarray(self.pred)
        pred = pd.DataFrame({"Predictions: ", pred})
        model = pandas_view_model.pandasModel(pred)
        self.table_display_pred.setModel(model)

        self.pred_hist_diag.exec_()


       # res_model.model.predict(pred_data)

    def saveModel(self):
        savemodel, _ = QtWidgets.QFileDialog.getSaveFileName(self.hist_diag,
                                                      "QFileDialog.getSaveFileName()", "",
                                                      "(*.h5)")
        with open(savemodel, 'wb') as sm:
            res_model.model.build_model(activation_func='relu').save(savemodel)



    def write_pred(self):
        wr, _ = QtWidgets.QFileDialog.getSaveFileName(self.centralwidget,
                                                                              "QFileDialog.getSaveFileName()", "",
                                                                              "(*.txt)")
        with open(wr, 'wb') as p:
            p.write(str.encode(np.asarray(self.pred)))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
