from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets
import plotly as plt
import easygui
import numpy as np
import pandas as pd
import timeseries_prepearing
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import configparser
import statsmodels.api as sm
import subprocess
from io import StringIO
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error



class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1099, 757)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget = gl.GLViewWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(0, 500))
        self.widget.setObjectName("widget")
        self.verticalLayout_9.addWidget(self.widget)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setObjectName("label")
        self.verticalLayout_6.addWidget(self.label)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(0, -1, 100, -1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.loadFileButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadFileButton.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadFileButton.sizePolicy().hasHeightForWidth())
        self.loadFileButton.setSizePolicy(sizePolicy)
        self.loadFileButton.setObjectName("loadFileButton")
        self.horizontalLayout_6.addWidget(self.loadFileButton)
        self.ShowTraceButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShowTraceButton.sizePolicy().hasHeightForWidth())
        self.ShowTraceButton.setSizePolicy(sizePolicy)
        self.ShowTraceButton.setObjectName("ShowTraceButton")
        self.horizontalLayout_6.addWidget(self.ShowTraceButton)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_6)
        spacerItem3 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem3)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_6.addWidget(self.label_2)
        spacerItem4 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_6.addItem(spacerItem4)
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget.sizePolicy().hasHeightForWidth())
        self.listWidget.setSizePolicy(sizePolicy)
        self.listWidget.setMaximumSize(QtCore.QSize(16777215, 100))
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_6.addWidget(self.listWidget)
        spacerItem5 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_6.addItem(spacerItem5)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_7.addWidget(self.label_3)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_7.addWidget(self.label_11)
        self.verticalLayout_8.addLayout(self.horizontalLayout_7)
        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_8.addItem(spacerItem6)
        self.verticalLayout_6.addLayout(self.verticalLayout_8)
        self.horizontalLayout_8.addLayout(self.verticalLayout_6)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_8.addLayout(self.horizontalLayout_4)
        spacerItem7 = QtWidgets.QSpacerItem(200, 20, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem7)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_2.sizePolicy().hasHeightForWidth())
        self.lineEdit_2.setSizePolicy(sizePolicy)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_5.addWidget(self.lineEdit_2)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.label_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_5.addWidget(self.lineEdit)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3.addLayout(self.verticalLayout_5)
        spacerItem8 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_3.addItem(spacerItem8)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_3.setContentsMargins(0, -1, 0, -1)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem9)
        self.predictButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictButton.sizePolicy().hasHeightForWidth())
        self.predictButton.setSizePolicy(sizePolicy)
        self.predictButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.predictButton.setObjectName("predictButton")
        self.horizontalLayout_3.addWidget(self.predictButton)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem10)
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveButton.sizePolicy().hasHeightForWidth())
        self.saveButton.setSizePolicy(sizePolicy)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout_3.addWidget(self.saveButton, 0, QtCore.Qt.AlignRight)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem11)
        self.saveMetricsButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveMetricsButton.setObjectName("saveMetricsButton")
        self.horizontalLayout_3.addWidget(self.saveMetricsButton)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mad_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mad_label.sizePolicy().hasHeightForWidth())
        self.mad_label.setSizePolicy(sizePolicy)
        self.mad_label.setObjectName("mad_label")
        self.horizontalLayout.addWidget(self.mad_label)
        self.mad_value = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.mad_value.sizePolicy().hasHeightForWidth())
        self.mad_value.setSizePolicy(sizePolicy)
        self.mad_value.setObjectName("mad_value")
        self.horizontalLayout.addWidget(self.mad_value)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem12)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.mse_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mse_label.sizePolicy().hasHeightForWidth())
        self.mse_label.setSizePolicy(sizePolicy)
        self.mse_label.setObjectName("mse_label")
        self.horizontalLayout_2.addWidget(self.mse_label)
        self.mse_value = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.mse_value.sizePolicy().hasHeightForWidth())
        self.mse_value.setSizePolicy(sizePolicy)
        self.mse_value.setMinimumSize(QtCore.QSize(0, 0))
        self.mse_value.setObjectName("mse_value")
        self.horizontalLayout_2.addWidget(self.mse_value)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem13)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.mape_label = QtWidgets.QLabel(self.centralwidget)
        self.mape_label.setObjectName("mape_label")
        self.horizontalLayout_9.addWidget(self.mape_label)
        self.mape_value = QtWidgets.QLabel(self.centralwidget)
        self.mape_value.setObjectName("mape_value")
        self.horizontalLayout_9.addWidget(self.mape_value)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem14)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.mpe_label = QtWidgets.QLabel(self.centralwidget)
        self.mpe_label.setObjectName("mpe_label")
        self.horizontalLayout_10.addWidget(self.mpe_label)
        self.mpe_value = QtWidgets.QLabel(self.centralwidget)
        self.mpe_value.setObjectName("mpe_value")
        self.horizontalLayout_10.addWidget(self.mpe_value)
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem15)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.df_label = QtWidgets.QLabel(self.centralwidget)
        self.df_label.setObjectName("df_label")
        self.horizontalLayout_11.addWidget(self.df_label)
        self.df_value = QtWidgets.QLabel(self.centralwidget)
        self.df_value.setObjectName("df_value")
        self.horizontalLayout_11.addWidget(self.df_value)
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem16)
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        spacerItem17 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout_2.addItem(spacerItem17)
        self.horizontalLayout_8.addLayout(self.verticalLayout_2)
        self.verticalLayout_9.addLayout(self.horizontalLayout_8)
        self.verticalLayout_10.addLayout(self.verticalLayout_9)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1099, 21))
        self.menubar.setObjectName("menubar")
        
        config.read("models.ini")
        self.NNmodels = {}
        self.ModelsPaths = {}
        for key in config['MODELS']:
            value = config.get('MODELS', key)
            self.NNmodels[key] = value
            self.ModelsPaths[value] = config.get(value, "path", fallback="")
            it = QtWidgets.QListWidgetItem(self.listWidget)
            self.listWidget.setItemWidget(it, QtWidgets.QRadioButton(value))
        
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Predictor"))
        self.label.setText(_translate("MainWindow", "Отображение:"))

        self.loadFileButton.setText(_translate("MainWindow", "Файл"))
        self.ShowTraceButton.setText(_translate("MainWindow", "Отобразить"))
        
        self.saveMetricsButton.setText(_translate("MainWindow", "Сохранить метрики"))

        self.ShowTraceButton.setEnabled(False)
        self.saveMetricsButton.setEnabled(False)
        self.label_2.setText(_translate("MainWindow", "<Имя файла>"))

        self.label_3.setText(_translate("MainWindow", "Количество траекторных точек: "))

        self.label_11.setText(_translate("MainWindow", "0"))

        self.label_4.setText(_translate("MainWindow", "Прогнозирование: "))
        self.label_6.setText(_translate("MainWindow", "Точек: "))

        self.label_5.setText(_translate("MainWindow", "От точки: "))

        self.predictButton.setText(_translate("MainWindow", "Спрогнозировать"))

        self.predictButton.setEnabled(False)
        self.saveButton.setText(_translate("MainWindow", "Сохранить прогноз"))

        self.saveButton.setEnabled(False)

        self.mad_label.setText(_translate("MainWindow", "MAD: "))
        self.mse_label.setText(_translate("MainWindow", "MSE: "))
        self.mape_label.setText(_translate("MainWindow", "MAPE: "))
        self.mpe_label.setText(_translate("MainWindow", "MPE: "))
        self.df_label.setText(_translate("MainWindow", "DF: "))

        self.mad_value.setText(_translate("MainWindow", "0"))
        self.mse_value.setText(_translate("MainWindow", "0"))
        self.mape_value.setText(_translate("MainWindow", "0"))
        self.mpe_value.setText(_translate("MainWindow", "0"))
        self.df_value.setText(_translate("MainWindow", "0"))

        self.lineEdit.setEnabled(False)

        self.lineEdit_2.setEnabled(False)
        self.add_functions()

    def add_functions(self):
        self.loadFileButton.clicked.connect(lambda: self.loadFileButtonPressed())
        self.ShowTraceButton.clicked.connect(lambda: self.ShowTraceButtonPressed(self.succesfull_loading_file))
        self.predictButton.clicked.connect(lambda: self.predictButtonPressed())
        self.saveButton.clicked.connect(lambda: self.saveButtonPressed())

    def show_information_about_trace(self, successful_flag, number_of_points):
        print(f"show_information_about_trace is called; number_of_points={number_of_points}")
        self.number_of_points_in_dataet = number_of_points
        if(successful_flag):
            self.label_3.show()
            self.label_11.show()
            self.label_11.setText( str(number_of_points))
        else:
            self.label_11.setText(_translate("MainWindow", str(number_of_points)))
    
    def show_trace_function(self, successful_flag, dataset):
        if(successful_flag):
            self.widget.clear()
            self.show_graph(dataset)
        else:
            self.widget.clear()

    def show_information_about_file(self, successful_flag, name_of_file):
        if(successful_flag):
            self.ShowTraceButton.show()
            self.ShowTraceButton.setEnabled(True)
            self.label_2.show()
            self.label_2.setText(name_of_file)
        else:
            self.ShowTraceButton.setEnabled(False)
            self.label_2.show()
            self.label_2.setText("Неправильный путь")

    def show_prediction_manipulators(self, successful_flag):
        if(successful_flag):
            self.predictButton.show()
            self.predictButton.setEnabled(True)
            
            self.saveButton.show()
            self.saveButton.setEnabled(True)
            
            self.saveMetricsButton.show()
            self.saveMetricsButton.setEnabled(True)
            
            self.lineEdit.show()
            self.lineEdit.setEnabled(True)
            
            self.lineEdit_2.show()
            self.lineEdit_2.setEnabled(True)
            
            
        else:
            self.predictButton.setEnabled(False)
            self.saveButton.setEnabled(False)
            
            self.lineEdit.setEnabled(False)
            self.lineEdit_2.setEnabled(False)

    def show_prediction_information(self, successful_flag, predicted_data, SSE):
        if(successful_flag):
            self.mad_label.show()
            self.mad_value.show()
            
            self.mse_label.show()
            self.mse_value.show()
            
            self.mad_label.show()
            self.mad_value.show()
            
            self.mape_label.show()
            self.mape_value.show()
            
            self.mpe_label.show()
            self.mpe_value.show()
            
            self.df_label.show()
            self.df_value.show()
            
            self.mad_label.setText(str('MAD'))
            self.mad_value.setText(str(SSE['MAD']))
            
            self.mse_label.setText(str('MSE'))
            self.mse_value.setText(str(SSE['MSE']))
            
            self.mad_label.setText(str('MAD'))
            self.mad_value.setText(str(SSE['MAD']))
            
            self.mape_label.setText(str('MAPE'))
            self.mape_value.setText(str(SSE['MAPE']))
            
            self.mpe_label.setText(str('MPE'))
            self.mpe_value.setText(str(SSE['MPE']))
            
            self.df_label.setText(str('СКР'))
            self.df_value.setText(str(SSE['Стандартная ошибка']))
            
        else:

            self.mad_value.setText(str(0))
            
            self.mse_value.setText(str(0))
            

            self.mad_value.setText(str(0))
            
            self.mape_value.setText(str(0))
            
            self.mpe_value.setText(str(0))
            
            self.df_value.setText(str(0))
    
    
    def loading_function(self):
        path_to_file = ""
        name_of_file = ""
        number_of_points = 0
        self.dataset = []
        input_file = easygui.fileopenbox(default="./saved_data/*.xlsx")
        self.path_to_dataset = input_file
        self.dataset = []
        self.grid = gl.GLGridItem()
        self.axiz = gl.GLAxisItem()
        self.number_of_points_in_dataet = 0

        self.dataset = pd.read_excel(input_file)
        self.path_to_model = ""
        number_of_points = len(self.dataset)
        name_of_file = input_file

        print(self.dataset.columns)
        return (True, path_to_file, name_of_file, number_of_points)

    def metrics_short(self, real, forecast, coumn_name, start, end):
        real_arr = np.array((real.loc[start:end])[coumn_name])
        forecast_arr = np.array((forecast.loc[start:end])[coumn_name])

        dict2 = {
            'MAD': round(abs(real_arr-forecast_arr).mean(),4),
            'MSE': round(((real_arr-forecast_arr)**2).mean(),4),
            'MAPE': round((abs(real_arr-forecast_arr)/real_arr).mean(),4),
            'MPE': round(((real_arr-forecast_arr)/real_arr).mean(),4),
            'Стандартная ошибка': round(((real_arr-forecast_arr)**2).mean()**0.5,4)
        }
        return dict2

    def prediction_function(self, dataset, path, from_point=0, number_of_points=0):
        print(path)
        end_point = from_point + number_of_points if (from_point + number_of_points)<self.number_of_points_in_dataet else self.number_of_points_in_dataet
        pred_long = self.number_of_points_in_dataet - from_point
        
        subprocess.run(['python', str(path), self.path_to_dataset, str(from_point), str(from_point + number_of_points)])
        print('here7')
        prediction = pd.read_excel('./models/temp_dataset.xlsx')
        prediction = prediction.set_index('Unnamed: 0')
        result_analisisX = self.metrics_short(dataset, prediction, 'X', from_point, end_point)
        result_analisisY = self.metrics_short(dataset, prediction, 'Y', from_point, end_point)
        result_analisisZ = self.metrics_short(dataset, prediction, 'Z', from_point, end_point)

        result_analisisX['MAD'] = np.mean([result_analisisX['MAD'], result_analisisY['MAD'], result_analisisZ['MAD']])
        result_analisisX['MSE'] = np.mean([result_analisisX['MSE'], result_analisisY['MSE'], result_analisisZ['MSE']])
        result_analisisX['MAPE'] = np.mean([result_analisisX['MAPE'], result_analisisY['MAPE'], result_analisisZ['MAPE']])
        result_analisisX['MPE'] = np.mean([result_analisisX['MPE'], result_analisisY['MPE'], result_analisisZ['MPE']])
        result_analisisX['Стандартная ошибка'] = np.mean([result_analisisX['Стандартная ошибка'], result_analisisY['Стандартная ошибка'], result_analisisZ['Стандартная ошибка']])

        return (True, prediction, result_analisisX)
        
    def loadFileButtonPressed(self):
        number_of_points=0
        name_of_file = ""
        result = self.loading_function()
        
        number_of_points = result[3]
        name_of_file = result[2]
        self.path_to_file = result[1]
        succesful_flag = result[0]
        
        if(succesful_flag):
            self.show_information_about_file(succesful_flag, name_of_file)
        else:
            self.show_information_about_file(succesful_flag, name_of_file)
            
        self.succesfull_loading_file = (succesful_flag, number_of_points)
        
    def ShowTraceButtonPressed(self, succesfull_loading_file):
        successfil_flag = self.succesfull_loading_file[0]
        number_of_points = self.succesfull_loading_file[1]
        if(successfil_flag):
            self.show_trace_function(True, self.dataset)
            self.show_prediction_manipulators(True)
            self.show_information_about_trace(True, number_of_points)

        else:
            self.show_trace_function(False, 0)
            self.show_prediction_manipulators(False)
            self.show_information_about_trace(False, 0)

        
    
    def predictButtonPressed(self):
    
        if((self.listWidget.currentItem())!=None):
            print(self.listWidget.currentRow())
            print(self.ModelsPaths[list(self.ModelsPaths.keys())[self.listWidget.currentRow()]])
            self.path_to_model=self.ModelsPaths[list(self.ModelsPaths.keys())[self.listWidget.currentRow()]]
        
        dataframe_indexes = np.array(self.dataset.index)
        from_point = dataframe_indexes[-1]
        print('predictButtonPressed [from_point]: ', from_point)
        number_of_points = 1
        print('predictButtonPressed [number_of_points]: ', number_of_points)
        
        if(self.lineEdit_2.text().isdigit()):
            from_point = int(self.lineEdit_2.text())
        if(self.lineEdit.text().isdigit()):
            number_of_points = int(self.lineEdit.text())
        result = self.prediction_function(self.dataset, self.path_to_model, from_point, number_of_points)
        successful_of_prediction = result[0]
        predicted_data = result[1]
        SSE = result[2]
        
        print('see here2')
        if(successful_of_prediction):
            self.show_prediction_information(True, predicted_data, SSE)
            self.saveButton.show()
            self.saveButton.setEnabled(True)
            self.predicted_data_for_saving = predicted_data
            self.show_graph(self.dataset, 1)
            self.show_add_trace_graph(predicted_data, 1)
            print('see here2')
            if((self.listWidget.currentItem())!=None):
                print(self.listWidget.currentRow())
                print(self.ModelsPaths)
                self.label.setText(self.ModelsPaths[list(self.ModelsPaths.keys())[self.listWidget.currentRow()]])

        else:
            self.show_prediction_information(False, 0, 0)
            self.saveButton.setEnabled(False)
        
    
    def saveButtonPressed(self):
        pass
        
    def show_graph(self, result_df, count_of_traces = 1, graph_color = (0, 0, 1, 1)):
        self.widget.clear()
        
        mashtab = 1./1000
        x_axis = result_df["X"].max()*mashtab*2
        y_axis = (result_df["Y"].max() - result_df["Y"].min())*mashtab*2
        z_axis = result_df["Z"].max()*mashtab*2
        
        self.grid.setSize(x = x_axis, y = y_axis, z = z_axis)
        self.axiz.setSize(x = max([x_axis, y_axis, z_axis]), y = max([x_axis, y_axis, z_axis]), z = max([x_axis, y_axis, z_axis]))

        self.widget.addItem(self.grid)
        self.widget.addItem(self.axiz)

        positions = (np.array([np.array(result_df["X"])*mashtab, np.array(result_df["Y"])*mashtab,  np.array(result_df["Z"])*mashtab])).transpose() 
        self.widget.addItem(gl.GLGraphItem(pos = positions, nodeColor = graph_color ))
        self.widget.addItem(gl.GLLinePlotItem(pos = positions, color = graph_color))

        self.widget.show()
        
    def show_add_trace_graph(self, new_trace_df, count_of_traces = 1, graph_color = (1, 0, 0, 1)):

        mashtab = 1./1000
        max_value =  new_trace_df["X"].max() if (new_trace_df["X"].max() > self.dataset["X"].max()) else self.dataset["X"].max()
        x_axis = max_value*mashtab*2
        max_value = new_trace_df["Y"].max() if (new_trace_df["Y"].max() > self.dataset["Y"].max()) else self.dataset["Y"].max()
        min_value = new_trace_df["Y"].min() if (new_trace_df["Y"].min() < self.dataset["Y"].min()) else self.dataset["Y"].min()
        y_axis = (max_value - min_value)*mashtab*2
        max_value = new_trace_df["Z"].max() if (new_trace_df["Z"].max() > self.dataset["Z"].max()) else self.dataset["Z"].max()
        z_axis = max_value*mashtab*2

        self.widget.removeItem(self.grid)
        self.widget.removeItem(self.axiz)
        
        self.grid.setSize(x = x_axis, y = y_axis, z = z_axis)
        self.axiz.setSize(x = max([x_axis, y_axis, z_axis]), y = max([x_axis, y_axis, z_axis]), z = max([x_axis, y_axis, z_axis]))

        self.widget.addItem(self.grid)
        self.widget.addItem(self.axiz)

        positions = (np.array([np.array(new_trace_df["X"])*mashtab, np.array(new_trace_df["Y"])*mashtab,  np.array(new_trace_df["Z"])*mashtab])).transpose() 
        self.widget.addItem(gl.GLGraphItem(pos = positions, nodeColor = graph_color))
        self.widget.addItem(gl.GLLinePlotItem(pos = positions, color = graph_color))
        self.widget.addItem(gl.GLScatterPlotItem(pos=positions[0], color=(1, 0, 1, 1), size = 4))

        self.widget.show()
        
            


if __name__ == "__main__":
    import sys
    
    config = configparser.ConfigParser()
    config.read("settings.ini")
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
