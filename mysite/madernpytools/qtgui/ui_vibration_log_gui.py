# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'vibration_log_gui.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from madernpytools.qtgui.signal_viewer import SignalViewer
from madernpytools.qtgui.project_log_widget import ProjectLogWidget


class Ui_VibrationLogGui(object):
    def setupUi(self, VibrationLogGui):
        if not VibrationLogGui.objectName():
            VibrationLogGui.setObjectName(u"VibrationLogGui")
        VibrationLogGui.resize(768, 419)
        self.actionConnect = QAction(VibrationLogGui)
        self.actionConnect.setObjectName(u"actionConnect")
        self.actionDisconnect = QAction(VibrationLogGui)
        self.actionDisconnect.setObjectName(u"actionDisconnect")
        self.centralwidget = QWidget(VibrationLogGui)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.horizontalSpacer = QSpacerItem(340, 20, QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.tab_time = QTabWidget(self.centralwidget)
        self.tab_time.setObjectName(u"tab_time")
        font = QFont()
        font.setBold(False)
        font.setWeight(50)
        self.tab_time.setFont(font)
        self.tab_timedomain = QWidget()
        self.tab_timedomain.setObjectName(u"tab_timedomain")
        self.horizontalLayout = QHBoxLayout(self.tab_timedomain)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.time_display = SignalViewer(self.tab_timedomain)
        self.time_display.setObjectName(u"time_display")

        self.horizontalLayout.addWidget(self.time_display)

        self.tab_time.addTab(self.tab_timedomain, "")
        self.tab_frequencydomain = QWidget()
        self.tab_frequencydomain.setObjectName(u"tab_frequencydomain")
        self.horizontalLayout_2 = QHBoxLayout(self.tab_frequencydomain)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.frequency_display = SignalViewer(self.tab_frequencydomain)
        self.frequency_display.setObjectName(u"frequency_display")

        self.horizontalLayout_2.addWidget(self.frequency_display)

        self.tab_time.addTab(self.tab_frequencydomain, "")

        self.gridLayout_2.addWidget(self.tab_time, 2, 0, 1, 2)

        self.log_control = ProjectLogWidget(self.centralwidget)
        self.log_control.setObjectName(u"log_control")
        self.log_control.setMinimumSize(QSize(500, 0))
        self.log_control.setMaximumSize(QSize(1000, 16777215))

        self.gridLayout_2.addWidget(self.log_control, 0, 0, 1, 1)

        VibrationLogGui.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(VibrationLogGui)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 768, 21))
        self.menuDevice = QMenu(self.menubar)
        self.menuDevice.setObjectName(u"menuDevice")
        VibrationLogGui.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(VibrationLogGui)
        self.statusbar.setObjectName(u"statusbar")
        VibrationLogGui.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuDevice.menuAction())
        self.menuDevice.addAction(self.actionConnect)
        self.menuDevice.addAction(self.actionDisconnect)

        self.retranslateUi(VibrationLogGui)

        self.tab_time.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(VibrationLogGui)
    # setupUi

    def retranslateUi(self, VibrationLogGui):
        VibrationLogGui.setWindowTitle(QCoreApplication.translate("VibrationLogGui", u"NI-Measurement GUI", None))
        self.actionConnect.setText(QCoreApplication.translate("VibrationLogGui", u"Connect", None))
        self.actionDisconnect.setText(QCoreApplication.translate("VibrationLogGui", u"Disconnect", None))
        self.tab_time.setTabText(self.tab_time.indexOf(self.tab_timedomain), QCoreApplication.translate("VibrationLogGui", u"Time-View", None))
        self.tab_time.setTabText(self.tab_time.indexOf(self.tab_frequencydomain), QCoreApplication.translate("VibrationLogGui", u"Frequency-View", None))
        self.menuDevice.setTitle(QCoreApplication.translate("VibrationLogGui", u"Device", None))
    # retranslateUi

