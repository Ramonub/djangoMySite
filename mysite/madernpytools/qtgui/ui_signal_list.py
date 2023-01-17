# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'signal_list.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_SignalList(object):
    def setupUi(self, SignalList):
        if not SignalList.objectName():
            SignalList.setObjectName(u"SignalList")
        SignalList.resize(188, 89)
        self.verticalLayout_2 = QVBoxLayout(SignalList)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.signal_area = QScrollArea(SignalList)
        self.signal_area.setObjectName(u"signal_area")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.signal_area.sizePolicy().hasHeightForWidth())
        self.signal_area.setSizePolicy(sizePolicy)
        self.signal_area.setMinimumSize(QSize(170, 0))
        self.signal_area.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 186, 87))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalSpacer = QSpacerItem(20, 66, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.signal_area.setWidget(self.scrollAreaWidgetContents)

        self.verticalLayout_2.addWidget(self.signal_area)


        self.retranslateUi(SignalList)

        QMetaObject.connectSlotsByName(SignalList)
    # setupUi

    def retranslateUi(self, SignalList):
        SignalList.setWindowTitle(QCoreApplication.translate("SignalList", u"Form", None))
    # retranslateUi

