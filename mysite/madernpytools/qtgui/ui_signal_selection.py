# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'signal_selection.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from madernpytools.qtgui.signal_list import SignalList


class Ui_SignalSelection(object):
    def setupUi(self, SignalSelection):
        if not SignalSelection.objectName():
            SignalSelection.setObjectName(u"SignalSelection")
        SignalSelection.resize(129, 176)
        self.horizontalLayout = QHBoxLayout(SignalSelection)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox = QGroupBox(SignalSelection)
        self.groupBox.setObjectName(u"groupBox")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setFlat(True)
        self.groupBox.setCheckable(True)
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(9, 9, 9, 9)
        self.signal_list = SignalList(self.groupBox)
        self.signal_list.setObjectName(u"signal_list")

        self.horizontalLayout_2.addWidget(self.signal_list)


        self.horizontalLayout.addWidget(self.groupBox)


        self.retranslateUi(SignalSelection)

        QMetaObject.connectSlotsByName(SignalSelection)
    # setupUi

    def retranslateUi(self, SignalSelection):
        SignalSelection.setWindowTitle(QCoreApplication.translate("SignalSelection", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("SignalSelection", u"Signal Selection", None))
    # retranslateUi

