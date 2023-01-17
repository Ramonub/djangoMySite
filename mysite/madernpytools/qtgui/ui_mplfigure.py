# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mplfigure.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MplFigure(object):
    def setupUi(self, MplFigure):
        if not MplFigure.objectName():
            MplFigure.setObjectName(u"MplFigure")
        MplFigure.resize(311, 218)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MplFigure.sizePolicy().hasHeightForWidth())
        MplFigure.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(MplFigure)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.mpl_wid = QWidget(MplFigure)
        self.mpl_wid.setObjectName(u"mpl_wid")
        sizePolicy.setHeightForWidth(self.mpl_wid.sizePolicy().hasHeightForWidth())
        self.mpl_wid.setSizePolicy(sizePolicy)
        self.mplvl = QVBoxLayout(self.mpl_wid)
        self.mplvl.setObjectName(u"mplvl")

        self.gridLayout.addWidget(self.mpl_wid, 0, 0, 1, 1)


        self.retranslateUi(MplFigure)

        QMetaObject.connectSlotsByName(MplFigure)
    # setupUi

    def retranslateUi(self, MplFigure):
        MplFigure.setWindowTitle(QCoreApplication.translate("MplFigure", u"Form", None))
    # retranslateUi

