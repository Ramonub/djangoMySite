# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'signal_viewer.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from madernpytools.qtgui import MplFigure
from madernpytools.qtgui.signal_selection import SignalSelection


class Ui_SignalViewer(object):
    def setupUi(self, SignalViewer):
        if not SignalViewer.objectName():
            SignalViewer.setObjectName(u"SignalViewer")
        SignalViewer.resize(580, 406)
        self.gridLayout = QGridLayout(SignalViewer)
        self.gridLayout.setObjectName(u"gridLayout")
        self.signal_selection = SignalSelection(SignalViewer)
        self.signal_selection.setObjectName(u"signal_selection")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.signal_selection.sizePolicy().hasHeightForWidth())
        self.signal_selection.setSizePolicy(sizePolicy)
        self.signal_selection.setMinimumSize(QSize(0, 100))

        self.gridLayout.addWidget(self.signal_selection, 0, 0, 1, 1)

        self.signal_display = MplFigure(SignalViewer)
        self.signal_display.setObjectName(u"signal_display")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.signal_display.sizePolicy().hasHeightForWidth())
        self.signal_display.setSizePolicy(sizePolicy1)
        self.signal_display.setMinimumSize(QSize(100, 100))

        self.gridLayout.addWidget(self.signal_display, 0, 1, 2, 1)

        self.verticalSpacer = QSpacerItem(20, 50, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout.addItem(self.verticalSpacer, 1, 0, 1, 1)


        self.retranslateUi(SignalViewer)

        QMetaObject.connectSlotsByName(SignalViewer)
    # setupUi

    def retranslateUi(self, SignalViewer):
        SignalViewer.setWindowTitle(QCoreApplication.translate("SignalViewer", u"Form", None))
    # retranslateUi

