# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'tool_view_widget.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from  . import gui_sources_rc

class Ui_ToolViewWidget(object):
    def setupUi(self, ToolViewWidget):
        if not ToolViewWidget.objectName():
            ToolViewWidget.setObjectName(u"ToolViewWidget")
        ToolViewWidget.resize(520, 290)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ToolViewWidget.sizePolicy().hasHeightForWidth())
        ToolViewWidget.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(ToolViewWidget)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.image_widget = QWidget(ToolViewWidget)
        self.image_widget.setObjectName(u"image_widget")
        sizePolicy.setHeightForWidth(self.image_widget.sizePolicy().hasHeightForWidth())
        self.image_widget.setSizePolicy(sizePolicy)
        self.image_widget.setMinimumSize(QSize(520, 290))
        self.image_widget.setStyleSheet(u"image: url(:/newPrefix/ema_schematic_tooling.png);")

        self.gridLayout.addWidget(self.image_widget, 0, 0, 1, 1)


        self.retranslateUi(ToolViewWidget)

        QMetaObject.connectSlotsByName(ToolViewWidget)
    # setupUi

    def retranslateUi(self, ToolViewWidget):
        ToolViewWidget.setWindowTitle(QCoreApplication.translate("ToolViewWidget", u"Form", None))
    # retranslateUi

