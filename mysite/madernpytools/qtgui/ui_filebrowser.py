# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'filebrowser.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from madernpytools.qtgui.traitlet_widgets import QTraitLineEdit


class Ui_FileBrowser(object):
    def setupUi(self, FileBrowser):
        if not FileBrowser.objectName():
            FileBrowser.setObjectName(u"FileBrowser")
        FileBrowser.resize(307, 23)
        self.horizontalLayout = QHBoxLayout(FileBrowser)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.le_path = QTraitLineEdit(FileBrowser)
        self.le_path.setObjectName(u"le_path")
        self.le_path.setInputMethodHints(Qt.ImhNone)

        self.horizontalLayout.addWidget(self.le_path)

        self.pb_browse = QPushButton(FileBrowser)
        self.pb_browse.setObjectName(u"pb_browse")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pb_browse.sizePolicy().hasHeightForWidth())
        self.pb_browse.setSizePolicy(sizePolicy)
        self.pb_browse.setMinimumSize(QSize(21, 0))
        self.pb_browse.setMaximumSize(QSize(21, 16777215))

        self.horizontalLayout.addWidget(self.pb_browse)


        self.retranslateUi(FileBrowser)

        QMetaObject.connectSlotsByName(FileBrowser)
    # setupUi

    def retranslateUi(self, FileBrowser):
        FileBrowser.setWindowTitle(QCoreApplication.translate("FileBrowser", u"Form", None))
        self.pb_browse.setText(QCoreApplication.translate("FileBrowser", u"...", None))
    # retranslateUi

