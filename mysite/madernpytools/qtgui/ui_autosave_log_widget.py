# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'autosave_log_widget.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from madernpytools.qtgui.traitlet_widgets import QTraitLineEdit


class Ui_AutosaveLogWidget(object):
    def setupUi(self, AutosaveLogWidget):
        if not AutosaveLogWidget.objectName():
            AutosaveLogWidget.setObjectName(u"AutosaveLogWidget")
        AutosaveLogWidget.resize(479, 126)
        self.horizontalLayout = QHBoxLayout(AutosaveLogWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox_2 = QGroupBox(AutosaveLogWidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.btn_start = QPushButton(self.groupBox_2)
        self.btn_start.setObjectName(u"btn_start")

        self.gridLayout_2.addWidget(self.btn_start, 0, 0, 1, 1)

        self.btn_stop = QPushButton(self.groupBox_2)
        self.btn_stop.setObjectName(u"btn_stop")

        self.gridLayout_2.addWidget(self.btn_stop, 1, 0, 1, 1)


        self.horizontalLayout.addWidget(self.groupBox_2)

        self.groupBox = QGroupBox(AutosaveLogWidget)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.le_logdir = QTraitLineEdit(self.groupBox)
        self.le_logdir.setObjectName(u"le_logdir")

        self.gridLayout.addWidget(self.le_logdir, 0, 1, 1, 1)

        self.btn_select_dir = QPushButton(self.groupBox)
        self.btn_select_dir.setObjectName(u"btn_select_dir")
        self.btn_select_dir.setMinimumSize(QSize(10, 0))
        self.btn_select_dir.setMaximumSize(QSize(25, 16777215))

        self.gridLayout.addWidget(self.btn_select_dir, 0, 2, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.le_description = QTraitLineEdit(self.groupBox)
        self.le_description.setObjectName(u"le_description")

        self.gridLayout.addWidget(self.le_description, 2, 1, 1, 1)

        self.le_basefilename = QTraitLineEdit(self.groupBox)
        self.le_basefilename.setObjectName(u"le_basefilename")

        self.gridLayout.addWidget(self.le_basefilename, 1, 1, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)


        self.horizontalLayout.addWidget(self.groupBox)


        self.retranslateUi(AutosaveLogWidget)

        self.btn_start.setDefault(False)


        QMetaObject.connectSlotsByName(AutosaveLogWidget)
    # setupUi

    def retranslateUi(self, AutosaveLogWidget):
        AutosaveLogWidget.setWindowTitle(QCoreApplication.translate("AutosaveLogWidget", u"Form", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("AutosaveLogWidget", u"Log Controls", None))
        self.btn_start.setText(QCoreApplication.translate("AutosaveLogWidget", u"Start", None))
        self.btn_stop.setText(QCoreApplication.translate("AutosaveLogWidget", u"Stop", None))
        self.groupBox.setTitle(QCoreApplication.translate("AutosaveLogWidget", u"Log settings", None))
        self.label_2.setText(QCoreApplication.translate("AutosaveLogWidget", u"Description", None))
        self.le_logdir.setText(QCoreApplication.translate("AutosaveLogWidget", u"~/", None))
        self.btn_select_dir.setText(QCoreApplication.translate("AutosaveLogWidget", u"...", None))
        self.label.setText(QCoreApplication.translate("AutosaveLogWidget", u"Log directory", None))
#if QT_CONFIG(tooltip)
        self.le_basefilename.setToolTip(QCoreApplication.translate("AutosaveLogWidget", u"Base filename, log.csv will generate files log000000.csv, log000001.csv, etc..", None))
#endif // QT_CONFIG(tooltip)
        self.le_basefilename.setText(QCoreApplication.translate("AutosaveLogWidget", u"log.csv", None))
        self.label_3.setText(QCoreApplication.translate("AutosaveLogWidget", u"Base filename", None))
    # retranslateUi

