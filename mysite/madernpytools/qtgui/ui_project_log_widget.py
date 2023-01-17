# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'project_log_widget.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from madernpytools.qtgui.traitlet_widgets import QTraitLineEdit


class Ui_ProjectLogWidget(object):
    def setupUi(self, ProjectLogWidget):
        if not ProjectLogWidget.objectName():
            ProjectLogWidget.setObjectName(u"ProjectLogWidget")
        ProjectLogWidget.resize(307, 132)
        self.horizontalLayout = QHBoxLayout(ProjectLogWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.groupBox_2 = QGroupBox(ProjectLogWidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.btn_start = QPushButton(self.groupBox_2)
        self.btn_start.setObjectName(u"btn_start")

        self.gridLayout_2.addWidget(self.btn_start, 0, 0, 1, 1)

        self.btn_stop = QPushButton(self.groupBox_2)
        self.btn_stop.setObjectName(u"btn_stop")

        self.gridLayout_2.addWidget(self.btn_stop, 1, 0, 1, 1)

        self.btn_save = QPushButton(self.groupBox_2)
        self.btn_save.setObjectName(u"btn_save")

        self.gridLayout_2.addWidget(self.btn_save, 2, 0, 1, 1)


        self.horizontalLayout.addWidget(self.groupBox_2)

        self.groupBox = QGroupBox(ProjectLogWidget)
        self.groupBox.setObjectName(u"groupBox")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.le_project = QTraitLineEdit(self.groupBox)
        self.le_project.setObjectName(u"le_project")

        self.gridLayout.addWidget(self.le_project, 0, 1, 1, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)

        self.le_maleid = QTraitLineEdit(self.groupBox)
        self.le_maleid.setObjectName(u"le_maleid")

        self.gridLayout.addWidget(self.le_maleid, 1, 1, 1, 1)

        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.le_femaleid = QTraitLineEdit(self.groupBox)
        self.le_femaleid.setObjectName(u"le_femaleid")

        self.gridLayout.addWidget(self.le_femaleid, 2, 1, 1, 1)


        self.horizontalLayout.addWidget(self.groupBox)


        self.retranslateUi(ProjectLogWidget)

        self.btn_start.setDefault(False)


        QMetaObject.connectSlotsByName(ProjectLogWidget)
    # setupUi

    def retranslateUi(self, ProjectLogWidget):
        ProjectLogWidget.setWindowTitle(QCoreApplication.translate("ProjectLogWidget", u"Form", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("ProjectLogWidget", u"Log Controls", None))
        self.btn_start.setText(QCoreApplication.translate("ProjectLogWidget", u"Start", None))
        self.btn_stop.setText(QCoreApplication.translate("ProjectLogWidget", u"Stop", None))
        self.btn_save.setText(QCoreApplication.translate("ProjectLogWidget", u"Save", None))
        self.groupBox.setTitle(QCoreApplication.translate("ProjectLogWidget", u"Log settings", None))
        self.le_project.setText("")
        self.label_2.setText(QCoreApplication.translate("ProjectLogWidget", u"Female ID", None))
        self.label_3.setText(QCoreApplication.translate("ProjectLogWidget", u"Male ID", None))
#if QT_CONFIG(tooltip)
        self.le_maleid.setToolTip(QCoreApplication.translate("ProjectLogWidget", u"Base filename, log.csv will generate files log000000.csv, log000001.csv, etc..", None))
#endif // QT_CONFIG(tooltip)
        self.le_maleid.setText("")
        self.label.setText(QCoreApplication.translate("ProjectLogWidget", u"Project", None))
    # retranslateUi

