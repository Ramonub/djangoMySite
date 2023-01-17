# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'lowpass_filter_settings.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from madernpytools.qtgui.traitlet_widgets import QTraitDoubleSpinBox
from madernpytools.qtgui.traitlet_widgets import QTraitSpinBox
from madernpytools.qtgui.traitlet_widgets import QTraitCheckBox


class Ui_LowPassFilterSettings(object):
    def setupUi(self, LowPassFilterSettings):
        if not LowPassFilterSettings.objectName():
            LowPassFilterSettings.setObjectName(u"LowPassFilterSettings")
        LowPassFilterSettings.resize(235, 136)
        self.horizontalLayout = QHBoxLayout(LowPassFilterSettings)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.gb_lp_settings = QGroupBox(LowPassFilterSettings)
        self.gb_lp_settings.setObjectName(u"gb_lp_settings")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gb_lp_settings.sizePolicy().hasHeightForWidth())
        self.gb_lp_settings.setSizePolicy(sizePolicy)
        self.gb_lp_settings.setFlat(True)
        self.gb_lp_settings.setCheckable(True)
        self.gridLayout = QGridLayout(self.gb_lp_settings)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_2 = QLabel(self.gb_lp_settings)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)

        self.sp_sampling_frequency = QTraitDoubleSpinBox(self.gb_lp_settings)
        self.sp_sampling_frequency.setObjectName(u"sp_sampling_frequency")
        self.sp_sampling_frequency.setMinimum(1.000000000000000)
        self.sp_sampling_frequency.setMaximum(50000.000000000000000)
        self.sp_sampling_frequency.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.sp_sampling_frequency, 0, 1, 1, 1)

        self.cb_overrule = QTraitCheckBox(self.gb_lp_settings)
        self.cb_overrule.setObjectName(u"cb_overrule")

        self.gridLayout.addWidget(self.cb_overrule, 0, 2, 1, 1)

        self.label = QLabel(self.gb_lp_settings)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)

        self.sp_lowpass_frequency = QTraitDoubleSpinBox(self.gb_lp_settings)
        self.sp_lowpass_frequency.setObjectName(u"sp_lowpass_frequency")
        self.sp_lowpass_frequency.setMinimum(0.000000000000000)
        self.sp_lowpass_frequency.setMaximum(100000.000000000000000)
        self.sp_lowpass_frequency.setSingleStep(0.010000000000000)

        self.gridLayout.addWidget(self.sp_lowpass_frequency, 1, 1, 1, 1)

        self.label_3 = QLabel(self.gb_lp_settings)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.sp_order = QTraitSpinBox(self.gb_lp_settings)
        self.sp_order.setObjectName(u"sp_order")
        self.sp_order.setMinimum(1)

        self.gridLayout.addWidget(self.sp_order, 2, 1, 1, 1)


        self.horizontalLayout.addWidget(self.gb_lp_settings)


        self.retranslateUi(LowPassFilterSettings)

        QMetaObject.connectSlotsByName(LowPassFilterSettings)
    # setupUi

    def retranslateUi(self, LowPassFilterSettings):
        LowPassFilterSettings.setWindowTitle(QCoreApplication.translate("LowPassFilterSettings", u"Form", None))
        self.gb_lp_settings.setTitle(QCoreApplication.translate("LowPassFilterSettings", u"Active", None))
        self.label_2.setText(QCoreApplication.translate("LowPassFilterSettings", u"Sampling frequency", None))
        self.cb_overrule.setText("")
        self.label.setText(QCoreApplication.translate("LowPassFilterSettings", u"Low-pass frequency", None))
        self.label_3.setText(QCoreApplication.translate("LowPassFilterSettings", u"Order", None))
    # retranslateUi

