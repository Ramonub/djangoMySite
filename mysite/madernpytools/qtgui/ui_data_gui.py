# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'data_gui.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from madernpytools.qtgui.tool_view_widget import ToolViewWidget
from madernpytools.qtgui.signal_viewer import SignalViewer
from madernpytools.qtgui.autosave_log_widget import AutosaveLogWidget


class Ui_DataGui(object):
    def setupUi(self, DataGui):
        if not DataGui.objectName():
            DataGui.setObjectName(u"DataGui")
        DataGui.resize(768, 419)
        self.actionConnect = QAction(DataGui)
        self.actionConnect.setObjectName(u"actionConnect")
        self.centralwidget = QWidget(DataGui)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.log_control = AutosaveLogWidget(self.centralwidget)
        self.log_control.setObjectName(u"log_control")
        self.log_control.setMinimumSize(QSize(500, 0))
        self.log_control.setMaximumSize(QSize(1000, 16777215))

        self.gridLayout_2.addWidget(self.log_control, 0, 0, 1, 1)

        self.horizontalSpacer = QSpacerItem(340, 20, QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 1, 1, 1)

        self.tab_time = QTabWidget(self.centralwidget)
        self.tab_time.setObjectName(u"tab_time")
        font = QFont()
        font.setBold(False)
        font.setWeight(50)
        self.tab_time.setFont(font)
        self.layout = QWidget()
        self.layout.setObjectName(u"layout")
        self.gridLayout = QGridLayout(self.layout)
        self.gridLayout.setObjectName(u"gridLayout")
        self.widget = QWidget(self.layout)
        self.widget.setObjectName(u"widget")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.horizontalLayout_3 = QHBoxLayout(self.widget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.btn_zero = QPushButton(self.widget)
        self.btn_zero.setObjectName(u"btn_zero")

        self.horizontalLayout_3.addWidget(self.btn_zero)

        self.horizontalSpacer_4 = QSpacerItem(624, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_4)


        self.gridLayout.addWidget(self.widget, 0, 0, 1, 3)

        self.verticalSpacer = QSpacerItem(20, 141, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 1, 1, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(248, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 2, 0, 1, 1)

        self.tool_view = ToolViewWidget(self.layout)
        self.tool_view.setObjectName(u"tool_view")

        self.gridLayout.addWidget(self.tool_view, 2, 1, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(247, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 2, 2, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 140, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 3, 1, 1, 1)

        self.tab_time.addTab(self.layout, "")
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

        self.gridLayout_2.addWidget(self.tab_time, 1, 0, 1, 2)

        DataGui.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(DataGui)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 768, 21))
        self.menuDevice = QMenu(self.menubar)
        self.menuDevice.setObjectName(u"menuDevice")
        DataGui.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(DataGui)
        self.statusbar.setObjectName(u"statusbar")
        DataGui.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuDevice.menuAction())
        self.menuDevice.addAction(self.actionConnect)

        self.retranslateUi(DataGui)

        self.tab_time.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(DataGui)
    # setupUi

    def retranslateUi(self, DataGui):
        DataGui.setWindowTitle(QCoreApplication.translate("DataGui", u"NI-Measurement GUI", None))
        self.actionConnect.setText(QCoreApplication.translate("DataGui", u"Connect", None))
#if QT_CONFIG(tooltip)
        self.layout.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.btn_zero.setText(QCoreApplication.translate("DataGui", u"Zero Sensors", None))
        self.tab_time.setTabText(self.tab_time.indexOf(self.layout), QCoreApplication.translate("DataGui", u"Signal View", None))
        self.tab_time.setTabText(self.tab_time.indexOf(self.tab_timedomain), QCoreApplication.translate("DataGui", u"Time-View", None))
        self.tab_time.setTabText(self.tab_time.indexOf(self.tab_frequencydomain), QCoreApplication.translate("DataGui", u"Frequency-View", None))
        self.menuDevice.setTitle(QCoreApplication.translate("DataGui", u"Device", None))
    # retranslateUi

