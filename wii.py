# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21, 2019

@author: David Engel (after Adrian Schütz)
"""

import time
import pickle
import pywinusb.hid as hid
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class WiiBBRuntime(QWidget):
    def __init__(self):
        super(WiiBBRuntime, self).__init__()

        self.all_hids = hid.find_all_hid_devices()
        self.index_option = 15  # check for device number with hidtesting script
        self.int_option = int(self.index_option)

        self.result_wii = {'data': [[]], 'start_time': [[]], 'ts': [], 'frame': [[]]}
        self.record = False
        self.first = True
        self.setWindowTitle("Wii Balance Board")

        self.rec_button = QPushButton('Start Recording', self)
        self.save_button = QPushButton('Save', self)
        self.new_button = QPushButton('New File', self)
        self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')

        self.layout = QGridLayout()
        self.layout.addWidget(self.rec_button)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.new_button)
        self.setLayout(self.layout)

        self.timer = QTimer(self)
        # self.timer.timeout.connect(self.wiibb_run)
        self.timer.start(0)

        self.new_button.clicked.connect(self.new_file)
        self.save_button.clicked.connect(self.save_data)
        self.rec_button.clicked.connect(self.record_switch)

    def record_switch(self, start_time=None):
        self.record = self.record ^ True

        if self.record:
            self.start_time = time.time()
            self.curframe = 1
            if self.first:
                self.result_wii = {'start_time': [self.start_time], 'ts': [[]], 'data': [[]], 'frame': [[]]}
                self.first = False
            else:
                self.result_wii['start_time'].append(self.start_time)
                self.result_wii['ts'].append([])
                self.result_wii['data'].append([])
                self.result_wii['frame'].append([])

            if self.int_option:

                self.device = self.all_hids[self.int_option - 1]

                self.device.open()
                self.report = self.device.find_output_reports()
                # print(report[0])
                buffer = [0x15] * 22
                # print(buffer)
                self.report[0].set_raw_data(buffer)
                self.report[0].send()
                # set custom raw data handler
                self.device.set_raw_data_handler(self.record_handler)
            self.rec_button.setText("Stop")

        else:
            self.save_data()
            self.device.close()
            print("Recording finished, data saved.")
            self.rec_button.setText("Start Recording")


    def new_file(self):
        self.save_data()
        self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
        self.result_wii = {'start_time': [self.start_time], 'ts': [], 'data': []}


    def save_data(self):
        if self.save_path is not "":
            output = open(self.save_path + ".p", 'wb')
            pickle.dump(self.result_wii, output)

        else:
            self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
            if self.save_path is not "":
                output = open(self.save_path + ".p", 'wb')
                pickle.dump(self.result_wii, output)

    def closeEvent(self, event):

        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            reply = QMessageBox.question(
                self, "Message",
                "Wanna Save?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if self.save_path is not "":
                    output = open(self.save_path + ".p", 'wb')
                    pickle.dump(self.result_wii, output)

                else:
                    self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
                    if self.save_path is not "":
                        output = open(self.save_path + ".p", 'wb')

                        pickle.dump(self.result_wii, output)
                app.quit()
        else:
            event.ignore()

    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Escape:
            self.close()

    def record_handler(self, data):
        print("Recording...")
        temp = []
        for i in range(len(data)):
            temp.append(data[i])
        self.result_wii['ts'][-1].append(time.time() - self.start_time)
        self.result_wii['data'][-1].append(temp)
        self.result_wii['frame'][-1].append(self.curframe)
        self.curframe += 1


if __name__ == '__main__':
    # first be kind with local encodings
    import sys

    if sys.version_info >= (3,):
        # as is, don't handle unicodes
        unicode = str
        raw_input = input
    else:
        # allow to show encoded strings
        import codecs
        sys.stdout = codecs.getwriter('mbcs')(sys.stdout)

    app = QApplication(sys.argv)

    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = WiiBBRuntime()
    window.show()

sys.exit(app.exec_())


