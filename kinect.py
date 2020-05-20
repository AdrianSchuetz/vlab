# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:35:09 2018

@author: adrian
"""

import numpy as np
import pickle
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import sys

import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import ctypes
import _ctypes

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread


def get_3D_joint_pos(joint, joint_index):
    pos = joint[joint_index].Position

    return [pos.x, pos.y, pos.z]


class GLWidget(QOpenGLWidget):
    def __init__(self, parent):
        super(GLWidget, self).__init__(parent)
        self.parent = parent
        self.record = False
        self.show_template = False
        self.background = QBrush(Qt.black)
        self.circlePen = QPen(Qt.white)
        self.circlePen.setWidth(10)
        self.kinect = None
        self.bodies = None
        self.x_size = 400
        self.y_size = 400
        self.x_scale = 1
        self.y_scale = 1
        self.setFixedSize(self.x_size, self.y_size)
        self.setAutoFillBackground(False)
        self.trial = 0

    def paintEvent(self, event):

        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.record:
            timestamp = time.time() - self.parent.start_time
        painter.fillRect(event.rect(), self.background)

        painter.save()
        painter.setPen(self.circlePen)
        if self.bodies is not None and self.kinect is not None:
            for j in range(0, 6):
                pos_data = []
                body = self.bodies.bodies[j]
                if not body.is_tracked:
                    continue
                joints = body.joints
                joint_coords = self.kinect.body_joints_to_color_space(joints)

                for i in range(0, 25):

                    if (joints[i].TrackingState != PyKinectV2.TrackingState_NotTracked and
                            joints[i].TrackingState != PyKinectV2.TrackingState_Inferred):
                        pos_data.append(get_3D_joint_pos(joints, i))
                        painter.drawPoint(
                            # TODO: fix weird distortion
                            self.x_scale * (joint_coords[i].x / 1920 / 2 * self.x_size) + (self.x_size / 4),
                            self.y_scale * (joint_coords[i].y / 1080 / 2 * self.y_size) + (self.y_size / 4))


                    else:
                        pos_data.append([np.nan, np.nan, np.nan])
                if pos_data is not [] and self.record:
                    self.parent.result_kinect['time'][-1].append(timestamp)
                    self.parent.result_kinect['pos_data'][-1].append(pos_data)
        painter.restore()

        painter.end()


class kinect_v2_runtime(QWidget):
    def __init__(self):
        super(kinect_v2_runtime, self).__init__()
        self.start_time = 0
        self.canvas = GLWidget(self)
        self.result_kinect = {'start_time': [self.start_time], 'time': [[]], 'pos_data': [[]],
                       'scale': [[self.canvas.x_scale, self.canvas.y_scale]], 'comment': [[]]}
        self.first = True
        self.setWindowTitle("Kinect V2")
        self.kinect_handle = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
        self.rec_button = QPushButton('Start Recording', self)
        self.save_button = QPushButton('Save', self)
        self.new_button = QPushButton('New File', self)
        self.show_template = QPushButton('Show Template', self)
        self.add_comment = QPushButton('Add Comment', self)
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setToolTip('X Scale')
        self.x_slider.setMaximum(100)
        self.x_slider.setMinimum(0)
        self.x_slider.setValue(50)
        self.x_slider.setTickInterval(10)
        self.x_slider.setTickPosition(QSlider.TicksBothSides)

        self.y_slider = QSlider(Qt.Vertical)
        self.x_slider.setToolTip('Y Scale')
        self.y_slider.setMaximum(100)
        self.y_slider.setMinimum(0)
        self.y_slider.setValue(50)
        self.y_slider.setTickInterval(10)
        self.y_slider.setTickPosition(QSlider.TicksBothSides)
        # self.bodies_signal = pyqtSignal(object, name='new_body_frame')
        self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')

        self.bodies = None
        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas, 0, 0)
        self.layout.addWidget(self.rec_button, 0, 2, 6, 1)
        self.layout.addWidget(self.save_button, 1, 2, )
        self.layout.addWidget(self.show_template, 0, 2, 1, 1)
        self.layout.addWidget(self.add_comment, 0, 2, 15, 1)
        self.layout.addWidget(self.new_button, 0, 2, 20, 1)
        self.layout.addWidget(self.x_slider, 1, 0)
        self.layout.addWidget(self.y_slider, 0, 1)
        self.setLayout(self.layout)

        self.timer = QTimer(self)

        self.timer.timeout.connect(self.get_kinect_body)
        self.timer.start(0)

        self.new_button.clicked.connect(self.new_file)
        self.save_button.clicked.connect(self.save_data)
        self.show_template.clicked.connect(self.template_switch)
        self.rec_button.clicked.connect(self.record_switch)
        self.add_comment.clicked.connect(self.get_comment)
        self.x_slider.valueChanged.connect(self.rescale)
        self.y_slider.valueChanged.connect(self.rescale)

    def record_switch(self, start_time=None):
        self.canvas.record = self.canvas.record ^ True

        if self.canvas.record:
            self.rec_button.setText("Recording ...")
            if start_time is None:
                self.start_time = start_time
            else:
                self.start_time = time.time()
            if self.first:
                self.result_kinect = {'start_time': [self.start_time], 'time': [[]], 'pos_data': [[]],
                               'scale': [[self.canvas.x_scale, self.canvas.y_scale]], 'comment': [[]]}
                self.first = False
            else:
                self.result_kinect['start_time'].append(self.start_time)
                self.result_kinect['time'].append([])
                self.result_kinect['pos_data'].append([])
                self.result_kinect['scale'].append([[self.canvas.x_scale, self.canvas.y_scale]])
                self.result_kinect['comment'].append([])

        else:
            self.save_data()
            self.rec_button.setText("Start Recording")

    def new_file(self):
        self.save_data()
        self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
        self.result_kinect = {'start_time': [self.start_time], 'time': [[]], 'pos_data': [[]],
                       'scale': [[self.canvas.x_scale, self.canvas.y_scale]], 'comment': [[]]}

    def get_comment(self):
        # TODO: add threading
        if not self.canvas.record:
            text, ok = QInputDialog.getText(self, 'Add Comment',
                                            'Enter your comment:')
            if ok:
                self.add_comment_data(text)

    def add_comment_data(self, comment=''):
        self.result_kinect['comment'][-1].append(comment)

    def template_switch(self):

        self.canvas.show_template = self.canvas.show_template ^ True

    def rescale(self):
        self.canvas.x_scale = 1 + (self.x_slider.value() - 50) / 100
        self.canvas.y_scale = 1 + (self.y_slider.value() - 50) / 100
        self.result_kinect['scale'][-1] = [self.canvas.x_scale, self.canvas.y_scale]

    def get_kinect_body(self):

        if self.kinect_handle.has_new_body_frame():

            self.bodies = self.kinect_handle.get_last_body_frame()

            if self.bodies is not None:
                # self.bodies_signal.emit(bodies)

                self.canvas.kinect = self.kinect_handle
                self.canvas.bodies = self.bodies
                self.canvas.update()

    def save_data(self):
        if self.save_path is not "":
            output = open(self.save_path + ".p", 'wb')
            pickle.dump(self.result_kinect, output)

        else:
            self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
            if self.save_path is not "":
                output = open(self.save_path + ".p", 'wb')
                pickle.dump(self.result_kinect, output)

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
                    pickle.dump(self.result_kinect, output)

                else:
                    self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
                    if self.save_path is not "":
                        output = open(self.save_path + ".p", 'wb')

                        pickle.dump(self.result_kinect, output)
                app.quit()

        else:
            event.ignore()

    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Escape:
            self.close()


# TODO: Add external control
if __name__ == '__main__':
    app = QApplication(sys.argv)

    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = kinect_v2_runtime()
    window.show()

sys.exit(app.exec_())
