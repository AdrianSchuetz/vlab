# -*- coding: utf-8 -*-
"""
Created on Thu Feb 7 17:26 2019

@author: David Engel after Adrian Schütz
"""

# !/bin/env python

# file qt_pyside_app.py

import sys
from PyQt5.QtOpenGL import QGLWidget, QGLFormat
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from OpenGL.GL import *
import numpy as np
import openvr
import pyrr
import time
import pickle
import random as rand
import pandas as pd
import math


""""""""""""""""""""""""""""
Initializing Global Functions
"""""""""""""""""""""""""""


def load_ground_truth(path):
    gt_df = pd.read_csv(path, delimiter=',')

    gt = np.ones((np.size(gt_df, 0), 3))
    factor = 1 / 30
    gt[:, 0] = gt_df.x * factor
    gt[:, 1] = gt_df.y * factor
    gt[:, 2] = gt_df.z * factor
    # gt[:, 3] = gt_df.r
    # gt[:, 4] = gt_df.g
    # gt[:, 5] = gt_df.b

    #    if np.size(gt_df, 1) == 3:
    #        gt[:, 2] = gt_df.z
    #    else:
    #        gt[:, 2] = np.zeros((np.size(gt_df, 0),))

    gt_list = []
    for i in range(np.size(gt_df, 0)):
        gt_list.append(list(gt[i, :]))
    return gt_list


def randlist(num_dots, dot_range):
    rand_list = []

    for i in range(0, num_dots):
        x = rand.uniform(dot_range[0], dot_range[1])
        rand_list.append(x)

    return rand_list


def pointlist(num_dots, dot_range):
    xyz_rgb = np.ndarray((num_dots, 6))
    xyz_rgb[:, 0] = randlist(num_dots, dot_range[0, :])
    xyz_rgb[:, 1] = randlist(num_dots, dot_range[1, :])
    xyz_rgb[:, 2] = randlist(num_dots, dot_range[2, :])

    xyz_rgb[:, 3:6] = np.ones((num_dots, 3))

    point_list = list(xyz_rgb.reshape(np.size(xyz_rgb, 0) * np.size(xyz_rgb, 1), ))
    return point_list


def gen_poslist_zylinder(density=1, origin=[0, 0, 0], radius=0, height=0):
    # get point positions
    num_points = int(max(1, radius ** 2 * height * density))
    print(num_points)
    angle = 2 * np.pi * np.random.rand(1, num_points)
    x_pos = np.sin(angle) * radius - origin[0]
    z_pos = np.cos(angle) * radius - origin[2]

    y_pos = (np.random.rand(1, num_points) - 0.5) * height - origin[1]
    print(y_pos)

    # lifetime
    lt = np.random.rand(1, num_points)
    ltp = np.random.rand(1, num_points)
    # lt = 0.2 * np.ones(self.num_points)

    pos_list = np.ones(5 * num_points)
    pos_list[0::5] = x_pos
    pos_list[1::5] = y_pos
    pos_list[2::5] = z_pos
    pos_list[3::5] = lt
    pos_list[4::5] = ltp

    return np.array(pos_list, dtype=np.float32)


def gen_poslist_tunnel(density=1, origin=None, radius=0, length=None, lifetime=False):
    # get point positions
    if length is None:
        length = 30
    else:
        length = length

    if origin is None:
        origin = [0, user_height, 0]

    num_points = int(max(1, radius ** 2 * length * density))
    # print(num_points)
    angle = 2 * np.pi * np.random.rand(1, num_points)
    x_pos = np.sin(angle) * radius - origin[0]
    y_pos = np.cos(angle) * radius - origin[1]
    z_pos = (np.random.rand(1, num_points) - 0.5) * length - origin[2]
    # print(z_pos)

    if lifetime:
        lt = np.random.rand(1, num_points)
        ltp = np.random.rand(1, num_points)
        # lt = 0.2 * np.ones(self.num_points)

        pos_list = np.ones(5 * num_points)
        pos_list[0::5] = x_pos
        pos_list[1::5] = y_pos
        pos_list[2::5] = z_pos
        pos_list[3::5] = lt
        pos_list[4::5] = ltp
    else:
        lt = np.random.rand(1, num_points) * 0.0 + 0.0 * np.ones(num_points)
        ltp = np.random.rand(1, num_points)

        pos_list = np.ones(5 * num_points)
        pos_list[0::5] = x_pos
        pos_list[1::5] = y_pos
        pos_list[2::5] = z_pos
        pos_list[3::5] = lt
        pos_list[4::5] = ltp

    return np.array(pos_list, dtype=np.float32)


def gen_poslist(density=1, origin=[0, 0, 0], size=[0, 0, 0], lifetime=True):
    # get point positions
    num_points = int(max(1, size[0] * size[1] * size[2] * density))
    
    x_pos = size[0] * (np.random.rand(1, num_points) - 0.5) - origin[0]
    y_pos = size[1] * (np.random.rand(1, num_points) - 0.5) - origin[1]
    z_pos = size[2] * (np.random.rand(1, num_points) - 0.5) - origin[2]

    # lifetime
    if lifetime:
        lt = np.random.rand(1, num_points) * 0.0 + 0.0 * np.ones(num_points)
        ltp = np.random.rand(1, num_points)
        # lt = 0.2 * np.ones(self.num_points)
        print('life')

        pos_list = np.ones(5 * num_points)
        pos_list[0::5] = x_pos
        pos_list[1::5] = y_pos
        pos_list[2::5] = z_pos
        pos_list[3::5] = lt
        pos_list[4::5] = ltp
    else:
        pos_list = np.ones(3 * num_points)
        pos_list[0::3] = x_pos
        pos_list[1::3] = y_pos
        pos_list[2::3] = z_pos

    return np.array(pos_list, dtype=np.float32)


def matrixForOpenVrMatrix(mat):
    if len(mat.m) == 4:  # HmdMatrix44_t?
        result = np.matrix(
            ((mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0]),
             (mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1]),
             (mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2]),
             (mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]),)
            , np.float32)
        return result
    elif len(mat.m) == 3:  # HmdMatrix34_t?
        result = np.matrix(
            ((mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0),
             (mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0),
             (mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0),
             (mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0),)
            , np.float32)
        return result


class CustTimer(object):
    def __init__(self, running=True):
        self.starttime = time.time()
        self.running = running

    def get_time(self):
        if self.running:
            return time.time() - self.starttime
        else:
            return 0

    def start(self):
        if not self.running:
            self.running = True
            self.reset()

    def stop(self):
        if self.running:
            self.running = False
            self.reset()

    def reset(self):
        self.starttime = time.time()

    def synch(self, timer):
        self.starttime = timer.get_time()





"""""""""""""""""""""""""""""
QT Windows & Main Application
"""""""""""""""""""""""""""""


class MyGlWidget(QGLWidget):
    "PySideApp uses Qt library to create an opengl context, listen to keyboard events, and clean up"

    def __init__(self, renderer, glformat, app):
        "Creates an OpenGL context and a window, and acquires OpenGL resources"
        super(MyGlWidget, self).__init__(glformat)
        self.renderer = renderer
        self.app = app
        # Use a timer to rerender as fast as possible
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        # set ifi. 0 = as fast as possible
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.render_vr)
        # Accept keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

    def __enter__(self):
        "setup for RAII using 'with' keyword"
        return self

    def __exit__(self, type_arg, value, traceback):
        "cleanup for RAII using 'with' keyword"
        self.dispose_gl()

    def initializeGL(self):
        print("initialize OpenGL...")

        print("initialize Trials...")

        if self.renderer is not None:
            vrsystem = self.renderer.init_gl()
            print("Vrsystem")
        for trial in self.app.control.trial_list:
            trial.init_gl(vrsystem)
        self.app.control.calibration.init_gl(vrsystem)
        self.timer.start()

        print("successfully initialized OpenGL...")

    def paintGL(self):
        "render scene one time"

        self.renderer.render_scene()
        self.swapBuffers()  # Seems OK even in single-buffer mode

    def render_vr(self):
        self.makeCurrent()
        self.paintGL()
        self.doneCurrent()
        self.timer.start()  # render again real soon now

    def disposeGL(self):
        if self.renderer is not None:
            self.makeCurrent()
            self.renderer.dispose_gl()
            self.doneCurrent()

    def keyPressEvent(self, event):
        "press ESCAPE to quit the application"
        key = event.key()
        if key == Qt.Key_Escape:
            self.app.quit()


class MyQt5App(QApplication):
    def __init__(self, renderer, trial_list):
        QApplication.__init__(self, sys.argv)
        print("initialize Vive runtime")
        self.window = QMainWindow()
        self.window.setWindowTitle("Vive Runtime")
        self.window.resize(800, 600)
        # Get OpenGL 4.1 context
        glformat = QGLFormat()
        glformat.setVersion(4, 1)
        glformat.setProfile(QGLFormat.CoreProfile)
        glformat.setDoubleBuffer(False)
        
        self.control = TrialControl(trial_list)
        self.glwidget = MyGlWidget(renderer, glformat, self)
        self.window.setCentralWidget(self.glwidget)
        

        self.control.show()  # this is the trial control window
        self.window.show()  # this is the OpenGL window which shows the stimulus
        print("Vive runtime initialized")

    def __enter__(self):
        "setup for RAII using 'with' keyword"
        return self

    def __exit__(self, type_arg, value, traceback):
        "cleanup for RAII using 'with' keyword"
        self.glwidget.disposeGL()

    def append_trial(self, trial, n=1):
        self.control.append_trial(trial, n)

    def run_loop(self):
        retval = self.exec_()
        sys.exit(retval)


class CalibControlPanel(QMainWindow):
    def __init__(self, trial):
        super(CalibControlPanel, self).__init__()
        self.setWindowTitle("Calibration Control Panel")
        self.trial = trial
        self.ts = np.empty((len(self.trial.pos_list), 1))
        self.done = np.zeros((len(self.trial.pos_list), 1))
        self.savepos = np.empty((len(self.trial.pos_list), 3))
        self.nb = QPushButton('Next Dot')
        self.rb = QPushButton('Previous Dot')
        self.resb = QPushButton('Reset Calibration')
        self.acc = QPushButton('Accept Calibration')
        layout = QVBoxLayout()
        layout.addWidget(self.nb)
        layout.addWidget(self.rb)
        layout.addWidget(self.resb)
        layout.addWidget(self.acc)
        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)

        self.nb.pressed.connect(self.next)
        self.rb.pressed.connect(self.backward)
        self.resb.pressed.connect(self.reset)
        self.acc.pressed.connect(self.accept)
        self.pointer = 0
        self.pos_order = np.random.permutation(range(len(self.trial.pos_list)))
        self.curpos = self.pos_order[self.pointer]
        self.trial.actor_list[0].move(self.trial.pos_list[self.curpos])
        self.statusBar().showMessage(
            'Calibration dot:' + str(int(np.sum(self.done))) + '/' + str(len(self.trial.pos_list)))

    def reset(self):
        self.pos_order = np.random.permutation(range(len(self.trial.pos_list)))
        self.pointer = 0
        self.curpos = self.pos_order[self.pointer]
        self.trial.actor_list[0].move(self.trial.pos_list[self.curpos])
        print('Restart Calibration.')

    def next(self):
        if self.pointer < (len(self.trial.pos_list) - 1):
            self.pointer += 1
            self.curpos = self.pos_order[self.pointer]
            self.trial.actor_list[0].move(self.trial.pos_list[self.curpos])
        else:
            print("calibration done")
        self.statusBar().showMessage(
            'Calibration dot:' + str(int(np.sum(self.done))) + '/' + str(len(self.trial.pos_list)))

    def backward(self):
        if self.pointer is not 0:
            self.pointer -= 1
            self.curpos = self.pos_order[self.pointer]
            self.trial.actor_list[0].move(self.trial.pos_list[self.curpos])
        else:
            print("Can't go back further")
        return

    def accept(self):
        self.ts[self.curpos] = time.time()
        self.done[self.curpos] = 1
        self.savepos[self.curpos] = self.trial.pos_list[self.curpos]
        if np.prod(self.done) == 1:
            self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
            if self.save_path is not "" and not None:
                output = open(self.save_path + ".p", 'wb')

                pickle.dump([self.ts, self.savepos], output)
            else:
                print("calibration data was not saved")
            self.trial.end()
        self.next()


class TrialControl(QMainWindow):
    def __init__(self, trial_list):
        super(TrialControl, self).__init__()
        print("init trial control panel")
        self.done = False
        self.setWindowTitle("Trial Control Panel")
        self.pointer = 0
        self.renderer = renderer
        self.nb = QPushButton('Next')
        self.rb = QPushButton('Return')
        self.resb = QPushButton('Reset Trial')
        self.calib = QPushButton('Calibrate Eye Tracker')
        self.save_dat = QPushButton('Save Data')

        # Get OpenGL 4.1 context
        #glformat = QGLFormat()
        #glformat.setVersion(4, 1)
        #glformat.setProfile(QGLFormat.CoreProfile)
        #glformat.setDoubleBuffer(False)

        #self.stim_window = MyGlWidget(renderer, glformat, self)

        layout = QVBoxLayout()
        layout.addWidget(self.nb)
        layout.addWidget(self.rb)
        layout.addWidget(self.resb)
        layout.addWidget(self.calib)
        layout.addWidget(self.save_dat)
        # layout.addWidget(self.stim_window)
        self.voidtrial = Trial()
        menubar = self.menuBar()
        addonmenubar = menubar.addMenu('Settings')
        viewaddonact = QAction('Choose Calibration', self, checkable=False)

        viewaddonact.triggered.connect(self.choose_calibration)
        addonmenubar.addAction(viewaddonact)

        w = QWidget()
        w.setLayout(layout)
        self.calibration = Calibration(trial_control=self)
        self.setCentralWidget(w)
        self.nb.pressed.connect(self.next)
        self.rb.pressed.connect(self.backward)
        self.resb.pressed.connect(self.reset)
        self.calib.pressed.connect(self.calibrate)
        self.save_dat.pressed.connect(self.save_data)
        self.curtrial_nr = -1

        self.trial_list = trial_list

        for trial in self.trial_list:
            trial.signals.result.connect(self.trial_end)

        self.trial_order = list(np.random.permutation(list(range(np.size(self.trial_list)))))

        self.renderer.curtrial = self.voidtrial
        self.statusBar().showMessage('Click Next to start')

        self.result_dic = {"trial_data": []}
        self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')

    def save_data(self):
        if self.save_path is not "":
            output = open(self.save_path + ".p", 'wb')

            pickle.dump(self.result_dic, output)
        else:
            self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
            output = open(self.save_path + ".p", 'wb')
            pickle.dump(self.result_dic, output)

    def trial_end(self, data):

        self.result_dic["trial_data"].append(data)

        self.next()

    def reset(self):
        self.renderer.curtrial.reset()

    def all_reset(self):
        for trial in self.trial_list:
            trial.reset()

        # self.update_trial_order(True)
        self.curtrial_nr = -1
        self.renderer.curtrial = self.voidtrial

    def next(self):
        if self.curtrial_nr < (len(self.trial_order) - 1):
            self.curtrial_nr += 1
            self.renderer.curtrial = self.trial_list[int(self.trial_order[self.curtrial_nr])]
            self.reset()
            self.statusBar().showMessage(
                'Trial ' + str(self.curtrial_nr + 1) + "/" + str(len(self.trial_order)) + ": " +
                self.renderer.curtrial.name)
        else:
            print("no more trials")
            self.renderer.curtrial = self.voidtrial
            if self.curtrial_nr == (len(self.trial_order) - 1):
                self.curtrial_nr += 1
            self.statusBar().showMessage('Finished. Click Save Data to save.')

    def backward(self):

        if self.curtrial_nr is not 0:
            self.curtrial_nr -= 1
            self.renderer.curtrial = self.trial_list[int(self.trial_order[self.curtrial_nr])]

            self.reset()
        else:
            print("can't go back further")
        self.statusBar().showMessage('Trial ' + str(self.curtrial_nr + 1) + "/" + str(len(self.trial_order)) + ": " +
                                     self.renderer.curtrial.name)

    def calibrate(self):
        self.calibration.start()
        self.renderer.curtrial = self.calibration

    # TODO: rework!!!!
    def update_trial_order(self, randomize=False, keep_first=False, master_order=[]):
        print('before', self.trial_order)
        if not master_order:
            if randomize:
                if keep_first:
                    self.trial_order = self.trial_order[0:1] + list(np.random.permutation(self.trial_order[1:]))
                else:
                    self.trial_order = list(np.random.permutation(self.trial_order))
        else:
            self.trial_order = master_order
        print('after', self.trial_order)
        self.renderer.curtrial = self.trial_list[int(self.trial_order[self.curtrial_nr])]
        self.statusBar().showMessage('Trial ' + str(self.curtrial_nr + 1) + "/" + str(len(self.trial_order)) + ": " +
                                     self.renderer.curtrial.name)

    def append_trial(self, trial, reps=1):

        self.trial_list.append(trial)
        self.trial_order = self.trial_order + list(np.ones((reps,)) * (len(self.trial_list) - 1))

        self.renderer.curtrial = self.trial_list[int(self.trial_order[self.curtrial_nr])]
        print(trial.name + " Trial appended")
        self.statusBar().showMessage('Trial ' + str(self.curtrial_nr + 1) + "/" + str(len(self.trial_order)) + ": " +
                                     self.renderer.curtrial.name)

    def choose_calibration(self):
        calib_path, _ = QFileDialog.getOpenFileName(self, 'Load Calibration')
        temp = load_ground_truth(calib_path)
        self.calibration.pos_list = temp

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
                    pickle.dump(self.result_dic, output)

                else:
                    self.save_path, _ = QFileDialog.getSaveFileName(self, 'Save Path')
                    if self.save_path is not "":
                        output = open(self.save_path + ".p", 'wb')

                        pickle.dump(self.result_dic, output)
                self.close()
        else:
            event.ignore()

    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Escape:
            self.closeEvent(event)


""""""""""""""""""""""""""""
Trial Management & Objects
"""""""""""""""""""""""""""""


class Trial(QObject):
    # init funktion. hier sollen alle werte, die für jeden aufruf des trials gleich sind
    # gesetzt werden (z.B. Name, Dauer, Actor). Außerdem sollen hier alle variablen initialisiert werden
    def __init__(self, dur=None, name=None):
        super(Trial, self).__init__()
        self.left_cid = None
        self.right_cid = None
        if dur is not None:
            self.dur = dur
        else:
            self.dur = np.inf
        if name is not None:
            self.name = name
        else:
            self.name = "Default"
        self.curframe = 0
        self.sr = 90
        self.actor_list = []
        self.vr_system = None
        self.start_time = time.time()
        self.data = {'name': self.name, 'starttime': self.start_time, 'displaytarget_key': ['l', 'r', 'm'],
                     'ts': [], 'framenr': [], 'headmatrix': [], 'displaytarget': [], 'userheight': user_height}
        self.signals = WorkerSignals()
        self.timer = CustTimer()

    def append_actor(self, actor):
        self.actor_list.append(actor)

    def init_gl(self, vr_system=None):

        print("init actors")
        for actor in self.actor_list:
            actor.init_gl()
        if vr_system is not None:
            self.vr_system = vr_system
            for i in range(openvr.k_unMaxTrackedDeviceCount):

                device_class = self.vr_system.getTrackedDeviceClass(i)
                if device_class == openvr.TrackedDeviceClass_Controller:
                    role = self.vr_system.getControllerRoleForTrackedDeviceIndex(i)
                    if role == openvr.TrackedControllerRole_RightHand:
                        print("found right hand")
                        self.right_cid = i
                    if role == openvr.TrackedControllerRole_LeftHand:
                        print("found left hand")
                        self.left_cid = i

    # wird bei jedem trial start aufgerufen. hier muss der aktuelle frame resettet werden. außerdem können hier
    # random werte neu gewürfelt werden. außerdem werden alle actor auf anfang zurückgesetzt.

    def reset(self):
        self.curframe = 0
        self.timer.reset()
        self.start_time = time.time()
        self.data = {'name': self.name, 'starttime': self.start_time, 'displaytarget_key': ['l', 'r', 'm'],
                     'ts': [], 'framenr': [], 'headmatrix': [], 'displaytarget': [], 'userheight': user_height}
        for actor in self.actor_list:
            actor.reset()

    # diese methode wird bei jedem frame aufgerufen. hier muss der aktuelle frame hochgezählt werden und gecheckt werden,
    # ob der trial weiter gehen soll oder nicht
    def display_gl(self, modelview=None, projection=None):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        self.curframe += 1 / 3
        self.data['ts'].append(self.timer.get_time())

        if modelview is not None:
            self.data['headmatrix'].append((modelview))
        else:
            self.data['headmatrix'].append(np.empty((4, 4,)) * np.nan)

        self.data['displaytarget'].append(int(self.curframe * 3 + 1) % 3)

        self.data['framenr'].append(int(np.ceil(self.curframe)))
        if self.curframe >= self.dur:
            self.end()

    # wird aufgerufen wenn der trial endet. muss immer True zurückgeben, kann aber auch andere Ding, bei Bedarf z.B. Kontrolpanele schließen

    def end(self):

        self.signals.result.emit(self.data)


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Calibration(Trial):
    def __init__(self, pos_list=None, trial_control=None):
        super(Calibration, self).__init__()

        if pos_list is not None:
            self.pos_list = pos_list
        else:
            #pos_list_temp = load_ground_truth(
               # 'C:\\Users\\VRLab\\Dropbox\\PhD\\Experimente\\PyCharmSetup\\BodySway_VR\\Calib_Collection\\checkerboard2DReduced.txt')
            pos_list_temp = load_ground_truth(
                'C:\\Users\\VRLab\\Desktop\\BodySway_VR\\Calib_Collection\\checkerboard2DReduced.txt')

            
            # pos_list_temp = load_ground_truth('C:\\Users\\VRLab\\Dropbox\\PYTHON\\PeterStimulus\\Calib_Collection\\checkerboard2DReduced.txt')
            self.pos_list = pos_list_temp
        self.tc = trial_control

        point = GlPointStatic(size=100)

        self.actor_list.append(point)
        self.name = "Calibration"

    def start(self):
        self.control = CalibControlPanel(self)
        self.control.show()

    def append_actor(self, actor):
        print("calibration can not append further actors")
        return

    def reset(self):
        super(Calibration, self).reset()
        self.control.reset()

    def display_gl(self, modelview=None, projection=None):
        glClearColor(0.5, 0.5, 0.5, 0.0)
        for actor in self.actor_list:
            actor.draw_gl(projection=projection)
        self.curframe += 1
        if self.curframe >= self.dur:
            self.end()

    def end(self):

        self.control.close()
        if self.tc is not None:
            self.tc.all_reset()


class DriftCorr(Trial):
    def __init__(self, name=None, size=None, pointlist=None):
        super(DriftCorr, self).__init__(name=name)

        self.ms = 0
        self.stat_time = 0

        if pointlist is not None:
            self.pointlist = pointlist
        else:
            pointlist_temp = np.array([1.0, 0.0, 0.0,
                                       0.0, 1.0, 0.0,
                                       -1.0, 0.0, 0.0,
                                       0.0, -1.0, 0.0,
                                       1.0, 1.0, 0.0,
                                       -1.0, 1.0, 0.0,
                                       1.0, -1.0, 0.0,
                                       -1.0, -1.0, 0.0]) / 30
            self.pointlist = np.concatenate([pointlist_temp, 2 * pointlist_temp, 3 * pointlist_temp,
                                             4 * pointlist_temp, 5 * pointlist_temp], axis=0)

        points = GlPointStatic(size=size, points=self.pointlist)
        centerpoint = GlPointStatic(size=size, color=[1.0, 0.0, 0.0])

        self.name = "DriftCorrection"
        self.data['motiononset'] = self.ms
        self.data['statictime'] = self.stat_time
        self.actor_list = [points, centerpoint]

    def display_gl(self, modelview=None, projection=None):
        super(DriftCorr, self).display_gl(modelview=modelview, projection=projection)

        glClearColor(0.5, 0.5, 0.5, 0.0)

        for actor in self.actor_list:
            actor.draw_gl(projection=projection)

        if self.timer.get_time() >= 5:
             self.end()


class InterTrial(Trial):
    def __init__(self, dur=None, name=None, cloud=None, particlesize=None, fpcoords=None):
        super(InterTrial, self).__init__(dur, name=name)
        if cloud is None:
            cloud = gen_poslist(1, size=[100, 1, 100])
        if particlesize is None:
            particlesize = 1000
        if fpcoords is None:
            fpcoords = np.array([0, user_height, -24, 0, 0])

        c = GlFadePoint(size=particlesize, points=cloud)
        fp = GlFixPoint(size=5, points=fpcoords, color=[1, 0, 0])

        if name is not None:
            self.name = name
        else:
            self.name = "InterTrial"
        self.actor_list = [c, fp]

        self.mov_time = 30
        self.stat_time = 10

    def reset(self):
        super(InterTrial, self).reset()

    def display_gl(self, modelview=None, projection=None):
        super(InterTrial, self).display_gl(modelview=modelview, projection=projection)

        glClearColor(0.5, 0.5, 0.5, 1.0)

        for actor in self.actor_list:

                actor.draw_gl(modelview, projection)



class StaticCloud(Trial):
    def __init__(self, dur=None, name=None, cloud=None, particlesize=None, fpcoords=None):
        super(StaticCloud, self).__init__(dur, name=name)
        if cloud is None:
            cloud = gen_poslist(1, size=[100, 1, 100])
        if particlesize is None:
            particlesize = 1000
        if fpcoords is None:
            fpcoords = np.array([0, user_height, -24, 0, 0])

        c = GlFadePoint(size=particlesize, points=cloud)
        fp = GlFixPoint(size=5, points=fpcoords, color=[1, 0, 0])

        if name is not None:
            self.name = name
        else:
            self.name = "StaticCloud"
        self.actor_list = [c, fp]

        self.mov_time = 30
        self.stat_time = 5

        self.phase = 0

        self.calc_attention(self.mov_time, self.sr)
        self.fix_index = 0

        self.ms_low = 4
        self.ms_up = 8
        self.roll_ms()
        self.data['motiononset'] = self.ms
        self.data['statictime'] = self.stat_time
        self.data['movtime'] = self.mov_time
        self.data['phase'] = self.phase
        self.data['binary_list'] = self.binary_list
        self.data['count'] = self.count

    def calc_attention(self, time, sr):
        if sr is None:
            self.sr = 90
        else:
            self.sr = sr
        if time is None:
            self.s = 30
        else:
            self.s = time

        self.binary_list = []
        self.count = 0

        for i in range(self.sr * int(self.s / 10)):
            temp = np.random.rand()
            if temp < 0.99:
                temp = 0
            else:
                temp = 1
            self.binary_list.append(temp)

        self.binary_list = np.repeat(self.binary_list, 30)  # repeat each step 30 times, to get ten frames per screen
        for j in self.binary_list:
            if j == 1:
                self.count += 1
        self.count = self.count / 30

    def roll_ms(self):
        self.ms = np.random.uniform(self.ms_low, self.ms_up)

    def reset(self):
        super(StaticCloud, self).reset()
        self.fix_index = 0
        self.phase = 0
        self.roll_ms()
        self.calc_attention(self.mov_time, self.sr)
        self.data['motiononset'] = self.ms
        self.data['statictime'] = self.stat_time
        self.data['movtime'] = self.mov_time
        self.data['binary_list'] = self.binary_list
        self.data['count'] = self.count
        self.data['phase'] = self.phase

    def display_gl(self, modelview=None, projection=None):
        super(StaticCloud, self).display_gl(modelview=modelview, projection=projection)

        glClearColor(0.5, 0.5, 0.5, 1.0)

        for actor in self.actor_list:

            if actor == self.actor_list[0]:
                actor.draw_gl(modelview, projection)

                if self.timer.get_time() >= self.mov_time + self.ms + self.stat_time:
                    self.end()

            if actor == self.actor_list[1]:
                actor.draw_gl(modelview, projection)
                if self.timer.get_time() <= self.ms:
                    actor.color = [0.8, 0.8, 0.8]
                if self.timer.get_time() > self.ms:
                    if self.fix_index <= len(self.binary_list) - 1:
                        if self.binary_list[self.fix_index - 1] == 0:
                            actor.color = [0.8, 0.8, 0.8]
                            self.fix_index += 1
                        if self.binary_list[self.fix_index - 1] == 1:
                            actor.color = [0, 0, 0]
                            self.fix_index += 1
                    if self.fix_index > len(self.binary_list) - 1:
                        actor.color = [0.8, 0.8, 0.8]


class SinusoidalMotion(Trial):
    def __init__(self, dur=None, name=None, cloud=None, particlesize=None, fpcoords=None, freq=None):
        super(SinusoidalMotion, self).__init__(dur, name)
        if cloud is None:
            cloud = gen_poslist(1, size=[100, 1, 100])
        if particlesize is None:
            particlesize = 1000
        if fpcoords is None:
            fpcoords = np.array([0, user_height, -24, 0, 0])
        if freq is None:
            freq = 0.2
        self.name = "SinusoidalMotion"
        self.timer = CustTimer()
        self.ms_up = 8
        self.ms_low = 4
        self.mov_time = 30
        self.stat_time = 5
        self.roll_ms()
        self.freq = freq
        self.amp = np.array([0.0, 0.0, 0.01]) / self.freq  # /freq for velocity matching
        self.phase_up = 2   # in multiples of pi
        self.phase_low = 0
        self.roll_phase()
        self.calc_attention(self.mov_time, 90)
        self.fix_index = 0

        c = GlFadePoint(size=particlesize, points=cloud)
        fp = GlFixPoint(size=5, points=fpcoords, color=[1, 0, 0])
        self.actor_list = [c, fp]



        self.data['freq'] = self.freq
        self.data['motiononset'] = self.ms
        self.data['phase'] = self.phase
        self.data['amplitude'] = self.amp
        self.data['binary_list'] = self.binary_list
        self.data['count'] = self.count
        self.data['statictime'] = self.stat_time
        self.data['movtime'] = self.mov_time

    def roll_ms(self):
        self.ms = np.random.uniform(self.ms_low, self.ms_up)

    def roll_phase(self):
        self.phase = np.random.uniform(self.phase_low, self.phase_up)*np.pi

    def calc_attention(self, time=None, sr=None):
        if sr is None:
            self.sr = 90
        else:
            self.sr = sr
        if time is None:
            self.s = 30
        else:
            self.s = time

        self.binary_list = []
        self.count = 0

        for i in range(self.sr * int(self.s / 10)):
            temp = np.random.rand()
            if temp < 0.99:
                temp = 0
            else:
                temp = 1
            self.binary_list.append(temp)

        self.binary_list = np.repeat(self.binary_list, 30)  # repeat each step 30 times, to get ten frames per screen
        for j in self.binary_list:
            if j == 1:
                self.count += 1
        self.count = self.count / 30

    def reset(self):
        super(SinusoidalMotion, self).reset()
        self.timer.reset()
        self.roll_ms()
        self.roll_phase()
        self.calc_attention(self.mov_time, self.sr)
        self.fix_index = 0
        self.data['freq'] = self.freq
        self.data['motiononset'] = self.ms
        self.data['statictime'] = self.stat_time
        self.data['movtime'] = self.mov_time
        self.data['phase'] = self.phase
        self.data['amplitude'] = self.amp
        self.data['binary_list'] = self.binary_list
        self.data['count'] = self.count

    def display_gl(self, modelview=None, projection=None):
        super(SinusoidalMotion, self).display_gl(modelview=modelview, projection=projection)
        glClearColor(0.5, 0.5, 0.5, 1.0)
        for actor in self.actor_list:

            if actor == self.actor_list[0]:
                actor.ori_pos = self.amp * np.sin(2 * np.pi * self.freq * self.ms + self.phase)
                actor.draw_gl(modelview, projection)
                if self.timer.get_time() < self.ms:
                    actor.move(actor.ori_pos)
                if self.timer.get_time() >= self.ms:
                    actor.move(self.amp *
                               np.sin(2 * np.pi * self.freq * (self.timer.get_time()) + self.phase))
                if self.timer.get_time() >= self.ms + self.mov_time:
                    actor.move(actor.ori_pos)
                if self.timer.get_time() > self.ms + self.mov_time + self.stat_time:
                    self.end()

            if actor == self.actor_list[1]:
                actor.draw_gl(modelview, projection)
                if self.timer.get_time() <= self.ms:
                    actor.color = [0.8, 0.8, 0.8]
                if self.timer.get_time() > self.ms:
                    if self.fix_index <= len(self.binary_list) - 1:
                        if self.binary_list[self.fix_index - 1] == 0:
                            actor.color = [0.8, 0.8, 0.8]
                            self.fix_index += 1
                        if self.binary_list[self.fix_index - 1] == 1:
                            actor.color = [0, 0, 0]
                            self.fix_index += 1
                    if self.fix_index > len(self.binary_list) - 1:
                        actor.color = [0.8, 0.8, 0.8]


class SumOfSines(Trial):
    def __init__(self, dur=None, name=None, cloud=None, particlesize=None, fpcoords=None):
        super(SumOfSines, self).__init__(dur, name)
        if cloud is None:
            cloud = gen_poslist(1, size=[100, 1, 100])
        if particlesize is None:
            particlesize = 1000
        if fpcoords is None:
            fpcoords = np.array([0, user_height, -24, 0, 0])

        c = GlFadePoint(size=particlesize, points=cloud)
        fp = GlFixPoint(size=5, points=fpcoords, color=[1, 0, 0])
        self.name = "SumOfSines"
        self.actor_list = [c, fp]
        self.timer = CustTimer()
        self.ms_up = 8
        self.ms_low = 4
        self.mov_time = 30
        self.stat_time = 5
        self.roll_ms()
        self.freqs = [0.2, 0.8, 1.2]
        self.phase_up = 2
        self.phase_low = 0
        self.calc_amps()
        self.roll_phases()
        self.calc_attention(self.mov_time, self.sr)
        self.fix_index = 0

        self.data['freqs'] = self.freqs
        self.data['motiononset'] = self.ms
        self.data['phases'] = self.phases
        self.data['amplitudes'] = self.amps
        self.data['binary_list'] = self.binary_list
        self.data['count'] = self.count
        self.data['statictime'] = self.stat_time
        self.data['movtime'] = self.mov_time

    def calc_amps(self):
        amps = []
        for i in self.freqs:
            amps.append(0.01/i)
        self.amps = amps

    def roll_phases(self):
        phases = []
        for i in range(len(self.freqs)):
            phases.append(np.random.uniform(self.phase_low, self.phase_up)*np.pi)
        self.phases = phases

    def roll_ms(self):
        self.ms = np.random.uniform(self.ms_low, self.ms_up)

    def calc_attention(self, time, sr):
        if sr is None:
            self.sr = 90
        else:
            self.sr = sr
        if time is None:
            self.s = 30
        else:
            self.s = time

        self.binary_list = []
        self.count = 0

        for i in range(self.sr * int(self.s / 10)):
            temp = np.random.rand()
            if temp < 0.99:
                temp = 0
            else:
                temp = 1
            self.binary_list.append(temp)

        self.binary_list = np.repeat(self.binary_list,
                                     30)  # repeat each step 30 times, to get ten frames per screen
        for j in self.binary_list:
            if j == 1:
                self.count += 1
        self.count = self.count / 30

    def reset(self):
        super(SumOfSines, self).reset()
        self.timer.reset()
        self.roll_ms()
        self.roll_phases()
        self.calc_attention(self.mov_time, self.sr)
        self.fix_index = 0
        self.data['freqs'] = self.freqs
        self.data['motiononset'] = self.ms
        self.data['statictime'] = self.stat_time
        self.data['movtime'] = self.mov_time
        self.data['phases'] = self.phases
        self.data['amplitudes'] = self.amps
        self.data['binary_list'] = self.binary_list
        self.data['count'] = self.count

    def display_gl(self, modelview=None, projection=None):
        super(SumOfSines, self).display_gl(modelview=modelview, projection=projection)
        glClearColor(0.5, 0.5, 0.5, 1.0)
        for actor in self.actor_list:

            if actor == self.actor_list[0]:

                actor.ori_pos = np.array([0, 0, (self.amps[0] * np.sin(2 * np.pi * self.freqs[0] * self.ms + self.phases[0]) +
                                                 self.amps[1] * np.sin(2 * np.pi * self.freqs[1] * self.ms + self.phases[1]) +
                                                 self.amps[2] * np.sin(2 * np.pi * self.freqs[2] * self.ms + self.phases[2]))])

                actor.draw_gl(modelview, projection)

                if self.timer.get_time() < self.ms:
                    actor.move(actor.ori_pos)
                if self.timer.get_time() >= self.ms:
                    actor.move(np.array([0, 0, (self.amps[0] * np.sin(2 * np.pi * self.freqs[0] * self.timer.get_time() + self.phases[0]) +
                                                self.amps[1] * np.sin(2 * np.pi * self.freqs[1] * self.timer.get_time() + self.phases[1]) +
                                                self.amps[2] * np.sin(2 * np.pi * self.freqs[2] * self.timer.get_time() + self.phases[2]))]))
                if self.timer.get_time() >= self.ms + self.mov_time:
                    actor.move(actor.ori_pos)
                if self.timer.get_time() > self.ms + self.mov_time + self.stat_time:
                    self.end()

            if actor == self.actor_list[1]:
                actor.draw_gl(modelview, projection)
                if self.timer.get_time() <= self.ms:
                    actor.color = [0.8, 0.8, 0.8]
                if self.timer.get_time() > self.ms:
                    if self.fix_index <= len(self.binary_list) - 1:
                        if self.binary_list[self.fix_index - 1] == 0:
                            actor.color = [0.8, 0.8, 0.8]
                            self.fix_index += 1
                        if self.binary_list[self.fix_index - 1] == 1:
                            actor.color = [0, 0, 0]
                            self.fix_index += 1
                    if self.fix_index > len(self.binary_list) - 1:
                        actor.color = [0.8, 0.8, 0.8]


class RandomSequence(Trial):
    def __init__(self, dur=None, name=None, cloud=None, particlesize=None, fpcoords=None):
        super(RandomSequence, self).__init__(dur, name=name)
        if cloud is None:
            cloud = gen_poslist(1, size=[100, 1, 100])
        if particlesize is None:
            particlesize = 1000
        if fpcoords is None:
            fpcoords = np.array([0, user_height, -24, 0, 0])

        c = GlFadePoint(size=particlesize, points=cloud)
        fp = GlFixPoint(size=5, points=fpcoords, color=[1, 0, 0])

        if name is not None:
            self.name = name
        else:
            self.name = "RandomSequence"
        self.actor_list = [c, fp]
        
        self.sr = 90  # sampling rate in Hz
        self.mov_time = 30
        self.stat_time = 5

        self.phase = 0

        self.calc_sequence()
        self.index = 0
        self.fix_index = 0
        self.ms_low = 4
        self.ms_up = 8
        self.roll_ms()
        self.calc_attention(self.mov_time, self.sr)
        self.amplitude = 0.8

        self.data['motiononset'] = self.ms
        self.data['sequence'] = self.seq
        self.data['binary_list'] = self.binary_list
        self.data['count'] = self.count
        self.data['statictime'] = self.stat_time
        self.data['movtime'] = self.mov_time
        self.data['phase'] = self.phase
        self.data['amplitude'] = self.amplitude

    def roll_ms(self):
        self.ms = np.random.uniform(self.ms_low, self.ms_up)

    def calc_sequence(self):

        seq = np.array(np.random.rand(1, self.sr*self.mov_time))  # create sequence of length self.sr*n=seconds
        fft = np.fft.rfft(seq)
        fft = fft / abs(fft)  # smoothing the signal
        freqs = np.fft.fftfreq(seq.size, 1/self.sr)
        freqs = freqs[0:int(len(freqs) / 2)]  # only use first postive half of the spectrum
        freqs = np.append(freqs, 45)  # append value of 45 Hz (arbitrary) for suitable length of the array

        fft = fft / freqs  # scale frequencies for flat spectrum in the first derivative (slope otherwise)
        fft[:, 0] = 0  # set first value to zero to avoid sequence to crash (division by zero error)

        #for i in range(int(len(freqs))):
        #    if freqs[i] > 2:  # cut off all frequencies higher than 2 Hz
        #        fft[0, i] = 0.0

        seq = np.fft.irfft(fft)  # re-create sequence out of smoothed and filtered fft
        seq = (seq - np.mean(seq)) / np.std(seq)  # z-score sequence
        self.seq = np.repeat(seq[0, :], 3)   # repeat every step 3 times, since it is rendered 3 times

    def calc_attention(self, time, sr):
        if sr is None:
            self.sr = 90
        else:
            self.sr = sr
        if time is None:
            self.s = 30
        else:
            self.s = time

        self.binary_list = []
        self.count = 0

        for i in range(self.sr * int(self.s / 10)):
            temp = np.random.rand()
            if temp < 0.99:
                temp = 0
            else:
                temp = 1
            self.binary_list.append(temp)

        self.binary_list = np.repeat(self.binary_list, 30)  # repeat each step 30 times, to get ten frames per screen
        for j in self.binary_list:
            if j == 1:
                self.count += 1
        self.count = self.count / 30


    def reset(self):
        super(RandomSequence, self).reset()
        self.index = 0
        self.fix_index = 0
        self.phase = 0
        self.roll_ms()
        self.calc_sequence()
        self.calc_attention(self.mov_time, self.sr)
        self.data['motiononset'] = self.ms
        self.data['statictime'] = self.stat_time
        self.data['movtime'] = self.mov_time
        self.data['sequence'] = self.seq
        self.data['binary_list'] = self.binary_list
        self.data['count'] = self.count
        self.data['phase'] = self.phase
        self.data['amplitude'] = self.amplitude


    def display_gl(self, modelview=None, projection=None):
        super(RandomSequence, self).display_gl(modelview=modelview, projection=projection)

        glClearColor(0.5, 0.5, 0.5, 1.0)
        for actor in self.actor_list:

            if actor == self.actor_list[0]:
                actor.draw_gl(modelview, projection)
                # print(self.index)
                if self.timer.get_time() <= self.ms:
                    actor.move(np.array([0, 0, self.amplitude*self.seq[self.index]]))
                if self.timer.get_time() > self.ms:
                    if self.index <= len(self.seq)-1:
                        actor.move(np.array([0, 0, self.amplitude*self.seq[self.index]]))
                        self.index += 1
                    if self.index > len(self.seq)-1:
                        actor.move(np.array([0, 0, self.amplitude*self.seq[-1]]))
                        self.index += 1
                    if self.index > len(self.seq)+self.stat_time*self.sr*3:
                        self.end()

            if actor == self.actor_list[1]:
                actor.draw_gl(modelview, projection)
                if self.timer.get_time() <= self.ms:
                    actor.color = [0.8, 0.8, 0.8]
                if self.timer.get_time() > self.ms:
                    if self.fix_index <= len(self.binary_list)-1:
                        if self.binary_list[self.fix_index-1] == 0:
                            actor.color = [0.8, 0.8, 0.8]
                            self.fix_index += 1
                        if self.binary_list[self.fix_index-1] == 1:
                            actor.color = [0, 0, 0]
                            self.fix_index += 1
                    if self.fix_index > len(self.binary_list)-1:
                        actor.color = [0.8, 0.8, 0.8]



"""""""""""""""""""""""""""""
OpenGL Objects
"""""""""""""""""""""""""""""


class GlCloud(object):
    def __init__(self, position=None, size=None, motionstart=None, num_dots=None, dot_range=None, frequency=None,
                 amplitude=None, velocity=None, move_mode=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 0.0, 0.0]
        self.ori_pos = np.array(self.position)

        if size is not None:
            self.size = size
        else:
            self.size = 1

        if num_dots is not None:
            self.num_dots = num_dots
        else:
            self.num_dots = 100

        if dot_range is not None:
            self.dot_range = dot_range
        else:
            self.dot_range = np.asarray([[-1, 1], [-1, 1], [-1, 1]])

        if frequency is not None:
            self.frequency = frequency
        else:
            self.frequency = np.asarray([0, 0, 1])

        if amplitude is not None:
            self.amplitude = amplitude
        else:
            self.amplitude = np.asarray([0, 0, 1])

        if velocity is not None:
            self.velocity = velocity
        else:
            self.velocity = np.asarray([0, 0, 1])

        if move_mode is not None:
            self.move_mode = move_mode
        else:
            self.move_mode = 'translation'

        self.cloud_list = pointlist(self.num_dots, self.dot_range)

        self.points = np.array(self.cloud_list, dtype=np.float32)

        self.indices = np.array(list(range(self.num_dots)), dtype=np.uint32)

        self.shader = None
        self.pos_loc = None
        self.col_loc = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.view = None
        self.model = None
        self.projection = None
        self.view_loc = None
        self.proj_loc = None
        self.model_loc = None
        self.transform_loc = None

        if motionstart is not None:
            self.ms = motionstart
        else:
            self.ms = 0
        self.timer = CustTimer()
        self.mt = CustTimer(False)

    def reset(self):
        self.cloud_list = pointlist(self.num_dots, self.dot_range)
        # self.init_gl()
        self.timer.reset()
        self.mt.stop()
        self.points = self.size * np.array(self.cloud_list, dtype=np.float32)

    def move(self, new_pos=None):
        if new_pos is not None:
            self.position = new_pos

    def reinit(self):
        self.position = self.ori_pos

    def init_gl(self):
        print("init cloud")
        # if self.shader is None:
        print("init shader")
        vertex_shader = """
            #version 330
            in vec3 position;
            in vec3 color;
            uniform mat4 modelview;
            uniform mat4 projection;
            uniform mat4 model;
            out vec3 newColor;
            void main()
            {
                gl_Position = projection * modelview * model * vec4(position, 1.0f);
                newColor = color;
            }
            """

        fragment_shader = """
            #version 330
            in vec3 newColor;
            out vec4 outColor;
            void main()
            {
                outColor = vec4(newColor, 1.0f);
            }
            """
        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader,
                                            GL_FRAGMENT_SHADER))
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        self.modelview_loc = glGetUniformLocation(self.shader, "modelview")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.model_loc = glGetUniformLocation(self.shader, "model")

        self.pos_loc = glGetAttribLocation(self.shader, "position")

        glBindVertexArray(self.vao)
        # send vertices to graka

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)
        glBufferData(GL_ARRAY_BUFFER, self.num_dots * 6 * 4, self.points, GL_STATIC_DRAW)

        # send indices to graka

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 144, self.indices, GL_STATIC_DRAW)

        # get position of variable position in shader program

        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.pos_loc)

        self.col_loc = glGetAttribLocation(self.shader, "color")
        glVertexAttribPointer(self.col_loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.col_loc)

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        glBindVertexArray(0)
        print("done init cloud")

    def draw_gl(self, modelview=None, projection=None):

        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, -1.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))

        # TODO: Bewegung in Trialklasse überführen
        # if self.timer.get_time() >= self.ms and not self.mt.running:
        #     self.mt.start()
        #
        # self.move_temp = calc_move(self.move_mode, self.amplitude, self.frequency, self.velocity, self.mt.get_time())
        #
        # self.move_vec = pyrr.Vector3(self.move_temp)
        #
        # self.rot_matr = pyrr.matrix44.create_from_axis_rotation(pyrr.Vector3([1.0, 0.0, 0.0]), 0.5)
        # self.view_matr = pyrr.Vector3([0.0, 0.0, -30.0])
        # self.cam = 'static'
        # # Kamera statisch, Objekt bewegt sich
        # if self.cam == 'static':
        #     self.model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position) + self.move_vec)
        #     self.transform_matr = np.dot(pyrr.matrix44.create_from_translation(-1 * self.view_matr), self.rot_matr)
        #
        # # Objekt statisch, Kamera bewegt sich
        # elif self.cam == 'movable':
        #     self.model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))
        #     self.transform_matr = np.dot(pyrr.matrix44.create_from_translation(-1 * self.view_matr), self.rot_matr)
        #     self.transform_matr = np.dot(self.transform_matr, pyrr.matrix44.create_from_translation(-1 * self.move_vec))
        #
        # self.view = self.transform_matr

        # print('v', self.view)
        # print('m', self.model)
        # print('vxm', self.view @ self.model)

        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        # hier werden view und model noch einmal einzeln aufgerufen
        glUniformMatrix4fv(self.modelview_loc, 1, GL_FALSE, modelview)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        # glEnable(GL_POINT_SMOOTH)
        glPointSize(self.size)

        glDrawArrays(GL_POINTS, 0, self.num_dots)
        glBindVertexArray(0)
        glUseProgram(0)


class GlCube(object):
    def __init__(self, position=None, size=1):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 0.0, 0.0]

        self.ori_pos = self.position
        self.size = size

        self.points = self.size * np.array([-0.5, -0.5, 0.5, 1.0, 0.0, 0.0,
                                            0.5, -0.5, 0.5, 0.0, 1.0, 0.0,
                                            0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                                            -0.5, 0.5, 0.5, 1.0, 1.0, 0.0,
                                            -0.5, -0.5, -0.5, 1.0, 0.0, 1.0,
                                            0.5, -0.5, -0.5, 0.0, 1.0, 1.0,
                                            0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
                                            -0.5, 0.5, -0.5, 0.0, 0.0, 0.0], dtype=np.float32)

        self.indices = np.array([0, 1, 2,
                                 2, 3, 0,
                                 4, 5, 6,
                                 6, 7, 4,
                                 4, 5, 1,
                                 1, 0, 4,
                                 6, 7, 3,
                                 3, 2, 6,
                                 5, 6, 2,
                                 2, 1, 5,
                                 7, 4, 0,
                                 0, 3, 7], dtype=np.uint32)

        self.shader = None
        self.pos_loc = None
        self.col_loc = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.view = None
        self.model = None
        self.projection = None
        self.view_loc = None
        self.proj_loc = None
        self.model_loc = None
        self.transform_loc = None

    def reset(self):
        print(self.position)
        self.position = self.ori_pos
        print(self.position)
        return

    def move(self, new_pos=None):
        if new_pos is not None:
            self.position = new_pos

    def reinit(self):
        self.position = self.ori_pos

    def init_gl(self):
        print("init GLCube...")
        vertex_shader = """
            #version 330
            in vec3 position;
            in vec3 color;            
            uniform mat4 modelview;
            uniform mat4 model;
            uniform mat4 projection;            
            out vec3 newColor;
            void main()
            {
                gl_Position = projection * modelview * model * vec4(position, 1.0f);
                newColor = color;
            }
            """

        fragment_shader = """
            #version 330
            in vec3 newColor;
            out vec4 outColor;
            void main()
            {
                outColor = vec4(newColor, 1.0f);
            }
            """
        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                       GL_FRAGMENT_SHADER))
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        # send vertices to graka
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)
        glBufferData(GL_ARRAY_BUFFER, 192, self.points, GL_STATIC_DRAW)

        # send indices to graka
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 144, self.indices, GL_STATIC_DRAW)

        # get position of variable position in shader program
        self.pos_loc = glGetAttribLocation(self.shader, "position")
        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.pos_loc)

        self.col_loc = glGetAttribLocation(self.shader, "color")
        glVertexAttribPointer(self.col_loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.col_loc)

        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.modelview_loc = glGetUniformLocation(self.shader, "modelview")
        self.model_loc = glGetUniformLocation(self.shader, "model")

        glBindVertexArray(0)
        print("finish init GLCube")

    def draw_gl(self, modelview=None, projection=None):

        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, -30.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))
        glUseProgram(self.shader)

        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.modelview_loc, 1, GL_FALSE, modelview)

        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glUseProgram(0)


class GlPoint(object):

    # perspektiv point

    def __init__(self, position=None, size=None, points=None, color=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 0.0, 0.0]
        self.ori_pos = self.position
        if size is not None:
            self.size = size
        else:
            self.size = 1000.0
        if color is not None:
            self.color = np.array(color, dtype=np.float32)
        else:
            self.color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.shader = None
        self.vao = None
        self.timer = CustTimer()

        if points is not None:
            self.points = np.array(points, dtype=np.float32)
        else:
            self.points = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self):
        self.timer.reset()
        self.position = self.ori_pos

    def move(self, new_pos=None):
        if new_pos is not None:
            self.position = new_pos

    def init_gl(self):

        glEnable(GL_PROGRAM_POINT_SIZE)
        # glEnable(GL_POINT_SPRITE)

        vertex_shader = """
            #version 330
            uniform mat4 view;
            uniform mat4 projection;
            uniform mat4 model;
            uniform float size;
            uniform float time;
            uniform vec3 color;
            in float lifetime;
            in float lifetimephase;
            in vec3 position;
            out vec4 newcolor;
            out float alive;
            const float PI = 3.1415926;
            void main()
            {
                gl_Position = projection * view * model * vec4(position, 1.0f);
                vec3 ndc = gl_Position.xyz / gl_Position.w;
                float zDist = 1.0-ndc.z;
                gl_PointSize = size*zDist;
                newcolor = vec4(color, 1.0f);
                if (sin( 2*PI * (lifetimephase + lifetime * time)) < 0){
                    alive = -1.0;
                }
                else {
                    alive = 1.0;
                }

            }

        """

        fragment_shader = """
            #version 330
            in vec4 newcolor;
            in float alive;
            out vec4 outColor;
            void main()
            {
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;                
                float r = dot(cxy, cxy);
                if (r > 0.5) {
                    discard;
                }
                if (alive < 0){
                    discard;
                }

                outColor = newcolor;

            }
        """
        print('shader done')
        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                       GL_FRAGMENT_SHADER))
        print("shader compiled")
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # send vertices to graka
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)

        glBufferData(GL_ARRAY_BUFFER, len(self.points) * 4, self.points, GL_STATIC_DRAW)

        self.pos_loc = glGetAttribLocation(self.shader, "position")
        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.pos_loc)

        self.lt_loc = glGetAttribLocation(self.shader, "lifetime")
        self.ltphase_loc = glGetAttribLocation(self.shader, "lifetimephase")

        glVertexAttribPointer(self.lt_loc, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.lt_loc)

        glVertexAttribPointer(self.ltphase_loc, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(16))
        glEnableVertexAttribArray(self.ltphase_loc)

        self.col_loc = glGetAttribLocation(self.shader, "color")

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)
        self.col_loc = glGetUniformLocation(self.shader, "color")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.size_loc = glGetUniformLocation(self.shader, "size")
        self.time_loc = glGetUniformLocation(self.shader, "time")

    def draw_gl(self, modelview=None, projection=None):

        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, -30.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))
        glUseProgram(self.shader)

        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, modelview)
        glUniform3fv(self.col_loc, 1, self.color)
        glUniform1f(self.size_loc, self.size)
        glUniform1f(self.time_loc, self.timer.get_time())
        glDrawArrays(GL_POINTS, 0, int(len(self.points) / 3))
        glBindVertexArray(0)
        glUseProgram(0)


class GlFixPoint(object):

    # perspektiv point

    def __init__(self, position=None, size=None, points=None, color=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 0.0, 0.0]
        self.ori_pos = self.position
        if size is not None:
            self.size = size
        else:
            self.size = 1000.0
        if color is not None:
            self.color = np.array(color, dtype=np.float32)
        else:
            self.color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.shader = None
        self.vao = None
        self.timer = CustTimer()

        if points is not None:
            self.points = np.array(points, dtype=np.float32)
        else:
            self.points = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self):
        self.timer.reset()
        self.position = self.ori_pos

    def move(self, new_pos=None):
        if new_pos is not None:
            self.position = new_pos

    def init_gl(self):

        glEnable(GL_PROGRAM_POINT_SIZE)
        # glEnable(GL_POINT_SPRITE)

        vertex_shader = """
            #version 330
            uniform mat4 view;
            uniform mat4 projection;
            uniform mat4 model;
            uniform float size;
            uniform float time;
            uniform vec3 color;
            in float lifetime;
            in float lifetimephase;
            in vec3 position;
            out vec4 newcolor;
            out float alive;
            const float PI = 3.1415926;
            void main()
            {
                gl_Position = projection * view * model * vec4(position, 1.0f);
                vec3 ndc = gl_Position.xyz / gl_Position.w;
                float zDist = 1.0-ndc.z;
                gl_PointSize = size;
                newcolor = vec4(color, 1.0f);
                if (sin( 2*PI * (lifetimephase + lifetime * time)) < 0){
                    alive = -1.0;
                }
                else {
                    alive = 1.0;
                }

            }

        """

        fragment_shader = """
            #version 330
            in vec4 newcolor;
            in float alive;
            out vec4 outColor;
            void main()
            {
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;                
                float r = dot(cxy, cxy);
                if (r > 0.5) {
                    discard;
                }
                if (alive < 0){
                    discard;
                }

                outColor = newcolor;

            }
        """
        print('shader done')
        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                       GL_FRAGMENT_SHADER))
        print("shader compiled")
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # send vertices to graka
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)

        glBufferData(GL_ARRAY_BUFFER, len(self.points) * 4, self.points, GL_STATIC_DRAW)

        self.pos_loc = glGetAttribLocation(self.shader, "position")
        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.pos_loc)

        self.lt_loc = glGetAttribLocation(self.shader, "lifetime")
        self.ltphase_loc = glGetAttribLocation(self.shader, "lifetimephase")

        glVertexAttribPointer(self.lt_loc, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.lt_loc)

        glVertexAttribPointer(self.ltphase_loc, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(16))
        glEnableVertexAttribArray(self.ltphase_loc)

        self.col_loc = glGetAttribLocation(self.shader, "color")

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)
        self.col_loc = glGetUniformLocation(self.shader, "color")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.size_loc = glGetUniformLocation(self.shader, "size")
        self.time_loc = glGetUniformLocation(self.shader, "time")

    def draw_gl(self, modelview=None, projection=None):

        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, -30.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))
        glUseProgram(self.shader)

        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, modelview)
        glUniform3fv(self.col_loc, 1, self.color)
        glUniform1f(self.size_loc, self.size)
        glUniform1f(self.time_loc, self.timer.get_time())
        glDrawArrays(GL_POINTS, 0, int(len(self.points) / 3))
        glBindVertexArray(0)
        glUseProgram(0)


class GlFadePoint(object):

    # perspektiv point

    def __init__(self, position=None, size=None, points=None, color=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 0.0, 0.0]
        self.ori_pos = self.position
        if size is not None:
            self.size = size
        else:
            self.size = 1000.0
        if color is not None:
            self.color = np.array(color, dtype=np.float32)
        else:
            self.color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.shader = None
        self.vao = None
        self.timer = CustTimer()

        if points is not None:
            self.points = np.array(points, dtype=np.float32)
        else:
            self.points = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self):
        self.timer.reset()
        self.position = self.ori_pos

    def move(self, new_pos=None):
        if new_pos is not None:
            self.position = new_pos

    def init_gl(self):

        glEnable(GL_PROGRAM_POINT_SIZE)
        # glEnable(GL_POINT_SPRITE)

        vertex_shader = """
            #version 330
            uniform mat4 view;
            uniform mat4 projection;
            uniform mat4 model;
            uniform float size;
            uniform float time;
            uniform vec3 color;
            in float lifetime;
            in float lifetimephase;
            in vec3 position;
            out vec4 newcolor;
            out float alive;
            const float PI = 3.1415926;
            void main()
            {
                gl_Position = projection * view * model * vec4(position, 1.0f);
                vec3 ndc = gl_Position.xyz / gl_Position.w;
                float zDist = 1.0-ndc.z;
                gl_PointSize = size*zDist;
                newcolor = vec4(max(0,50*(-0.99 + ndc.z)), max(0,50*(-0.99 + ndc.z)), max(0,50*(-0.99 + ndc.z)), 1.0f);
                if (sin( 2*PI * (lifetimephase + lifetime * time)) < 0){
                    alive = -1.0;
                }
                else {
                    alive = 1.0;
                }

            }

        """

        fragment_shader = """
            #version 330
            in vec4 newcolor;
            in float alive;
            out vec4 outColor;
            void main()
            {
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;                
                float r = dot(cxy, cxy);
                if (r > 0.5) {
                    discard;
                }
                if (alive < 0){
                    discard;
                }

                outColor = newcolor;

            }
        """
        print('shader done')
        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                       GL_FRAGMENT_SHADER))
        print("shader compiled")
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # send vertices to graka
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)

        glBufferData(GL_ARRAY_BUFFER, len(self.points) * 4, self.points, GL_STATIC_DRAW)

        self.pos_loc = glGetAttribLocation(self.shader, "position")
        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.pos_loc)

        self.lt_loc = glGetAttribLocation(self.shader, "lifetime")
        self.ltphase_loc = glGetAttribLocation(self.shader, "lifetimephase")

        glVertexAttribPointer(self.lt_loc, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.lt_loc)

        glVertexAttribPointer(self.ltphase_loc, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(16))
        glEnableVertexAttribArray(self.ltphase_loc)

        self.col_loc = glGetAttribLocation(self.shader, "color")

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)
        self.col_loc = glGetUniformLocation(self.shader, "color")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.size_loc = glGetUniformLocation(self.shader, "size")
        self.time_loc = glGetUniformLocation(self.shader, "time")

    def draw_gl(self, modelview=None, projection=None):

        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, -30.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))
        glUseProgram(self.shader)

        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, modelview)
        glUniform3fv(self.col_loc, 1, self.color)
        glUniform1f(self.size_loc, self.size)
        glUniform1f(self.time_loc, self.timer.get_time())
        glDrawArrays(GL_POINTS, 0, int(len(self.points) / 3))
        glBindVertexArray(0)
        glUseProgram(0)


class GlPointStatic(object):
    def __init__(self, position=None, size=None, points=None, color=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 0.0, 0.0]
        self.ori_pos = self.position
        if size is not None:
            self.size = size
        else:
            self.size = 1000.0
        if color is not None:
            self.color = np.array(color, dtype=np.float32)
        else:
            self.color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.shader = None
        self.vao = None
        self.timer = CustTimer()

        if points is not None:
            self.points = np.array(points, dtype=np.float32)
        else:
            self.points = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self):
        self.timer.reset()
        self.position = self.ori_pos

    def move(self, new_pos=None):
        if new_pos is not None:
            self.position = new_pos

    def init_gl(self):

        glEnable(GL_PROGRAM_POINT_SIZE)
        # glEnable(GL_POINT_SPRITE)

        vertex_shader = """
            #version 330
            uniform mat4 view;
            uniform mat4 projection;
            uniform mat4 model;
            uniform float size;
            uniform vec3 color;
            in vec3 position;            
            out vec4 newcolor;
            void main()
            {
                gl_Position = projection * view * model * vec4(position, 1.0f);
                vec3 ndc = gl_Position.xyz / gl_Position.w;
                float zDist = 1.0-ndc.z;
                gl_PointSize = size*zDist;
                newcolor = vec4(color, 1.0f);
            }

        """

        fragment_shader = """
            #version 330
            in vec4 newcolor;
            out vec4 outColor;
            void main()
            {
                vec2 cxy = 2.0 * gl_PointCoord -1.0;                
                float r = dot(cxy, cxy);
                if (r > 0.5) {
                    discard;
                } 
                outColor = newcolor;
            }
        """
        print('shader done')
        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                       GL_FRAGMENT_SHADER))
        print("shader compiled")
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # send vertices to graka
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)

        glBufferData(GL_ARRAY_BUFFER, len(self.points) * 4, self.points, GL_STATIC_DRAW)

        self.pos_loc = glGetAttribLocation(self.shader, "position")
        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.pos_loc)

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)
        self.col_loc = glGetUniformLocation(self.shader, "color")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.size_loc = glGetUniformLocation(self.shader, "size")

    def draw_gl(self, modelview=None, projection=None):

        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, -1.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))
        glUseProgram(self.shader)

        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, modelview)
        glUniform3fv(self.col_loc, 1, self.color)
        glUniform1f(self.size_loc, self.size)

        glDrawArrays(GL_POINTS, 0, int(len(self.points) / 3))
        glBindVertexArray(0)
        glUseProgram(0)


class GlPointZylinder(object):
    def __init__(self, position=None, radius=None, height=None, density=None, size=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 0.0, 0.0]

        if radius is not None:
            self.radius = radius
        else:
            self.radius = 1

        if height is not None:
            self.height = height
        else:
            self.height = 1

        if density is not None:
            self.density = density
        else:
            self.density = 0.1

        if size is not None:
            self.size = size
        else:
            self.size = 1

        self.num_points = np.maximum(1, int(self.radius ** 2 * self.height * self.density))
        print(self.num_points)
        self.pos_list = self.gen_poslist()

    def gen_poslist(self):

        # get point positions
        angle = 2 * np.pi * np.random.rand(1, self.num_points)
        x_pos = np.sin(angle) * self.radius - self.position[0]
        z_pos = np.cos(angle) * self.radius - self.position[2]
        print(x_pos ** 2 + z_pos ** 2)
        y_pos = (np.random.rand(1, self.num_points) - 0.5) * self.height - self.position[1]

        # get colors
        c1 = np.ones(self.num_points)
        c2 = np.ones(self.num_points)
        c3 = np.ones(self.num_points)

        pos_list = np.ones(6 * self.num_points)
        pos_list[0::6] = x_pos
        pos_list[1::6] = y_pos
        pos_list[2::6] = z_pos
        pos_list[3::6] = c1
        pos_list[4::6] = c2
        pos_list[5::6] = c3
        print(pos_list)
        return np.array(pos_list, dtype=np.float32)

    def init_gl(self):
        print("init zylinder")

        vertex_shader = """
            #version 330
            in vec3 position;
            in vec3 color;
            uniform mat4 modelview;
            uniform mat4 projection;
            uniform mat4 model;
            uniform mat4 transform;
            out vec3 newColor;
            void main()
            {
                gl_Position = projection * modelview * model * transform * vec4(position, 1.0f);
                newColor = color;
            }
            """

        fragment_shader = """
            #version 330
            in vec3 newColor;
            out vec4 outColor;
            void main()
            {
                outColor = vec4(newColor, 1.0f);
            }
            """
        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader,
                                            GL_FRAGMENT_SHADER))
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        self.modelview_loc = glGetUniformLocation(self.shader, "modelview")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.trans_loc = glGetUniformLocation(self.shader, "transform")
        self.pos_loc = glGetAttribLocation(self.shader, "position")

        glBindVertexArray(self.vao)
        # send vertices to graka

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)
        glBufferData(GL_ARRAY_BUFFER, self.num_points * 6 * 4, self.pos_list, GL_STATIC_DRAW)

        # get position of variable position in shader program

        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.pos_loc)

        self.col_loc = glGetAttribLocation(self.shader, "color")
        glVertexAttribPointer(self.col_loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.col_loc)

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        glBindVertexArray(0)
        print("done init zylinder")

    def draw_gl(self, modelview=None, projection=None, model=None, transform=None):
        print("draw")
        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, 0.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        if model is None:
            model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))
        else:
            model = model @ pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))

        if transform is None:
            transform = pyrr.matrix44.create_identity()

        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.modelview_loc, 1, GL_FALSE, modelview)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.trans_loc, 1, GL_FALSE, transform)
        # glEnable(GL_POINT_SMOOTH)
        glPointSize(self.size)

        glDrawArrays(GL_POINTS, 0, self.num_points)
        glBindVertexArray(0)
        glUseProgram(0)


class GlObjectStructure(object):
    def __init__(self, position=None, size=None, particlegeo=None, structure=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.1, 0.0, 0.0]

        if size is not None:
            self.size = size
        else:
            self.size = 1

        if particlegeo is not None:
            self.pgeo = particlegeo
        else:
            self.pqeo = "cube"

        if structure is not None:
            self.instance_array = structure
        else:
            self.instance_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.ori_pos = self.position

        if True:
            self.vert = self.size * np.array([-0.5, -0.5, 0.5, 1.0, 1.0, 1.0,
                                              0.5, -0.5, 0.5, 1.0, 1.0, 1.0,
                                              0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
                                              -0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
                                              -0.5, -0.5, -0.5, 1.0, 1.0, 1.0,
                                              0.5, -0.5, -0.5, 1.0, 1.0, 1.0,
                                              0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
                                              -0.5, 0.5, -0.5, 1.0, 1.0, 1.0], dtype=np.float32)

            self.indices = np.array([0, 1, 2,
                                     2, 3, 0,
                                     4, 5, 6,
                                     6, 7, 4,
                                     4, 5, 1,
                                     1, 0, 4,
                                     6, 7, 3,
                                     3, 2, 6,
                                     5, 6, 2,
                                     2, 1, 5,
                                     7, 4, 0,
                                     0, 3, 7], dtype=np.uint32)

        self.shader = None
        self.pos_loc = None
        self.col_loc = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.view = None
        self.model = None
        self.projection = None
        self.view_loc = None
        self.proj_loc = None
        self.model_loc = None
        self.transform_loc = None
        self.starttime = time.time()

    def reset(self):

        self.position = self.ori_pos

        return

    def move(self, new_pos=None):
        if new_pos is not None:
            self.position = new_pos

    def reinit(self):
        self.position = self.ori_pos

    def init_gl(self):
        print("init object structure")
        vertex_shader = """
            #version 330
            in vec3 position;
            in vec3 color;  
            in vec3 offset;            
            in float lifetime;
            in float lifetimephase;

            uniform float time;          
            uniform mat4 modelview;
            uniform mat4 model;
            uniform mat4 projection;            
            out vec3 newColor;
            const float PI = 3.1415926;
            void main()
            {
                vec3 final_pos = vec3(position.x + offset.x, position.y + offset.y, position.z + offset.z );
                gl_Position = projection * modelview * model * vec4(final_pos, 1.0f);
                if (sin( 2*PI * (lifetimephase + time + lifetime)) <= 2){
                     newColor = color;
                }
                else {
                    newColor = vec3(1.0, 1.0, 1.0);
                }           

            }
            """

        fragment_shader = """
            #version 330
            in vec3 newColor;
            out vec4 outColor;
            void main()
            {
                outColor = vec4(newColor, 1.0f);
            }
            """
        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                       GL_FRAGMENT_SHADER))
        print("shader done")
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        # send vertices to graka
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)
        glBufferData(GL_ARRAY_BUFFER, len(self.vert) * self.vert.itemsize, self.vert, GL_STATIC_DRAW)

        # send indices to graka
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(self.indices) * self.vert.itemsize, self.indices, GL_STATIC_DRAW)

        # get position of variable position in shader program
        self.pos_loc = glGetAttribLocation(self.shader, "position")
        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(self.pos_loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.pos_loc)

        self.col_loc = glGetAttribLocation(self.shader, "color")
        glVertexAttribPointer(self.col_loc, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.col_loc)

        self.offset_loc = glGetAttribLocation(self.shader, "offset")
        self.lt_loc = glGetAttribLocation(self.shader, "lifetime")
        self.ltphase_loc = glGetAttribLocation(self.shader, "lifetimephase")
        self.ivbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.ivbo)
        glBufferData(GL_ARRAY_BUFFER, self.instance_array.itemsize * len(self.instance_array), self.instance_array,
                     GL_STATIC_DRAW)
        self.ins_len = int(len(self.instance_array) / 5)

        glVertexAttribPointer(self.offset_loc, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.offset_loc)

        glVertexAttribDivisor(self.offset_loc, 1)

        glVertexAttribPointer(self.lt_loc, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(self.lt_loc)

        glVertexAttribDivisor(self.lt_loc, 1)

        glVertexAttribPointer(self.ltphase_loc, 1, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(16))
        glEnableVertexAttribArray(self.ltphase_loc)

        glVertexAttribDivisor(self.ltphase_loc, 1)

        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.modelview_loc = glGetUniformLocation(self.shader, "modelview")
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.t_loc = glGetAttribLocation(self.shader, "time")
        glBindVertexArray(0)
        print("finish init object structure")

    def draw_gl(self, modelview=None, projection=None, model=None, transform=None):

        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, -30.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        if model is None:
            model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))

        glUseProgram(self.shader)

        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(self.modelview_loc, 1, GL_FALSE, modelview)

        glUniform1f(self.t_loc, time.time() - self.starttime)

        glDrawElementsInstanced(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None, self.ins_len)
        glBindVertexArray(0)
        glUseProgram(0)


"""""""""""""""""""""""""""""
OpenVR Objects
"""""""""""""""""""""""""""""


class OpenVrFramebuffer(object):
    "Framebuffer for rendering one eye"

    def __init__(self, width, height, multisample=0):
        self.fb = 0
        self.depth_buffer = 0
        self.texture_id = 0
        self.width = width
        self.height = height
        self.compositor = None
        self.multisample = multisample

    def init_gl(self):
        # Set up framebuffer and render textures
        self.fb = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb)
        self.depth_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer)
        if self.multisample > 0:
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, self.multisample, GL_DEPTH24_STENCIL8, self.width,
                                             self.height)
        else:
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_buffer)
        self.texture_id = int(glGenTextures(1))
        if self.multisample > 0:
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, self.texture_id)
            glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, self.multisample, GL_RGBA8, self.width, self.height,
                                    True)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, self.texture_id, 0)
        else:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture_id, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            raise Exception("Incomplete framebuffer")
        # Resolver framebuffer in case of multisample antialiasing
        if self.multisample > 0:
            self.resolve_fb = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.resolve_fb)
            self.resolve_texture_id = int(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D, self.resolve_texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.resolve_texture_id, 0)
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                raise Exception("Incomplete framebuffer")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        # OpenVR texture data
        self.texture = openvr.Texture_t()
        if self.multisample > 0:
            self.texture.handle = self.resolve_texture_id
        else:
            self.texture.handle = self.texture_id
        self.texture.eType = openvr.TextureType_OpenGL
        self.texture.eColorSpace = openvr.ColorSpace_Gamma

    def submit(self, eye):
        if self.multisample > 0:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fb)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.resolve_fb)
            glBlitFramebuffer(0, 0, self.width, self.height,
                              0, 0, self.width, self.height,
                              GL_COLOR_BUFFER_BIT, GL_LINEAR)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        openvr.VRCompositor().submit(eye, self.texture)

    def dispose_gl(self):
        glDeleteTextures([self.texture_id])
        glDeleteRenderbuffers(1, [self.depth_buffer])
        glDeleteFramebuffers(1, [self.fb])
        self.fb = 0
        if self.multisample > 0:
            glDeleteTextures([self.resolve_texture_id])
            glDeleteFramebuffers(1, [self.resolve_fb])


class MyOpenVrGlRenderer(object):
    "Renders to virtual reality headset using OpenVR and OpenGL APIs"

    def __init__(self, actor=None, window_size=(800, 600), multisample=0):

        self.vr_system = None
        self.left_fb = None
        self.right_fb = None
        self.curtrial = None
        self.window_size = window_size
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount

        self.poses = poses_t()

        self.multisample = multisample

    def init_gl(self):
        "allocate OpenGL resources"
        print("initiating renderer")
        self.vr_system = openvr.init(openvr.VRApplication_Scene)

        w, h = self.vr_system.getRecommendedRenderTargetSize()
        self.left_fb = OpenVrFramebuffer(w, h, multisample=self.multisample)
        self.right_fb = OpenVrFramebuffer(w, h, multisample=self.multisample)
        self.compositor = openvr.VRCompositor()
        if self.compositor is None:
            raise Exception("Unable to create compositor")
        self.left_fb.init_gl()
        self.right_fb.init_gl()
        # Compute projection matrix
        zNear = 0.2
        zFar = 25.0
        self.projection_left = np.asarray(matrixForOpenVrMatrix(self.vr_system.getProjectionMatrix(
            openvr.Eye_Left,
            zNear, zFar)))
        self.projection_right = np.asarray(matrixForOpenVrMatrix(self.vr_system.getProjectionMatrix(
            openvr.Eye_Right,
            zNear, zFar)))
        self.view_left = matrixForOpenVrMatrix(
            self.vr_system.getEyeToHeadTransform(openvr.Eye_Left)).I  # head_X_eye in Kane notation
        self.view_right = matrixForOpenVrMatrix(
            self.vr_system.getEyeToHeadTransform(openvr.Eye_Right)).I  # head_X_eye in Kane notation
        print("finished initiating renderer")
        return self.vr_system

    def render_scene(self):
        if self.compositor is None:
            return
        self.compositor.waitGetPoses(self.poses, openvr.k_unMaxTrackedDeviceCount, None, 0)
        hmd_pose0 = self.poses[openvr.k_unTrackedDeviceIndex_Hmd]
        if not hmd_pose0.bPoseIsValid:
            return
        hmd_pose1 = hmd_pose0.mDeviceToAbsoluteTracking  # head_X_room in Kane notation
        hmd_pose = matrixForOpenVrMatrix(hmd_pose1).I  # room_X_head in Kane notation
        # Use the pose to compute things
        modelview = hmd_pose
        mvl = modelview * self.view_left  # room_X_eye(left) in Kane notation
        mvr = modelview * self.view_right  # room_X_eye(right) in Kane notation
        # Repack the resulting matrices to have default stride, to avoid
        # problems with weird strides and OpenGL
        mvl = np.asarray(np.matrix(mvl, dtype=np.float32))
        mvr = np.asarray(np.matrix(mvr, dtype=np.float32))
        # 1) On-screen render:

        glViewport(0, 0, self.window_size[0], self.window_size[1])
        # Display left eye view to screen
        self.display_gl(mvl, self.projection_left)
        # 2) VR render
        # Left eye view
        glBindFramebuffer(GL_FRAMEBUFFER, self.left_fb.fb)
        glViewport(0, 0, self.left_fb.width, self.left_fb.height)
        self.display_gl(mvl, self.projection_left)
        self.left_fb.submit(openvr.Eye_Left)
        # self.compositor.submit(openvr.Eye_Left, self.left_fb.texture)
        # Right eye view
        glBindFramebuffer(GL_FRAMEBUFFER, self.right_fb.fb)
        self.display_gl(mvr, self.projection_right)
        self.right_fb.submit(openvr.Eye_Right)
        # self.compositor.submit(openvr.Eye_Right, self.right_fb.texture)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def display_gl(self, modelview, projection):
        # glClearColor(0.0, 0.0, 0.0, 0.0)  # black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.curtrial.display_gl(modelview, projection)

    def dispose_gl(self):

        if self.vr_system is not None:
            openvr.shutdown()
            self.vr_system = None
        if self.left_fb is not None:
            self.left_fb.dispose_gl()
            self.right_fb.dispose_gl()


"""""""""""""""""""""""""""
Initiate & Execute Program
"""""""""""""""""""""""""""


if __name__ == "__main__":

    from openvr.gl_renderer import OpenVrGlRenderer
    from openvr.color_cube_actor import ColorCubeActor

    renderer = MyOpenVrGlRenderer()

    density = 50
    room_size = [5, 5, 5]
    pointsize = 1000

    user_height = 1.70  # user's eye height in m

    center = [0, -user_height, 0]
    length = 50
    radius = user_height

    t0 = InterTrial(cloud=gen_poslist_tunnel(density, center, radius, length), particlesize=pointsize)
    t1 = StaticCloud(cloud=gen_poslist_tunnel(density, center, radius, length), particlesize=pointsize)
    t2 = SinusoidalMotion(cloud=gen_poslist_tunnel(density, center, radius, length), particlesize=pointsize, freq=0.2)
    t3 = SinusoidalMotion(cloud=gen_poslist_tunnel(density, center, radius, length), particlesize=pointsize, freq=0.8)
    t4 = SinusoidalMotion(cloud=gen_poslist_tunnel(density, center, radius, length), particlesize=pointsize, freq=1.2)
    # t5 = SumOfSines(cloud=gen_poslist_tunnel(density, center, radius, length), particlesize=pointsize)
    t5 = RandomSequence(cloud=gen_poslist_tunnel(density, center, radius, length), particlesize=pointsize)
    # t7 = DriftCorr(size=100)

    trial_list = [t0, t1]
    # trial_list = [t0, t7]
    repetitions = 2

    with MyQt5App(renderer, trial_list) as qtPysideApp:

        #qtPysideApp.control.trial_order = [0, 2, 0, 7, 0, 5, 0, 4, 0, 6, 0, 1, 0, 7, 0, 3, 0, 6] * repetitions
        #qtPysideApp.control.trial_order = [0, 2, 0, 5, 0, 4, 0, 1, 0, 3, 0, 5] * repetitions
        qtPysideApp.control.trial_order = [0, 1] * repetitions
        qtPysideApp.run_loop()
