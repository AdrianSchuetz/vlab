from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import QGLWidget, QGLFormat

from OpenGL.GL import *
import OpenGL.GL.shaders
import sys
import time
import numpy as np
import pyrr


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


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


def gen_poslist(density=1, origin=[0, 0, 0], size=[0, 0, 0]):
    # get point positions
    num_points = int(max(1, size[0] * size[1] * size[2] * density))
    print(num_points)
    x_pos = size[0] * (np.random.rand(1, num_points) - 0.5) - origin[0]
    y_pos = size[1] * (np.random.rand(1, num_points) - 0.5) - origin[1]
    z_pos = size[2] * (np.random.rand(1, num_points) - 0.5) - origin[2]
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


class TrialControl(QMainWindow):
    def __init__(self):
        super(TrialControl, self).__init__()

        self.done = False
        # self.tt = QTimer(self)

        # self.tt.timeout.connect(self.trial_done)
        # self.tt.start(10000)
        self.pointer = 0

        self.nb = QPushButton('Next')
        self.rb = QPushButton('Return')
        self.resb = QPushButton('Reset Trial')
        layout = QVBoxLayout()
        layout.addWidget(self.nb)
        layout.addWidget(self.rb)
        layout.addWidget(self.resb)
        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)

        self.nb.pressed.connect(self.next)
        self.rb.pressed.connect(self.backward)
        self.resb.pressed.connect(self.reset)
        self.curtrial = 0
        self.trial_list = [Trial()]

        self.trial_list[self.curtrial].signals.result.connect(self.trial_end)
    def reset(self):
        self.trial_list[self.curtrial].reset()

    def trial_end(self,data):
        self.next()

    def next(self):
        if self.curtrial < len(self.trial_list) - 1:
            self.curtrial += 1
            print(self.trial_list[self.curtrial].name)
        else:
            print("Last Trial")

    def backward(self):
        if self.curtrial is not 0:
            self.curtrial -= 1

        else:
            print("Can't go back further")
        return


class StimCanvas(QGLWidget):
    def __init__(self):
        super(StimCanvas, self).__init__()

        self.control = TrialControl()
        self.control.show()
        self.t = time.time()
        glformat = QGLFormat()
        glformat.setVersion(4, 1)
        glformat.setProfile(QGLFormat.CoreProfile)

        glformat.setDoubleBuffer(True)
        self.drawtimer = QTimer(self)
        self.drawtimer.timeout.connect(self.update)
        self.drawtimer.start(1000 / 60)
        self.curframe = 0
        self.curtrial = 0

        self.trial_seq = [0]

    def initializeGL(self):

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        print("init GL")
        for trial in self.control.trial_list:
            trial.init_gl()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self.control.trial_list == []:

            self.control.trial_list[self.control.curtrial].display_gl()

        self.curframe += 1

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

    def append_trial(self, trial):

        self.control.trial_list.append(trial)
        print(len(self.control.trial_list))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            self.control.close()


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


class GlCube(object):
    def __init__(self, position=None, size=1):
        if position is not None:
            self.position = position
        else:
            self.position = [0.1, 0.0, 0.0]

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
            in vec3 offset;          
            uniform mat4 modelview;
            uniform mat4 model;
            uniform mat4 projection;            
            out vec3 newColor;
            void main()
            {
                vec3 final_pos = vec3(position.x + offset.x, position.y + offset.y, position.z + offset.z );
                gl_Position = projection * modelview * model * vec4(final_pos, 1.0f);
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
        print("Cube shader done")
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

        instance_array = []
        offset = -50

        for z in range(0, 100, 2):
            for y in range(0, 100, 2):
                for x in range(0, 100, 2):
                    translation = pyrr.Vector3([0.0, 0.0, 0.0])
                    translation.x = x + offset
                    translation.y = y + offset
                    translation.z = z + offset
                    instance_array.append(translation)

        instance_array = np.array(instance_array, np.float32).flatten()
        print(len(instance_array))
        self.offset_loc = glGetAttribLocation(self.shader, "offset")
        self.ivbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.ivbo)
        glBufferData(GL_ARRAY_BUFFER, instance_array.itemsize * len(instance_array), instance_array, GL_STATIC_DRAW)
        self.ins_len = len(instance_array)
        glVertexAttribPointer(self.offset_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(self.offset_loc)

        glVertexAttribDivisor(self.offset_loc, 1)

        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.modelview_loc = glGetUniformLocation(self.shader, "modelview")
        self.model_loc = glGetUniformLocation(self.shader, "model")

        glBindVertexArray(0)
        print("finish init GLCube")

    def draw_gl(self, modelview=None, projection=None, model=None):

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
        glPointSize(10)
        glDrawElementsInstanced(GL_POINTS, len(self.indices), GL_UNSIGNED_INT, None, 125000)
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
                if (sin( 2*PI * (lifetimephase + time + lifetime)) >= 0){
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

    def draw_gl(self, modelview=None, projection=None, model=None):

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

        self.num_points = int(self.radius ** 2 * self.height * self.density)
        self.starttime = time.time()
        self.pos_list = self.gen_postlist()
        print(self.pos_list)

    def gen_postlist(self):

        # get point positions
        angle = 2 * np.pi * np.random.rand(1, self.num_points)
        x_pos = np.sin(angle) * self.radius - self.position[0]
        z_pos = np.cos(angle) * self.radius - self.position[2]

        y_pos = (np.random.rand(1, self.num_points) - 0.5) * self.height - self.position[1]

        # get colors
        c1 = np.ones(self.num_points)
        c2 = np.ones(self.num_points)
        c3 = np.ones(self.num_points)

        # lifetime
        lt = np.random.rand(1, self.num_points)
        # lt = 0.2 * np.ones(self.num_points)

        pos_list = np.ones(7 * self.num_points)
        pos_list[0::7] = x_pos
        pos_list[1::7] = y_pos
        pos_list[2::7] = z_pos
        pos_list[3::7] = c1
        pos_list[4::7] = c2
        pos_list[5::7] = c3
        pos_list[6::7] = lt
        print(pos_list)
        return np.array(pos_list, dtype=np.float32)

    def init_gl(self):
        print("init zylinder")

        vertex_shader = """
            #version 330
            in layout(location = 0) vec3 position;
            in layout(location = 1) vec3 color;
            in layout(location = 2) float lifetime;
            uniform mat4 modelview;
            uniform mat4 projection;
            uniform mat4 model;
            uniform float time;
            out vec3 newColor;
            
            const float PI = 3.1415926;
            void main()
            {
                gl_Position = projection * modelview * model * vec4(position, 1.0f);
                if (sin(2 * PI * (time + lifetime)) <= 0){
                     newColor = color;
                }
                else {
                    newColor = vec3(0.0, 0.0, 0.0);
                }           
                
            }
            """

        fragment_shader = """
            #version 330
            in vec3 newColor;
            
            out vec4 outColor;
            void main()
            {
                if (newColor == vec3(0.0, 0.0, 0.0) ){
                    discard;
                } 
                else {
                    outColor = vec4(newColor, 1.0f);
                }
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
        self.time_pos = glGetUniformLocation(self.shader, "time")

        glBindVertexArray(self.vao)
        # send vertices to graka

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # size in byte (pro zahl 4)
        glBufferData(GL_ARRAY_BUFFER, self.num_points * 7 * 4, self.pos_list, GL_STATIC_DRAW)

        # get position of variable position in shader program

        # (position of data retrieved, how many vertices, what data type, normalized ? ,
        # how many steps between relevant data, offset in the beginning)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        glBindVertexArray(0)
        print("done init zylinder")

    def draw_gl(self, modelview=None, projection=None, model=None):

        if modelview is None:
            modelview = pyrr.matrix44.create_from_translation(pyrr.Vector3(pyrr.Vector3([0.0, 0.0, 0.0])))

        if projection is None:
            projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        if model is None:
            model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))
        else:
            model = model @ pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position))

        glUseProgram(self.shader)
        glBindVertexArray(self.vao)

        glUniformMatrix4fv(self.modelview_loc, 1, GL_FALSE, modelview)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)
        print(np.float32(time.time() - self.starttime))
        glUniform1f(self.time_pos, time.time() - self.starttime)
        # glEnable(GL_POINT_SMOOTH)
        glPointSize(self.size)

        glDrawArrays(GL_POINTS, 0, self.num_points)
        glBindVertexArray(0)
        glUseProgram(0)


class GlPoint(object):
    def __init__(self, position=None, size=None, motionstart=None):
        if position is not None:
            self.position = position
        else:
            self.position = [0.0, 0.0, 0.0]
        self.ori_pos = position
        if size is not None:
            self.size = size
        else:
            self.size = 10
        if motionstart is not None:
            self.ms = motionstart
        else:
            self.ms = 0
        self.shader = None
        self.vao = None
        self.timer = CustTimer()
        self.mt = CustTimer(False)

    def rand_ms(self, low, up):
        self.ms = np.random.uniform(low, up)

    def reset(self):
        self.timer.reset()
        self.mt.stop()

    def move(self, new_pos=None):
        if new_pos is not None:
            self.position = new_pos

    def init_gl(self):
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
        vertex_shader = """
            #version 330
            uniform mat4 view;
            uniform mat4 projection;
            uniform mat4 model;
            void main()
            {
                gl_Position = projection * view * model *  vec4(0.0f, 0.0f, 0.0f, 1.0f);
                vec3 ndc = gl_Position.xyz / gl_Position.w ;
                float zDist = 1.0-ndc.z;
                gl_PointSize = 2000.0*zDist;
            }

        """

        fragment_shader = """
            #version 330
            in vec3 newColor;
            out vec4 outColor;
            void main()
            {
                vec2 cxy = 2.0 * gl_PointCoord -1.0;                
                float r = dot(cxy, cxy);
                if (r > 0.5) {
                    discard;
                } 
                outColor = vec4( 1.0f,1.0f,1.0f, 1.0f);
            }
        """

        self.shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(fragment_shader,
                                                                                       GL_FRAGMENT_SHADER))

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1, 0.1, 100)

        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.model_loc = glGetUniformLocation(self.shader, "model")

    def draw_gl(self, modelview=None, projection=None):

        self.view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -3.0]))

        if self.timer.get_time() >= self.ms and not self.mt.running:
            self.mt.start()

        self.model = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.position) +
                                                           pyrr.Vector3(
                                                               [0.0, 0.0,
                                                                np.sin(0.1 * 2 * np.pi * self.mt.get_time())]))

        glUseProgram(self.shader)

        glBindVertexArray(self.vao)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection)
        # glEnable(GL_POINT_SMOOTH)
        glPointSize(self.size)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, self.model)
        glDrawArrays(GL_POINTS, 0, 3)
        glBindVertexArray(0)
        glUseProgram(0)


class Trial(QObject):
    # init funktion. hier sollen alle werte, die für jeden aufruf des trials gleich sind
    # gesetzt werden (z.B. Name, Dauer, Actor). Außerdem sollen hier alle variable initialisiert werden
    def __init__(self, dur=None, name=None):
        if dur is not None:
            self.dur = dur
        else:
            self.dur = np.inf
        if name is not None:
            self.name = name
        else:
            self.name = "Default"

        self.curframe = 0
        self.actor_list = []
        self.data = {}
        self.signals = WorkerSignals()
    def append_actor(self, actor):
        self.actor_list.append(actor)

    def init_gl(self):
        print("init actors")
        for actor in self.actor_list:
            actor.init_gl()

    # wird bei jedem trial start aufgerufen. hier muss der aktuelle frame resettet werden. außerdem können hier
    # random werte neu gewürfelt werden. außerdem werden alle actor auf anfang zurückgesetzt.

    def reset(self):
        self.curframe = 0

        for actor in self.actor_list:
            actor.reset()

    # diese methode wird bei jedem frame aufgerufen. hier muss der aktuelle frame hochgezählt werden und gecheckt werden,
    # ob der trial weiter gehen soll oder nicht
    def display_gl(self, modelview=None, projection=None):

        for actor in self.actor_list:
            actor.draw_gl(modelview, projection)
        self.curframe += 1
        if self.curframe >= self.dur:
            self.end()

    # wird aufgerufen wenn der trial endet. muss immer True zurückgeben, kann aber auch andere Ding, bei Bedarf z.B. Kontrolpanele schließen
    def end(self):
        self.signals.result(self.data)
        self.reset()


class RotationZylinderTrial(Trial):
    def __init__(self, dur=None, name=None):
        super(RotationZylinderTrial, self).__init__(dur, name)
        self.name = "Rotation Zylinder"
        z = GlPointZylinder(position=[0, 0, 0], radius=10, height=100, size=10, density=10)
        self.actor_list = [z]
        self.vel = 1 / 1000

    def display_gl(self, modelview=None, projection=None):
        for actor in self.actor_list:
            transform = pyrr.matrix44.create_from_y_rotation(self.curframe * self.vel)

            actor.draw_gl(modelview, projection, model=transform)
        self.curframe += 1


class RotationZylinderTrial2(Trial):
    def __init__(self, dur=None, name=None):
        super(RotationZylinderTrial2, self).__init__(dur, name)
        self.name = "Rotation Zylinder"
        zyl = gen_poslist_zylinder(density=100, origin=[0, 0, 0], radius=10, height=100)
        z = GlObjectStructure(size=0.01, structure=zyl)
        self.actor_list = [z]
        self.vel = 1 / 1000

    def display_gl(self, modelview=None, projection=None):
        for actor in self.actor_list:
            transform = pyrr.matrix44.create_from_y_rotation(self.curframe * self.vel)

            actor.draw_gl(modelview, projection, model=transform)
        self.curframe += 1


class LinearMotion(Trial):
    def __init__(self, dur=None, name=None):
        super(LinearMotion, self).__init__(dur, name)
        cloud = gen_poslist(1000, size=[100, 1, 100])
        c = GlObjectStructure(size=0.01, structure=cloud)
        self.name = "Linear Motion"
        self.actor_list = [c]
        self.timer = CustTimer()
        self.ms_up = 0
        self.ms_low = 0
        self.roll_ms()
        self.traj = np.array([0.0, 0.0, 0.1])

    def roll_ms(self):
        self.ms = np.random.uniform(self.ms_low, self.ms_up)

    def reset(self):
        super(LinearMotion, self).reset()
        self.timer.reset()
        self.roll_ms()

    def display_gl(self, modelview=None, projection=None):
        for actor in self.actor_list:

            actor.draw_gl(modelview, projection)
            if self.timer.get_time() >= self.ms:
                actor.move(actor.ori_pos + self.traj * (self.timer.get_time() - self.ms))
        self.curframe += 1
        if self.curframe >= self.dur:
            self.end()


class StaticCloud(Trial):
    def __init__(self, dur=None, name=None):
        super(StaticCloud, self).__init__(dur, name)
        cloud = gen_poslist(1, size=[100, 1, 100])
        c = GlObjectStructure(size=0.01, structure=cloud)
        self.name = "Static Cloud"
        self.actor_list = [c]

    def display_gl(self, modelview=None, projection=None):
        for actor in self.actor_list:
            actor.draw_gl(modelview, projection)

        self.curframe += 1
        if self.curframe >= self.dur:
            self.end()


class SinosoidalMotion(Trial):
    def __init__(self, dur=None, name=None):
        super(SinosoidalMotion, self).__init__(dur, name)
        cloud = gen_poslist(1, size=[100, 1, 100])
        c = GlObjectStructure(size=0.01, structure=cloud)
        self.name = "Sinosoidal Motion"
        self.actor_list = [c]
        self.timer = CustTimer()
        self.ms_up = 0
        self.ms_low = 0
        self.roll_ms()
        self.amp = np.array([0.0, 0.0, 1])
        self.freq = 0.1
        self.phase = 0

    def roll_ms(self):
        self.ms = np.random.uniform(self.ms_low, self.ms_up)

    def reset(self):
        super(LinearMotion, self).reset()
        self.timer.reset()
        self.roll_ms()

    def display_gl(self, modelview=None, projection=None):
        for actor in self.actor_list:

            actor.draw_gl(modelview, projection)
            if self.timer.get_time() >= self.ms:
                actor.move(actor.ori_pos + self.amp *
                           np.sin(2 * np.pi * self.freq * (self.timer.get_time() - self.ms) + self.phase))
        self.curframe += 1
        if self.curframe >= self.dur:
            self.end()


class testTrial(Trial):
    def display(self):
        for actor in self.actor_list:
            model = pyrr.matrix44.create_from_y_rotation(self.curframe / 1000)

            actor.draw_gl(model=model)
        self.curframe += 1


test = testTrial()

app = QApplication(sys.argv)
sc = StimCanvas()
t1 = Trial()
t1.append_actor(GlPoint())
sc.append_trial(t1)

sc.append_trial(LinearMotion())
sc.append_trial(SinosoidalMotion())
sc.append_trial(StaticCloud())
sc.append_trial(RotationZylinderTrial())
sc.append_trial(RotationZylinderTrial2())
sc.show()
sys.exit(app.exec_())
