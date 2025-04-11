# --- START OF FILE graphics.py ---

import torch
import numpy as np
from collections import deque
import random
import os
import time
import ctypes
import math
from typing import Optional, Dict, Tuple, Deque, List

from config import MasterConfig as Config
from config import DEVICE, logger
from config import HEAD_MOVEMENT_LABELS
from utils import is_safe

try:
    from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QRect
    from PyQt5.QtGui import (QColor, QFont, QPainter, QPen, QOpenGLShader, QOpenGLShaderProgram,
                             QVector3D, QMouseEvent, QMatrix4x4)
    from OpenGL.GL import (
        glClearColor, glEnable, glDisable, glBlendFunc, glClear, glViewport, glDepthFunc, glDepthMask,
        glGenBuffers, glBindBuffer, glBufferData, glBufferSubData, glDeleteBuffers,
        glGenVertexArrays, glBindVertexArray, glEnableVertexAttribArray, glVertexAttribPointer,
        glDeleteVertexArrays, glDrawArrays, glGetAttribLocation, glUseProgram, glGetUniformLocation,
        glUniform1f, glUniform2f, glUniform3f, glPointSize, glHint, glGetError,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TRIANGLES, GL_POINTS,
        GL_FLOAT, GL_FALSE, GL_TRUE, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
        GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL, GL_PROGRAM_POINT_SIZE, GL_POINT_SMOOTH,
        GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_POINT_SMOOTH_HINT, GL_NICEST,
        GL_ARRAY_BUFFER, GL_NO_ERROR, GL_INVALID_ENUM, GL_INVALID_VALUE, GL_INVALID_OPERATION,
        glGetIntegerv, glUniformMatrix4fv, GL_VIEWPORT
    )
    from OpenGL.GL import GLfloat
    GL_ERROR_CODES = {
        GL_INVALID_ENUM: "GL_INVALID_ENUM",
        GL_INVALID_VALUE: "GL_INVALID_VALUE",
        GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
    }
except ImportError as e:
    logger.critical(f"graphics.py: PyQt5 or PyOpenGL import failed: {e}. Please install PyQt5, PyOpenGL, PyOpenGL-accelerate.")
    raise

try:
    import live2d.v3 as live2d_v3
    from live2d.v3 import LAppModel as Live2Dv3AppModel
except ImportError as e:
    logger.critical(f"graphics.py: live2d-py import failed: {e}. Ensure live2d-py[cubism3] and native Core lib are installed/accessible.")
    raise
except Exception as core_err:
    logger.critical(f"graphics.py: An unexpected error occurred related to the Live2D Core library: {core_err}. Ensure native lib installed.")
    raise

from dataclasses import dataclass, field

def check_gl_error(context_message=""):
    """Checks for and logs OpenGL errors."""
    error = glGetError()
    if error != GL_NO_ERROR:
        error_name = GL_ERROR_CODES.get(error, f"Unknown Error Code {error}")
        logger.error(f"OpenGL Error ({context_message}): {error_name} ({error})")
        return True
    return False

# --- Particle System Components ---
@dataclass
class Particle:
    position: QVector3D = field(default_factory=lambda: QVector3D(0, 0, 0))
    velocity: QVector3D = field(default_factory=lambda: QVector3D(0, 0, 0))
    color: QVector3D = field(default_factory=lambda: QVector3D(1, 1, 1))
    alpha: float = 1.0
    lifetime: float = 0.0
    initial_lifetime: float = 0.0
    size: float = 1.0

class ParticleSystem:
    def __init__(self, count: int = Config.Graphics.PARTICLE_COUNT):
        self.max_count = max(0, count)
        if self.max_count <= 0:
            logger.warning(f"ParticleSystem count is {self.max_count}. Disabling particles.")
        self.particles: List[Particle] = []
        self.particle_buffer = np.zeros(self.max_count * 8, dtype=np.float32) if self.max_count > 0 else np.array([], dtype=np.float32)
        self.initialize_particles()
        if self.max_count > 0:
            logger.info(f"Particle system initialized with {self.max_count} particles.")

    def initialize_particles(self):
        if self.max_count <= 0:
            return
        self.particles = [self.create_particle() for _ in range(self.max_count)]
        self._update_buffer()

    def create_particle(self) -> Particle:
        size_factor = random.uniform(Config.Graphics.PARTICLE_MIN_SIZE, Config.Graphics.PARTICLE_MAX_SIZE)
        pos = QVector3D(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-0.1, 0.1))
        vel = QVector3D(random.uniform(-0.01, 0.01), random.uniform(0.002, 0.016), random.uniform(-0.002, 0.002))
        col = QVector3D(random.uniform(0.6, 0.9), random.uniform(0.7, 1.0), random.uniform(0.8, 1.0))
        life = random.uniform(3.0, 6.0)
        return Particle(position=pos, velocity=vel, color=col, alpha=1.0, lifetime=life, initial_lifetime=life, size=size_factor)

    def update(self, delta_time: float, emotions: torch.Tensor):
        if self.max_count <= 0 or not self.particles:
            return
        dominant_emotion_idx = 0
        intensity = 0.5
        if isinstance(emotions, torch.Tensor) and emotions.numel() >= Config.Agent.EMOTION_DIM and is_safe(emotions):
            try:
                emo_cpu = emotions.cpu()
                dominant_emotion_idx = torch.argmax(emo_cpu).item()
                intensity = emo_cpu[dominant_emotion_idx].item()
            except Exception as e:
                logger.warning(f"Particle update error getting emotion: {e}")

        time_scale = delta_time * 60.0

        for i, particle in enumerate(self.particles):
            particle.position += particle.velocity * time_scale
            particle.lifetime -= delta_time

            lifetime_fraction = max(0.0, min(1.0, particle.lifetime / max(1e-6, particle.initial_lifetime)))
            particle.alpha = lifetime_fraction if lifetime_fraction < 0.2 else 1.0
            particle.alpha *= (0.7 + intensity * 0.3)
            particle.alpha = max(0.1, min(1.0, particle.alpha))

            base_color = self._get_base_color_for_emotion(dominant_emotion_idx)
            particle.color.setX(base_color[0])
            particle.color.setY(base_color[1])
            particle.color.setZ(base_color[2])

            if (particle.lifetime <= 0 or
                    abs(particle.position.x()) > 1.5 or
                    abs(particle.position.y()) > 1.5 or
                    abs(particle.position.z()) > 1.0):
                self.particles[i] = self.create_particle()

        self._update_buffer()

    def _get_base_color_for_emotion(self, index):
        colors = [(0.3, 1.0, 0.3), (1.0, 0.3, 0.3), (1.0, 1.0, 0.3),
                  (1.0, 0.6, 0.3), (0.3, 0.8, 1.0), (1.0, 0.3, 1.0)]
        default_color = (0.8, 0.8, 0.8)
        if 0 <= index < len(colors):
            return colors[index]
        return default_color

    def _update_buffer(self):
        if self.max_count <= 0 or not self.particles:
            return
        buffer_idx = 0
        for i, p in enumerate(self.particles):
            if buffer_idx + 8 > len(self.particle_buffer):
                logger.error(f"Particle buffer overflow at index {i}. Truncating.")
                break
            self.particle_buffer[buffer_idx:buffer_idx+8] = [
                p.position.x(), p.position.y(), p.position.z(),
                max(0.0, min(1.0, p.color.x())), max(0.0, min(1.0, p.color.y())), max(0.0, min(1.0, p.color.z())),
                max(0.1, min(1.0, p.alpha)),
                max(Config.Graphics.PARTICLE_MIN_SIZE, min(Config.Graphics.PARTICLE_MAX_SIZE, p.size))
            ]
            buffer_idx += 8

# --- Live2D Character Widget ---
class Live2DCharacter(QOpenGLWidget):
    character_initialized = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    interaction_detected = pyqtSignal()
    live2d_initialized_global = False

    def __init__(self, hud_widget=None, parent=None):
        super().__init__(parent)
        logger.info("Initializing Live2DCharacter...")
        self.model: Optional[Live2Dv3AppModel] = None
        self.live2d_initialized: bool = False
        self.model_loaded: bool = False
        self.model_path: str = Config.Graphics.MODEL_PATH
        self.emotions: torch.Tensor = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
        self.frame_count: int = 0
        self.time_elapsed: float = 0.0
        self.cursor_pos: Tuple[float, float] = (0.0, 0.0)
        self.cursor_history: Deque[Tuple[float, float]] = deque(maxlen=10)
        self.is_mouse_over: bool = False
        self.last_interaction_time: float = time.time()
        self.toggle_states: Dict[str, float] = {'blush': 0.0, 'wings': 0.0, 'mad': 0.0, 'confusion': 0.0}
        self.animation_params: Dict[str, Dict] = {
            'breath': {'phase': 0.0, 'speed': 0.08, 'magnitude': 0.5},
            'blink': {'timer': 0.0, 'interval': random.uniform(2.0, 5.0), 'state': 'WAITING', 'phase': 0.0},
            'hair_sway': {'phase': 0.0, 'speed': 0.06, 'magnitude': 8.0},
            'idle': {'phase': 0.0, 'speed': 0.03, 'magnitude': 5.0}
        }
        self.target_values: Dict[str, float] = {}
        self.current_values: Dict[str, float] = {}
        self.default_values: Dict[str, float] = {}
        self.parameter_velocities: Dict[str, float] = {}
        self.parameter_map: Dict[str | int, Dict] = {}
        self._failed_params: set = set()
        self.predicted_head_movement: Optional[str] = "idle"
        self.micro_movement_timer: float = 0.0
        self.micro_movement_interval: float = 0.1
        self.micro_movement_target_offset_x: float = 0.0
        self.micro_movement_target_offset_y: float = 0.0
        self.micro_movement_current_offset_x: float = 0.0
        self.micro_movement_current_offset_y: float = 0.0
        self.shader_program: Optional[QOpenGLShaderProgram] = None
        self.particle_shader: Optional[QOpenGLShaderProgram] = None
        self.projection_loc = -1
        self.particle_vao: int = 0
        self.particle_vbo: int = 0
        self.quad_vao: int = 0
        self.quad_vbo: int = 0
        self.particle_system = ParticleSystem()
        self.setMinimumSize(400, 600)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._init_live2d_library()
        self._setup_animation_timer()

    def _init_live2d_library(self):
        if not Live2DCharacter.live2d_initialized_global:
            try:
                live2d_v3.init()
                Live2DCharacter.live2d_initialized_global = True
                self.live2d_initialized = True
                logger.info("Live2D Core library initialized successfully.")
            except Exception as e:
                logger.critical(f"Live2D Core init FAILED: {e}", exc_info=True)
                self.error_occurred.emit(f"Live2D Core init failed: {e}")
                self.live2d_initialized = False
        else:
            self.live2d_initialized = True

    def _setup_animation_timer(self):
        self.animation_timer = QTimer(self)
        interval = max(10, int(1000.0 / Config.Graphics.FPS))
        self.animation_timer.setInterval(interval)
        self.animation_timer.timeout.connect(self._tick)

    def initializeGL(self):
        if not self.live2d_initialized:
            logger.error("Cannot initialize GL: Live2D Core not initialized.")
            self.character_initialized.emit(False)
            return
        try:
            self.makeCurrent()
        except Exception as e:
            logger.critical(f"Failed make context current: {e}", exc_info=True)
            self.error_occurred.emit(f"OpenGL context error: {e}")
            self.character_initialized.emit(False)
            return
        try:
            glClearColor(*Config.Graphics.BACKGROUND_COLOR)
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LEQUAL)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            glEnable(GL_PROGRAM_POINT_SIZE)
            if check_gl_error("After Enabling GL States"):
                return

            try:
                live2d_v3.glInit()
            except Exception as e:
                logger.critical(f"live2d_v3.glInit() FAILED: {e}", exc_info=True)
                raise RuntimeError("Live2D GL Initialization failed.") from e

            if not self._setup_background_shader():
                logger.warning("Failed to set up BG shader.")
            elif not self._setup_gl_buffers():
                logger.warning("Failed to set up BG buffers.")
            check_gl_error("After Background Shader/Buffer Setup")

            if not self._setup_particle_shader():
                logger.error("Failed to set up particle shader.")
            elif not self._setup_particle_buffers():
                logger.error("Failed to set up particle buffers.")
            else:
                self.particle_shader.bind()
                self.projection_loc = glGetUniformLocation(self.particle_shader.programId(), b"projection")
                self.particle_shader.release()
                if self.projection_loc == -1:
                    logger.error("Failed to get 'projection' uniform location.")
                else:
                    logger.info(f"Particle shader setup complete with projection_loc={self.projection_loc}")
            check_gl_error("After Particle Shader/Buffer Setup")

            self._load_model()
            if self.model_loaded and self.model:
                try:
                    self.model.Update()
                except Exception as e:
                    logger.error(f"Error initial model.Update() in initializeGL: {e}")
                self._create_parameter_mapper()
                self.animation_timer.start()
                logger.info("Live2D character ready, timer started.")
                self.character_initialized.emit(True)
            else:
                logger.error("GL init OK, but model load failed or model is null.")
                self.character_initialized.emit(False)
        except Exception as e:
            logger.critical(f"Critical error during OpenGL initialization: {e}", exc_info=True)
            self.error_occurred.emit(f"Graphics initialization error: {e}")
            self.character_initialized.emit(False)
        finally:
            try:
                self.doneCurrent()
            except Exception as e_done:
                logger.error(f"Error calling doneCurrent in initializeGL finally: {e_done}")

    def _setup_background_shader(self):
        try:
            self.shader_program = QOpenGLShaderProgram(self)
            vs_code = """#version 330 core
                layout (location = 0) in vec2 position; out vec2 fragCoord;
                void main() { gl_Position = vec4(position.x, position.y, 0.0, 1.0); fragCoord = position * 0.5 + 0.5; }"""
            fs_code = f"""#version 330 core
                uniform vec2 resolution; uniform float time; uniform vec3 emotion_color;
                in vec2 fragCoord; out vec4 FragColor;
                float random(vec2 st) {{ return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123); }}
                void main() {{
                    vec2 uv = fragCoord; vec3 finalColor = vec3({Config.Graphics.BACKGROUND_COLOR[0]}, {Config.Graphics.BACKGROUND_COLOR[1]}, {Config.Graphics.BACKGROUND_COLOR[2]});
                    float noise = random(uv * 5.0 + vec2(time * 0.1)) * 0.05; finalColor += emotion_color * noise * 2.0;
                    float vignette = smoothstep(0.8, 0.2, length(uv - 0.5)); finalColor *= vignette;
                    float dist = length(uv - 0.5); float pulse = sin(time * 1.5 + dist * 5.0) * 0.5 + 0.5;
                    float glow = smoothstep(0.5, 0.0, dist) * pulse * 0.1 * {Config.Graphics.GLOW_INTENSITY / 3.0:.3f}; finalColor += emotion_color * glow;
                    FragColor = vec4(clamp(finalColor, 0.0, 1.0), 1.0);
                }} """
            if not self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs_code):
                logger.error(f"BG VS compile failed: {self.shader_program.log()}")
                return False
            if not self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs_code):
                logger.error(f"BG FS compile failed: {self.shader_program.log()}")
                return False
            if not self.shader_program.link():
                logger.error(f"BG Shader link failed: {self.shader_program.log()}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error setting up background shader: {e}", exc_info=True)
            self.shader_program = None
            return False

    def _setup_particle_shader(self):
        try:
            self.particle_shader = QOpenGLShaderProgram(self)
            vs_code = """#version 330 core
                layout (location = 0) in vec3 position;
                layout (location = 1) in vec4 color;
                layout (location = 2) in float pointSize;
                uniform mat4 projection;
                out vec4 fragColor;
                void main() {
                    gl_Position = projection * vec4(position, 1.0);
                    fragColor = color;
                    gl_PointSize = pointSize;
                }"""
            fs_code = """#version 330 core
                in vec4 fragColor;
                out vec4 finalColor;
                void main() {
                    finalColor = fragColor;
                }"""
            if not self.particle_shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs_code):
                logger.error(f"Particle VS compile failed: {self.particle_shader.log()}")
                return False
            if not self.particle_shader.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs_code):
                logger.error(f"Particle FS compile failed: {self.particle_shader.log()}")
                return False
            if not self.particle_shader.link():
                logger.error(f"Particle Shader link failed: {self.particle_shader.log()}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error setting up particle shader: {e}", exc_info=True)
            if self.particle_shader:
                self.particle_shader.release()
            self.particle_shader = None
            return False

    def _setup_gl_buffers(self):
        try:
            quad_vertices = np.array([-1.0,-1.0, 1.0,-1.0, 1.0,1.0, -1.0,-1.0, 1.0,1.0, -1.0,1.0], dtype=np.float32)
            self.quad_vao = glGenVertexArrays(1)
            glBindVertexArray(self.quad_vao)
            self.quad_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
            glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
            position_loc = 0
            if self.shader_program and self.shader_program.isLinked():
                loc_check = glGetAttribLocation(self.shader_program.programId(), b"position")
                position_loc = loc_check if loc_check != -1 else 0
            else:
                logger.warning("BG Shader not linked, using default pos location 0.")
            glEnableVertexAttribArray(position_loc)
            glVertexAttribPointer(position_loc, 2, GL_FLOAT, GL_FALSE, 2 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            return True
        except Exception as e:
            logger.error(f"Error setting up background GL buffers: {e}", exc_info=True)
            if hasattr(self, 'quad_vbo') and self.quad_vbo:
                glDeleteBuffers(1, [self.quad_vbo])
                self.quad_vbo = 0
            if hasattr(self, 'quad_vao') and self.quad_vao:
                glDeleteVertexArrays(1, [self.quad_vao])
                self.quad_vao = 0
            return False

    def _setup_particle_buffers(self):
        if self.particle_system.max_count <= 0 or not self.particle_shader or not self.particle_shader.isLinked():
            logger.warning("Skipping particle buffer setup: Count is 0 or shader not ready.")
            return False
        try:
            self.particle_vao = glGenVertexArrays(1)
            glBindVertexArray(self.particle_vao)
            self.particle_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
            buffer_size = self.particle_system.max_count * 8 * ctypes.sizeof(GLfloat)
            glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_DYNAMIC_DRAW)
            stride = 8 * ctypes.sizeof(GLfloat)
            pos_loc = glGetAttribLocation(self.particle_shader.programId(), b"position")
            if pos_loc == -1:
                logger.error("Particle shader missing 'position' attribute.")
                glBindVertexArray(0)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                return False
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            color_loc = glGetAttribLocation(self.particle_shader.programId(), b"color")
            if color_loc == -1:
                logger.error("Particle shader missing 'color' attribute.")
                glBindVertexArray(0)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                return False
            color_offset = ctypes.c_void_p(3 * ctypes.sizeof(GLfloat))
            glEnableVertexAttribArray(color_loc)
            glVertexAttribPointer(color_loc, 4, GL_FLOAT, GL_FALSE, stride, color_offset)
            size_loc = glGetAttribLocation(self.particle_shader.programId(), b"pointSize")
            if size_loc == -1:
                logger.error("Particle shader missing 'pointSize' attribute.")
                glBindVertexArray(0)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                return False
            size_offset = ctypes.c_void_p(7 * ctypes.sizeof(GLfloat))
            glEnableVertexAttribArray(size_loc)
            glVertexAttribPointer(size_loc, 1, GL_FLOAT, GL_FALSE, stride, size_offset)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            logger.info(f"Particle VAO={self.particle_vao}/VBO={self.particle_vbo} setup OK for {self.particle_system.max_count} particles.")
            if check_gl_error("Particle Buffer Setup"):
                return False
            return True
        except Exception as e:
            logger.error(f"Error setting up particle GL buffers: {e}", exc_info=True)
            if hasattr(self, 'particle_vbo') and self.particle_vbo:
                glDeleteBuffers(1, [self.particle_vbo])
                self.particle_vbo = 0
            if hasattr(self, 'particle_vao') and self.particle_vao:
                glDeleteVertexArrays(1, [self.particle_vao])
                self.particle_vao = 0
            return False

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            self.error_occurred.emit(f"Model file not found: {self.model_path}")
            self.model_loaded = False
            return
        try:
            logger.info(f"Loading Live2D model from: {self.model_path}")
            self.model = Live2Dv3AppModel()
            w, h = self.width(), self.height()
            w, h = (400, 600) if w <= 0 or h <= 0 else (w, h)
            self.model.LoadModelJson(self.model_path)
            self.model.Resize(w, h)
            self.model_loaded = True
            logger.info(f"Loaded model: {os.path.basename(self.model_path)} (Initial size: {w}x{h})")
        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            self.model = None
            self.model_loaded = False
            self.error_occurred.emit(f"Model load failed: {e}")

    def _create_parameter_mapper(self):
        if not self.model or not self.model_loaded:
            logger.error("Cannot create param map: no model.")
            return
        logger.info("Creating parameter map using heuristic defaults.")
        self.parameter_map = {
            0: {'params': ['PARAM_BODY_ANGLE_X', 'PARAM_CHEEK', 'PARAM_MOUTH_FORM', 'PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN'], 'mapper': lambda p, x: x*10 if 'BODY' in p else (0.5+x*0.5), 'smoothing': 0.05},
            1: {'params': ['PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN', 'PARAM_MOUTH_OPEN_Y', 'PARAM_BROW_L_Y', 'PARAM_BROW_R_Y'], 'mapper': lambda p, x: (1.0-x*0.5) if 'EYE' in p else (x if 'MOUTH' in p else -x*0.8), 'smoothing': 0.1},
            2: {'params': ['PARAM_BROW_L_Y', 'PARAM_BROW_R_Y', 'PARAM_EYE_BALL_X', 'PARAM_ANGLE_Z'], 'mapper': lambda p, x: x*0.5 if 'BROW' in p else (x*0.6 if 'EYE' in p else x*15), 'smoothing': 0.1},
            3: {'params': ['PARAM_BROW_L_FORM', 'PARAM_BROW_R_FORM', 'PARAM_MOUTH_FORM', 'Param4'], 'mapper': lambda p, x: -x*0.8 if 'BROW' in p else (-x if 'MOUTH_F' in p else x), 'smoothing': 0.1},
            4: {'params': ['PARAM_BODY_ANGLE_X', 'PARAM_BREATH'], 'mapper': lambda p, x: x*0.1, 'smoothing': 0.05},
            5: {'params': ['PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN', 'PARAM_MOUTH_OPEN_Y'], 'mapper': lambda p, x: min(1.0, x*1.2), 'smoothing': 0.08},
            'cursor_x': {'params': ['PARAM_ANGLE_X', 'PARAM_EYE_BALL_X', 'PARAM_BODY_ANGLE_X', 'Param8', 'Param42'], 'mapper': lambda p, x: x*30 if 'ANGLE' in p else (x*0.8 if 'EYE' in p else x*10), 'smoothing': 0.05},
            'cursor_y': {'params': ['PARAM_ANGLE_Y', 'PARAM_EYE_BALL_Y', 'PARAM_BODY_ANGLE_Y', 'Param12', 'Param45'], 'mapper': lambda p, x: x*30 if 'ANGLE' in p else (x*0.8 if 'EYE' in p else x*10), 'smoothing': 0.05},
            'blush': {'params': ['Param'], 'mapper': lambda p, x: x, 'smoothing': 0.1},
            'wings': {'params': ['Param6'], 'mapper': lambda p, x: x, 'smoothing': 0.1},
            'mad': {'params': ['Param2'], 'mapper': lambda p, x: x, 'smoothing': 0.1},
            'confusion': {'params': ['Param4'], 'mapper': lambda p, x: x, 'smoothing': 0.1}
        }
        all_params = set()
        [all_params.update(p for p in cfg.get('params', []) if isinstance(p, str)) for cfg in self.parameter_map.values() if isinstance(cfg, dict)]
        all_params.update({'PARAM_BODY_ANGLE_Z', 'Param3', 'PARAM_BREATH'})
        logger.info(f"Initializing {len(all_params)} params from map/heuristics...")
        self._failed_params = set()
        self.default_values = {}
        self.target_values = {}
        self.current_values = {}
        self.parameter_velocities = {}
        model_param_ids = []
        if self.model:
            try:
                model_param_ids = self.model.GetParameterIds()
            except Exception as e:
                logger.warning(f"GetParameterIds() failed: {e}")
        for param_id in all_params:
            default_val = 0.0
            found_in_model = False
            if self.model and param_id in model_param_ids and hasattr(self.model, 'GetParameterDefaultValue'):
                try:
                    default_val = self.model.GetParameterDefaultValue(param_id)
                    found_in_model = True
                except Exception as e:
                    logger.warning(f"Could not get default for '{param_id}': {e}")
            if not found_in_model:
                default_val = Config.Graphics.EYE_PARAM_DEFAULT if 'EYE_L_OPEN' in param_id or 'EYE_R_OPEN' in param_id else (
                    Config.Graphics.MOUTH_PARAM_DEFAULT if 'MOUTH_OPEN_Y' in param_id else (0.5 if 'BREATH' in param_id else 0.0))
            self.default_values[param_id] = default_val
            self.target_values[param_id] = default_val
            self.current_values[param_id] = default_val
            self.parameter_velocities[param_id] = 0.0
            try:
                if self.model and param_id in model_param_ids and hasattr(self.model, 'SetParameterValue'):
                    self.model.SetParameterValue(param_id, default_val)
                elif not found_in_model:
                    pass
                else:
                    logger.warning(f"Model missing SetParameterValue for '{param_id}'.")
            except Exception as e:
                if param_id not in self._failed_params:
                    logger.warning(f"Initial set fail '{param_id}': {e}")
                self._failed_params.add(param_id)
        logger.info("Parameter map and initial values set.")

    def resizeGL(self, width: int, height: int):
        if width <= 0 or height <= 0:
            return
        try:
            self.makeCurrent()
            glViewport(0, 0, width, height)
            if self.model and self.model_loaded:
                self.model.Resize(width, height)
            if self.shader_program and self.shader_program.isLinked():
                self.shader_program.bind()
                self.shader_program.setUniformValue("resolution", float(width), float(height))
                self.shader_program.release()
        except Exception as e:
            logger.error(f"Error in resizeGL: {e}")
        finally:
            try:
                self.doneCurrent()
            except Exception as e_done:
                logger.error(f"Error calling doneCurrent in resizeGL finally: {e_done}")

    def paintGL(self):
        try:
            self.makeCurrent()
        except Exception as e:
            logger.error(f"Failed context in paintGL: {e}")
            return

        if not self.model_loaded or not self.model:
            glClearColor(*Config.Graphics.BACKGROUND_COLOR)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            try:
                self.doneCurrent()
            except Exception:
                pass
            return

        try:
            check_gl_error("Start paintGL")
            glClearColor(*Config.Graphics.BACKGROUND_COLOR)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # --- 1. Draw Background ---
            glDisable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            if self.shader_program and self.shader_program.isLinked() and self.quad_vao:
                self.shader_program.bind()
                self.shader_program.setUniformValue("time", self.time_elapsed)
                glBindVertexArray(self.quad_vao)
                glDrawArrays(GL_TRIANGLES, 0, 6)
                glBindVertexArray(0)
                self.shader_program.release()
            else:
                logger.warning("Background shader/VAO not ready.")
            if check_gl_error("After Background"):
                return

            # --- 2. Draw Particles (BEFORE Live2D Model) ---
            if self.particle_system.max_count > 0 and self.particle_shader and self.particle_shader.isLinked() and self.particle_vao and self.particle_vbo:
                try:
                    glUseProgram(0)
                    glBindVertexArray(0)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                    glDisable(GL_DEPTH_TEST)
                    glDepthMask(GL_FALSE)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glEnable(GL_PROGRAM_POINT_SIZE)
                    glEnable(GL_POINT_SMOOTH)
                    glViewport(0, 0, self.width(), self.height())
                    if check_gl_error("Before Particle Setup"):
                        logger.error("Particle setup failed due to GL error.")
                        return

                    self.particle_shader.bind()
                    glBindVertexArray(self.particle_vao)

                    projection_matrix = QMatrix4x4()
                    projection_matrix.ortho(-1.5, 1.5, -1.5, 1.5, -1.0, 1.0)
                    matrix_data = np.array(projection_matrix.data(), dtype=np.float32).reshape((4, 4)).T
                    if self.projection_loc != -1:
                        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, matrix_data)
                    else:
                        logger.error("Projection uniform not found. Skipping particle draw.")
                        self.particle_shader.release()
                        glBindVertexArray(0)
                        return
                    if check_gl_error("After Projection Matrix"):
                        return

                    glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
                    buffer_data = self.particle_system.particle_buffer
                    if buffer_data is not None and buffer_data.size >= self.particle_system.max_count * 8:
                        glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_data.nbytes, buffer_data)
                    else:
                        logger.error("Invalid particle buffer data. Skipping draw.")
                        self.particle_shader.release()
                        glBindVertexArray(0)
                        glBindBuffer(GL_ARRAY_BUFFER, 0)
                        return
                    if check_gl_error("After Buffer Update"):
                        return

                    glDrawArrays(GL_POINTS, 0, self.particle_system.max_count)
                    if check_gl_error("After Particle Draw"):
                        return

                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                    glBindVertexArray(0)
                    self.particle_shader.release()
                except Exception as e:
                    logger.error(f"Particle rendering error: {e}", exc_info=True)
            else:
                pass

            # --- 3. Draw Live2D Model (AFTER Particles) ---
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_POINT_SMOOTH)
            glDisable(GL_PROGRAM_POINT_SIZE)
            glUseProgram(0)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            check_gl_error("Before Live2D Draw Setup")

            glClear(GL_DEPTH_BUFFER_BIT)

            if self.model:
                self.model.Draw()
                if check_gl_error("After Model Draw"):
                    return

        except Exception as e:
            logger.error(f"OpenGL rendering error: {e}", exc_info=True)
            self.error_occurred.emit(f"Render error: {e}")
            if hasattr(self, 'animation_timer') and self.animation_timer.isActive():
                self.animation_timer.stop()
            glClearColor(0.3, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        finally:
            try:
                self.doneCurrent()
            except Exception as e_done:
                logger.error(f"Error calling doneCurrent in paintGL: {e_done}")

    def _tick(self):
        if not self.model_loaded or not self.model:
            return
        self.frame_count += 1
        interval_ms = self.animation_timer.interval()
        delta_time = interval_ms / 1000.0 if interval_ms > 0 else (1.0 / Config.Graphics.FPS)
        self.time_elapsed += delta_time
        self.particle_system.update(delta_time, self.emotions)
        self._update_animations(delta_time)
        self._update_model_parameters(delta_time)
        try:
            self.model.Update()
        except Exception as e:
            logger.error(f"Error model.Update() in _tick: {e}", exc_info=True)
            return
        self.update()

    def _update_animations(self, delta_time: float):
        time_scale = delta_time * 60.0
        breath = self.animation_params['breath']
        breath['phase'] = (breath['phase'] + breath['speed'] * time_scale) % (2 * math.pi)
        blink = self.animation_params['blink']
        blink['timer'] += delta_time
        blink_close_speed = 8.0
        blink_open_speed = 6.0
        if blink['state'] == 'WAITING':
            if blink['timer'] >= blink['interval']:
                blink['state'] = 'CLOSING'
                blink['phase'] = 0.0
                blink['timer'] = 0.0
        elif blink['state'] == 'CLOSING':
            blink['phase'] = min(1.0, blink['phase'] + delta_time * blink_close_speed)
            if blink['phase'] >= 1.0:
                blink['state'] = 'OPENING'
        elif blink['state'] == 'OPENING':
            blink['phase'] = max(0.0, blink['phase'] - delta_time * blink_open_speed)
            if blink['phase'] <= 0.0:
                blink['state'] = 'WAITING'
                blink['interval'] = random.uniform(1.5, 4.5)
                blink['timer'] = 0.0
        hair = self.animation_params['hair_sway']
        hair['phase'] = (hair['phase'] + hair['speed'] * time_scale) % (2 * math.pi)
        idle = self.animation_params['idle']
        idle['phase'] = (idle['phase'] + idle['speed'] * time_scale) % (2 * math.pi)
        if 'surprise' in self.animation_params:
            surprise = self.animation_params['surprise']
            surprise['timer'] = surprise.get('timer', 0.0) + delta_time
            if surprise['timer'] >= surprise.get('duration', 1.5):
                self.animation_params.pop('surprise', None)
        fade_rate = 2.0
        if not self.is_mouse_over:
            self.toggle_states['confusion'] = max(0.0, self.toggle_states['confusion'] - delta_time * fade_rate)
        self.micro_movement_timer += delta_time
        if self.micro_movement_timer >= self.micro_movement_interval:
            self.micro_movement_timer = 0.0
            if random.random() < 0.2:
                self.micro_movement_target_offset_x = random.uniform(-0.5, 0.5)
                self.micro_movement_target_offset_y = random.uniform(-0.5, 0.5)

    def update_predicted_movement(self, predicted_label: Optional[str]):
        if predicted_label in HEAD_MOVEMENT_LABELS:
            self.predicted_head_movement = predicted_label
        else:
            self.predicted_head_movement = "idle"

    def _apply_head_movement(self, active_targets: Dict[str, float], delta_time: float) -> Dict[str, float]:
        target_angle_x, target_angle_y, target_angle_z = 0.0, 0.0, 0.0
        current_time = self.time_elapsed
        move_type = self.predicted_head_movement
        if move_type == "slight_tilt":
            target_angle_z = 3.0 * math.sin(current_time * 1.5)
        elif move_type == "small_nod":
            target_angle_y = 4.0 * math.sin(current_time * 4.0)
        elif move_type == "gentle_nod":
            target_angle_y = 5.0 * math.sin(current_time * 2.5)
        elif move_type == "quick_nod":
            pulse = max(0, math.sin(current_time * 8.0) - 0.5) * 2
            target_angle_y = 6.0 * pulse
        elif move_type == "slow_tilt" or move_type == "ponder_tilt":
            target_angle_x = 5.0 * math.sin(current_time * 0.8)
            target_angle_z = 4.0 * math.sin(current_time * 0.6 + 0.5)
        elif move_type == "concerned_tilt" or move_type == "sympathetic_tilt":
            target_angle_x = -4.0 * math.sin(current_time * 1.2)
            target_angle_z = 3.0 * math.sin(current_time * 1.0)
        elif move_type == "curious_turn":
            target_angle_x = 8.0 * math.sin(current_time * 1.8)
        elif move_type == "quick_turn":
            pulse = max(0, math.sin(current_time * 7.0) - 0.4) * 1.5
            target_angle_x = 10.0 * pulse * random.choice([-1, 1])
        elif move_type == "negative_tilt":
            target_angle_z = -5.0 * math.sin(current_time * 2.0)
        elif move_type == "confused_tilt":
            target_angle_x = 4.0 * math.sin(current_time * 2.5 + 0.3)
            target_angle_z = -3.0 * math.sin(current_time * 1.8)
        elif move_type == "restless_shift":
            target_angle_x = 2.0 * math.sin(current_time * 3.5)
            target_angle_z = 1.5 * math.cos(current_time * 3.0)
        if 'PARAM_ANGLE_X' in active_targets:
            active_targets['PARAM_ANGLE_X'] += target_angle_x
        if 'PARAM_ANGLE_Y' in active_targets:
            active_targets['PARAM_ANGLE_Y'] += target_angle_y
        if 'PARAM_ANGLE_Z' in active_targets:
            active_targets['PARAM_ANGLE_Z'] += target_angle_z
        if 'PARAM_BODY_ANGLE_X' in active_targets:
            active_targets['PARAM_BODY_ANGLE_X'] += target_angle_x * 0.5
        if 'PARAM_BODY_ANGLE_Y' in active_targets:
            active_targets['PARAM_BODY_ANGLE_Y'] += target_angle_y * 0.5
        if 'PARAM_BODY_ANGLE_Z' in active_targets:
            active_targets['PARAM_BODY_ANGLE_Z'] += target_angle_z * 0.5
        return active_targets

    def _apply_micro_movements(self, active_targets: Dict[str, float], delta_time: float) -> Dict[str, float]:
        lerp_factor = 1.0 - math.exp(-delta_time / 0.1)
        self.micro_movement_current_offset_x += (self.micro_movement_target_offset_x - self.micro_movement_current_offset_x) * lerp_factor
        self.micro_movement_current_offset_y += (self.micro_movement_target_offset_y - self.micro_movement_current_offset_y) * lerp_factor
        micro_magnitude = 0.5
        if 'PARAM_ANGLE_X' in active_targets:
            active_targets['PARAM_ANGLE_X'] += self.micro_movement_current_offset_x * micro_magnitude
        if 'PARAM_ANGLE_Y' in active_targets:
            active_targets['PARAM_ANGLE_Y'] += self.micro_movement_current_offset_y * micro_magnitude
        if 'PARAM_ANGLE_Z' in active_targets:
            active_targets['PARAM_ANGLE_Z'] += self.micro_movement_current_offset_x * micro_magnitude * 0.3
        return active_targets

    def _update_model_parameters(self, delta_time: float):
        if not self.model or not self.model_loaded or not self.parameter_map:
            return
        active_targets = self.default_values.copy()
        breath_value = 0.5 + 0.5 * math.sin(self.animation_params['breath']['phase'])
        if 'PARAM_BREATH' in active_targets:
            active_targets['PARAM_BREATH'] = breath_value
        idle_tilt_val = math.sin(self.animation_params['idle']['phase']) * self.animation_params['idle']['magnitude']
        if 'PARAM_ANGLE_Z' in active_targets:
            active_targets['PARAM_ANGLE_Z'] += idle_tilt_val * 0.3
        hair_sway_val = math.sin(self.animation_params['hair_sway']['phase']) * self.animation_params['hair_sway']['magnitude']
        if 'Param8' in active_targets:
            active_targets['Param8'] += hair_sway_val * 0.1
        if self.emotions.numel() >= Config.Agent.EMOTION_DIM and is_safe(self.emotions):
            emo_cpu = self.emotions.cpu()
            for emotion_idx, config_val in self.parameter_map.items():
                if isinstance(emotion_idx, int) and 0 <= emotion_idx < Config.Agent.EMOTION_DIM:
                    emotion_value = emo_cpu[emotion_idx].item()
                    if emotion_value > 0.05:
                        for param_id in config_val.get('params', []):
                            if param_id in active_targets:
                                mapped_value = config_val['mapper'](param_id, emotion_value)
                                if 'ANGLE' in param_id or 'FORM' in param_id:
                                    active_targets[param_id] += mapped_value - self.default_values.get(param_id, 0.0)
                                else:
                                    active_targets[param_id] = mapped_value
        if self.is_mouse_over and self.cursor_history:
            if len(self.cursor_history) > 0:
                avg_x = sum(x for x, _ in self.cursor_history) / len(self.cursor_history)
                avg_y = sum(y for _, y in self.cursor_history) / len(self.cursor_history)
                for key, loop_value in [('cursor_x', avg_x), ('cursor_y', avg_y)]:
                    config_val_lookup = self.parameter_map.get(key)
                    if config_val_lookup:
                        for param_id in config_val_lookup.get('params', []):
                            if param_id in active_targets:
                                mapped_value = config_val_lookup['mapper'](param_id, loop_value)
                                if 'ANGLE' in param_id or 'EYE_BALL' in param_id or 'BODY' in param_id:
                                    active_targets[param_id] = mapped_value
        for toggle_key, toggle_value in self.toggle_states.items():
            config_val = self.parameter_map.get(toggle_key)
            if config_val and toggle_value > 0.01:
                for param_id in config_val.get('params', []):
                    if param_id in active_targets:
                        mapped_value = config_val['mapper'](param_id, toggle_value)
                        if toggle_key == 'confusion':
                            active_targets[param_id] += mapped_value
                        else:
                            active_targets[param_id] = mapped_value
        active_targets = self._apply_head_movement(active_targets, delta_time)
        active_targets = self._apply_micro_movements(active_targets, delta_time)
        blink = self.animation_params['blink']
        eye_open_value = 1.0
        if blink['state'] == 'CLOSING':
            eye_open_value = 1.0 - blink['phase']
        elif blink['state'] == 'OPENING':
            eye_open_value = blink['phase']
        eye_params = ['PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN']
        for param_id in eye_params:
            if param_id in active_targets:
                active_targets[param_id] *= eye_open_value
        for param_id, target_val in active_targets.items():
            if param_id not in self.current_values:
                continue
            current_val = self.current_values[param_id]
            param_config = None
            for cfg in self.parameter_map.values():
                if isinstance(cfg, dict) and param_id in cfg.get('params', []):
                    param_config = cfg
                    break
            smoothing_alpha = param_config.get('smoothing', 0.1) if param_config else 0.1
            lerp_factor = 1.0 - math.exp(-delta_time / max(0.001, smoothing_alpha))
            new_current = current_val + (target_val - current_val) * lerp_factor
            min_val, max_val = -100.0, 100.0
            if self.model and hasattr(self.model, 'GetParameterMinimumValue') and hasattr(self.model, 'GetParameterMaximumValue'):
                try:
                    min_val = self.model.GetParameterMinimumValue(param_id)
                    max_val = self.model.GetParameterMaximumValue(param_id)
                except:
                    pass
            else:
                if 'ANGLE' in param_id:
                    min_val, max_val = -30.0, 30.0
                elif '_OPEN' in param_id or 'MOUTH_OPEN' in param_id or 'BREATH' in param_id or 'CHEEK' in param_id:
                    min_val, max_val = 0.0, 1.0
                elif '_FORM' in param_id or 'SMILE' in param_id:
                    min_val, max_val = -1.0, 1.0
                elif 'EYE_BALL' in param_id:
                    min_val, max_val = -1.0, 1.0
                elif param_id in ['Param', 'Param6', 'Param2', 'Param4']:
                    min_val, max_val = 0.0, 1.0
            clamped_new_current = max(min_val, min(max_val, new_current))
            self.current_values[param_id] = clamped_new_current
            try:
                if self.model and hasattr(self.model, 'SetParameterValue'):
                    try:
                        self.model.SetParameterValue(param_id, clamped_new_current)
                    except TypeError:
                        try:
                            self.model.SetParameterValue(param_id, clamped_new_current, 1.0)
                        except Exception as e_weight:
                            raise RuntimeError(f"SetParameterValue failed: {e_weight}") from e_weight
                    self._failed_params.discard(param_id)
                else:
                    if param_id not in self._failed_params:
                        logger.warning(f"Model/SetParam missing for '{param_id}'")
                    self._failed_params.add(param_id)
            except TypeError as te:
                if param_id not in self._failed_params:
                    logger.warning(f"TypeError setting param '{param_id}': {te}")
                self._failed_params.add(param_id)
            except Exception as e:
                if param_id not in self._failed_params:
                    logger.warning(f"Failed set param '{param_id}': {type(e).__name__}")
                self._failed_params.add(param_id)

    def update_emotions(self, emotions_tensor: torch.Tensor):
        if isinstance(emotions_tensor, torch.Tensor) and emotions_tensor.shape == (Config.Agent.EMOTION_DIM,) and is_safe(emotions_tensor):
            self.emotions = emotions_tensor.detach().to(DEVICE)
        else:
            logger.warning(f"Invalid emotion data: type={type(emotions_tensor)}, shape={emotions_tensor.shape if isinstance(emotions_tensor, torch.Tensor) else 'N/A'}")

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.model_loaded:
            return
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0:
            return
        try:
            pos = event.pos()
            current_x = pos.x()
            current_y = pos.y()
            x_norm = (current_x / width) * 2.0 - 1.0
            y_norm = 1.0 - (current_y / height) * 2.0
            self.cursor_pos = (x_norm, y_norm)
            self.cursor_history.append(self.cursor_pos)
            self.is_mouse_over = True
            self.last_interaction_time = time.time()
            self.toggle_states['confusion'] = min(1.0, self.toggle_states['confusion'] + 0.1)
        except Exception as e:
            logger.error(f"Error processing mouse event coordinates: {e}")

    def mousePressEvent(self, event: QMouseEvent):
        if not self.model_loaded:
            return
        try:
            button = event.button()
            self.interaction_detected.emit()
            self.last_interaction_time = time.time()
            if button == Qt.MouseButton.LeftButton:
                self.toggle_states['blush'] = 1.0 - self.toggle_states['blush']
            elif button == Qt.MouseButton.RightButton:
                self.toggle_states['wings'] = 1.0 - self.toggle_states['wings']
            elif button == Qt.MouseButton.MiddleButton:
                self.toggle_states['mad'] = 1.0 - self.toggle_states['mad']
            self.animation_params['surprise'] = {'timer': 0.0, 'duration': 1.5}
            self.setFocus()
        except Exception as e:
            logger.error(f"Error processing mousePressEvent: {e}", exc_info=True)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.is_mouse_over = True

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.is_mouse_over = False
        self.cursor_history.clear()

    def cleanup(self):
        logger.info("Cleaning up Live2DCharacter resources...")
        try:
            self.makeCurrent()
        except Exception as e:
            logger.error(f"Failed context for cleanup: {e}")
        try:
            if hasattr(self, 'animation_timer') and self.animation_timer:
                self.animation_timer.stop()
            if hasattr(self, 'quad_vao') and self.quad_vao:
                glDeleteVertexArrays(1, [self.quad_vao])
                self.quad_vao = 0
            if hasattr(self, 'quad_vbo') and self.quad_vbo:
                glDeleteBuffers(1, [self.quad_vbo])
                self.quad_vbo = 0
            if hasattr(self, 'particle_vao') and self.particle_vao:
                glDeleteVertexArrays(1, [self.particle_vao])
                self.particle_vao = 0
            if hasattr(self, 'particle_vbo') and self.particle_vbo:
                glDeleteBuffers(1, [self.particle_vbo])
                self.particle_vbo = 0
            if self.model:
                release_method = getattr(self.model, 'Release', None)
                if callable(release_method):
                    try:
                        release_method()
                    except Exception as e_rel:
                        logger.error(f"Error calling model.Release(): {e_rel}")
                self.model = None
                self.model_loaded = False
            if self.shader_program and hasattr(self.shader_program, 'release'):
                self.shader_program.release()
                self.shader_program = None
            if self.particle_shader and hasattr(self.particle_shader, 'release'):
                self.particle_shader.release()
                self.particle_shader = None
            if Live2DCharacter.live2d_initialized_global:
                try:
                    live2d_v3.dispose()
                    Live2DCharacter.live2d_initialized_global = False
                    logger.info("Live2D Core library disposed.")
                except Exception as e_dispose:
                    logger.error(f"Error calling live2d_v3.dispose(): {e_dispose}")
        except Exception as e:
            logger.error(f"Error during GL resource cleanup: {e}", exc_info=True)
        finally:
            try:
                self.doneCurrent()
            except Exception as e_done:
                logger.error(f"Error calling doneCurrent during cleanup: {e_done}")
            logger.info("Character cleanup finished.")

# --- END OF FILE graphics.py ---
