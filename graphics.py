# --- START OF FILE graphics.py ---

import torch
import numpy as np
from collections import deque
import random
import logging
import os
import time
import ctypes
import math
from typing import Optional, Dict, Tuple, Deque, List

# --- NEW: Import TextBlob ---
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob library not found ('pip install textblob' + 'python -m textblob.download_corpora'). Sentiment analysis for head movement disabled.")
# --- End Import ---


from config import MasterConfig as Config
from config import DEVICE, logger
from utils import is_safe

try:
    from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QRect
    from PyQt5.QtGui import (QColor, QFont, QPainter, QPen, QOpenGLShader, QOpenGLShaderProgram,
                             QVector3D, QMouseEvent)
    from OpenGL.GL import (
        glClearColor, glEnable, glDisable, glBlendFunc, glClear, glViewport, glDepthFunc, glDepthMask,
        glGenBuffers, glBindBuffer, glBufferData, glBufferSubData, glDeleteBuffers,
        glGenVertexArrays, glBindVertexArray, glEnableVertexAttribArray, glVertexAttribPointer,
        glDeleteVertexArrays, glDrawArrays, glGetAttribLocation, glUseProgram, glGetUniformLocation,
        glUniform1f, glUniform2f, glUniform3f, glPointSize, glHint,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TRIANGLES, GL_POINTS,
        GL_FLOAT, GL_FALSE, GL_TRUE, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
        GL_DEPTH_TEST, GL_BLEND, GL_LEQUAL, GL_PROGRAM_POINT_SIZE, GL_POINT_SMOOTH,
        GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_POINT_SMOOTH_HINT, GL_NICEST,
        GL_ARRAY_BUFFER
    )
    from OpenGL.GL import GLfloat
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

# --- Particle System Components (Remains the same) ---
@dataclass
class Particle:
    position: QVector3D = field(default_factory=lambda: QVector3D(0, 0, 0))
    velocity: QVector3D = field(default_factory=lambda: QVector3D(0, 0, 0))
    color: QVector3D = field(default_factory=lambda: QVector3D(1, 1, 1))
    alpha: float = 1.0
    lifetime: float = 0.0
    size: float = 1.0

class ParticleSystem:
    def __init__(self, count: int = Config.Graphics.PARTICLE_COUNT):
        self.max_count = count
        if self.max_count <= 0: logger.warning(f"ParticleSystem count is {self.max_count}. Disabling particles."); self.max_count = 0
        self.particles: List[Particle] = []
        self.particle_buffer = np.zeros(self.max_count * 7, dtype=np.float32) if self.max_count > 0 else np.array([], dtype=np.float32)
        self.initialize_particles()
        if self.max_count > 0: logger.info(f"Particle system initialized with {self.max_count} particles.")

    def initialize_particles(self):
        if self.max_count <= 0: return
        self.particles = [self.create_particle() for _ in range(self.max_count)]
        logger.debug(f"Initialized {len(self.particles)} particles.")
        self._update_buffer()

    def create_particle(self) -> Particle:
        size_factor = random.uniform(Config.Graphics.PARTICLE_MIN_SIZE, Config.Graphics.PARTICLE_MAX_SIZE)
        pos = QVector3D(random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2), random.uniform(-0.2, 0.2))
        vel = QVector3D(random.uniform(-0.005, 0.005), random.uniform(0.001, 0.008), random.uniform(-0.001, 0.001))
        col = QVector3D(random.uniform(0.6, 0.9), random.uniform(0.7, 1.0), random.uniform(0.8, 1.0))
        life = random.uniform(2.0, 5.0)
        return Particle(position=pos, velocity=vel, color=col, alpha=1.0, lifetime=life, size=size_factor)

    def update(self, delta_time: float, emotions: torch.Tensor):
        if self.max_count <= 0 or not self.particles: return
        dominant_emotion_idx = 0; intensity = 0.5
        if isinstance(emotions, torch.Tensor) and emotions.numel() >= Config.Agent.EMOTION_DIM and is_safe(emotions):
            try:
                emo_cpu = emotions.cpu()
                dominant_emotion_idx = torch.argmax(emo_cpu).item();
                intensity = emo_cpu[dominant_emotion_idx].item()
            except Exception as e: logger.warning(f"Particle update error getting emotion: {e}")
        buffer_idx = 0
        time_scale = delta_time * 60.0
        for i, particle in enumerate(self.particles):
            particle.position += particle.velocity * time_scale
            particle.lifetime -= delta_time
            particle.alpha = max(0.0, min(1.0, particle.lifetime / 2.0)) * (0.4 + intensity * 0.6)
            if dominant_emotion_idx == 0: particle.velocity.setY(particle.velocity.y() + 0.00015 * intensity * time_scale); particle.color.setX(min(1.0, particle.color.x() + 0.02 * intensity * time_scale)); particle.color.setY(min(1.0, particle.color.y() + 0.01 * intensity * time_scale))
            elif dominant_emotion_idx == 1: particle.velocity += QVector3D(random.uniform(-0.0008, 0.0008), random.uniform(-0.0008, 0.0008), 0) * intensity * time_scale; particle.color.setX(min(1.0, particle.color.x() + 0.02 * intensity * time_scale)); particle.color.setZ(min(1.0, particle.color.z() + 0.01 * intensity * time_scale))
            elif dominant_emotion_idx == 4: particle.velocity *= (1.0 - 0.03 * intensity * time_scale); particle.color.setZ(min(1.0, particle.color.z() + 0.02 * intensity * time_scale))
            needs_reset = (particle.lifetime <= 0 or abs(particle.position.x()) > 1.5 or particle.position.y() > 1.5 or particle.position.y() < -1.5)
            if needs_reset: self.particles[i] = self.create_particle(); particle = self.particles[i]
            if buffer_idx + 7 <= len(self.particle_buffer): self.particle_buffer[buffer_idx : buffer_idx+7] = [particle.position.x(), particle.position.y(), particle.position.z(), max(0.0, min(1.0, particle.color.x())), max(0.0, min(1.0, particle.color.y())), max(0.0, min(1.0, particle.color.z())), max(0.0, min(1.0, particle.alpha))]; buffer_idx += 7
            else: logger.warning("Particle buffer overflow detected during update."); break
        self._update_buffer()

    def _update_buffer(self):
        if self.max_count <= 0: return
        buffer_idx = 0
        for p in self.particles:
            if buffer_idx + 7 <= len(self.particle_buffer):
                self.particle_buffer[buffer_idx : buffer_idx + 7] = [p.position.x(), p.position.y(), p.position.z(), max(0.0, min(1.0, p.color.x())), max(0.0, min(1.0, p.color.y())), max(0.0, min(1.0, p.color.z())), max(0.0, min(1.0, p.alpha))]; buffer_idx += 7
            else: logger.warning("Particle buffer overflow in _update_buffer."); break


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

        # --- NEW: Monologue state ---
        self.current_monologue: str = ""
        self.monologue_sentiment: float = 0.0
        self.monologue_movement_type: Optional[str] = None # e.g., "ponder", "positive", None
        self.last_monologue_update_time: float = 0.0
        self.monologue_update_interval: float = 3.0 # How often to check monologue

        # --- NEW: Micro-movement state ---
        self.micro_movement_timer: float = 0.0
        self.micro_movement_interval: float = 0.1 # Update micro-movements frequently
        self.micro_movement_target_offset_x: float = 0.0
        self.micro_movement_target_offset_y: float = 0.0
        self.micro_movement_current_offset_x: float = 0.0
        self.micro_movement_current_offset_y: float = 0.0


        self.shader_program: Optional[QOpenGLShaderProgram] = None
        self.particle_shader: Optional[QOpenGLShaderProgram] = None
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

    # ... (_init_live2d_library, _setup_animation_timer, initializeGL, _setup_background_shader, _setup_gl_buffers, _setup_particle_buffers, _load_model, _create_parameter_mapper, resizeGL, paintGL remain the same) ...
    def _init_live2d_library(self):
        if not Live2DCharacter.live2d_initialized_global:
            try:
                live2d_v3.init()
                Live2DCharacter.live2d_initialized_global = True
                self.live2d_initialized = True
                logger.info("Live2D Core library initialized successfully.")
            except Exception as e:
                logger.critical(f"Live2D Core initialization FAILED: {e}", exc_info=True)
                self.error_occurred.emit(f"Live2D Core init failed: {e}")
                self.live2d_initialized = False
        else:
             self.live2d_initialized = True
             logger.debug("Live2D Core library already initialized globally.")

    def _setup_animation_timer(self):
        self.animation_timer = QTimer(self)
        interval = max(10, int(1000.0 / Config.Graphics.FPS))
        self.animation_timer.setInterval(interval)
        self.animation_timer.timeout.connect(self._tick)

    def initializeGL(self):
        if not self.live2d_initialized:
            logger.error("Cannot initialize GL: Live2D Core library not initialized.")
            self.character_initialized.emit(False)
            return
        try:
             self.makeCurrent()
        except Exception as e:
             logger.critical(f"Failed to make OpenGL context current in initializeGL: {e}", exc_info=True)
             self.error_occurred.emit(f"OpenGL context error: {e}")
             self.character_initialized.emit(False)
             return

        try:
            bg = Config.Graphics.BACKGROUND_COLOR; glClearColor(bg[0], bg[1], bg[2], bg[3])
            glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_PROGRAM_POINT_SIZE); glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

            try:
                live2d_v3.glInit(); logger.debug("Live2D OpenGL extensions initialized.")
            except Exception as e:
                logger.critical(f"live2d_v3.glInit() FAILED: {e}", exc_info=True)
                raise RuntimeError("Live2D GL Initialization failed.") from e

            if not self._setup_background_shader(): logger.warning("Failed to set up BG shader.")
            elif not self._setup_gl_buffers(): logger.warning("Failed to set up BG buffers.")

            if not self._setup_particle_buffers():
                logger.warning("Failed to set up particle buffers. Particles disabled.")
                self.particle_system.max_count = 0


            self._load_model()
            if self.model_loaded and self.model:
                try: self.model.Update(); logger.debug("Initial model update OK.")
                except Exception as e: logger.error(f"Error during initial model.Update() in initializeGL: {e}")

                self._create_parameter_mapper()
                self.animation_timer.start(); logger.info("Live2D character ready, timer started.")
                self.character_initialized.emit(True)
            else:
                logger.error("GL init OK, but model load failed or model is null.")
                self.character_initialized.emit(False)

        except Exception as e:
            logger.critical(f"Critical error during OpenGL initialization: {e}", exc_info=True)
            self.error_occurred.emit(f"Graphics initialization error: {e}")
            self.character_initialized.emit(False)
        finally:
            try: self.doneCurrent()
            except Exception as e_done: logger.error(f"Error calling doneCurrent in initializeGL finally: {e_done}")

    # --- Resource Setup Methods ---
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
                logger.error(f"Background Vertex Shader compile failed: {self.shader_program.log()}"); return False
            if not self.shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs_code):
                logger.error(f"Background Fragment Shader compile failed: {self.shader_program.log()}"); return False
            if not self.shader_program.link():
                logger.error(f"Background Shader link failed: {self.shader_program.log()}"); return False
            logger.debug("Background shader setup successful.")
            return True
        except Exception as e:
            logger.error(f"Error setting up background shader: {e}", exc_info=True)
            if self.shader_program: self.shader_program.release(); self.shader_program = None
            return False

    def _setup_gl_buffers(self):
        try:
            quad_vertices = np.array([-1.0,-1.0, 1.0,-1.0, 1.0,1.0, -1.0,-1.0, 1.0,1.0, -1.0,1.0], dtype=np.float32)
            self.quad_vao = glGenVertexArrays(1); glBindVertexArray(self.quad_vao)
            self.quad_vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
            glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
            position_loc = 0
            if self.shader_program and self.shader_program.isLinked():
                 loc_check = glGetAttribLocation(self.shader_program.programId(), b"position")
                 if loc_check != -1: position_loc = loc_check
                 else: logger.warning("BG Shader: 'position' attribute not found. Using default location 0.")
            else: logger.warning("BG Shader not linked when setting up buffers, using default location 0.")

            glEnableVertexAttribArray(position_loc)
            glVertexAttribPointer(position_loc, 2, GL_FLOAT, GL_FALSE, 2 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))
            glBindBuffer(GL_ARRAY_BUFFER, 0); glBindVertexArray(0)
            logger.debug("Background quad VAO/VBO setup successful.")
            return True
        except Exception as e:
            logger.error(f"Error setting up background GL buffers: {e}", exc_info=True)
            if hasattr(self, 'quad_vbo') and self.quad_vbo: glDeleteBuffers(1, [self.quad_vbo]); self.quad_vbo = 0
            if hasattr(self, 'quad_vao') and self.quad_vao: glDeleteVertexArrays(1, [self.quad_vao]); self.quad_vao = 0
            return False

    def _setup_particle_buffers(self):
        if self.particle_system.max_count <= 0: return True
        try:
            self.particle_vao = glGenVertexArrays(1); glBindVertexArray(self.particle_vao)
            self.particle_vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
            buffer_size = self.particle_system.max_count * 7 * ctypes.sizeof(GLfloat)
            glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_DYNAMIC_DRAW)
            stride = 7 * ctypes.sizeof(GLfloat)
            glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            color_offset = ctypes.c_void_p(3 * ctypes.sizeof(GLfloat))
            glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, color_offset)
            glBindBuffer(GL_ARRAY_BUFFER, 0); glBindVertexArray(0)
            logger.debug("Particle VAO/VBO created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error setting up particle GL buffers: {e}", exc_info=True)
            if hasattr(self, 'particle_vbo') and self.particle_vbo: glDeleteBuffers(1, [self.particle_vbo]); self.particle_vbo = 0
            if hasattr(self, 'particle_vao') and self.particle_vao: glDeleteVertexArrays(1, [self.particle_vao]); self.particle_vao = 0
            return False

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Model file/path not found: {self.model_path}")
            self.error_occurred.emit(f"Model path not found: {self.model_path}")
            self.model_loaded = False; return
        try:
            logger.info(f"Loading Live2D model from: {self.model_path}")
            self.model = Live2Dv3AppModel()
            w, h = self.width(), self.height()
            if w <= 0 or h <= 0: w, h = 400, 600
            load_success = self.model.LoadModelJson(self.model_path)
            self.model.Resize(w, h)
            self.model_loaded = True
            logger.info(f"Loaded model: {os.path.basename(self.model_path)}")
        except Exception as e:
            logger.error(f"Live2D model loading failed: {e}", exc_info=True)
            self.model = None; self.model_loaded = False;
            self.error_occurred.emit(f"Model load failed: {e}")

    def _create_parameter_mapper(self):
        if not self.model or not self.model_loaded: return
        logger.info("Creating parameter map using heuristic defaults.")
        self.parameter_map = {
            0: {'params': ['PARAM_BODY_ANGLE_X', 'PARAM_CHEEK', 'PARAM_MOUTH_FORM', 'PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN'], 'mapper': lambda p, x: x * 10 if 'BODY' in p else (0.5 + x * 0.5), 'smoothing': 0.05},
            1: {'params': ['PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN', 'PARAM_MOUTH_OPEN_Y', 'PARAM_BROW_L_Y', 'PARAM_BROW_R_Y'], 'mapper': lambda p, x: (1.0 - x*0.5) if 'EYE' in p else (x if 'MOUTH' in p else -x * 0.8), 'smoothing': 0.1},
            2: {'params': ['PARAM_BROW_L_Y', 'PARAM_BROW_R_Y', 'PARAM_EYE_BALL_X', 'PARAM_ANGLE_Z'], 'mapper': lambda p, x: x * 0.5 if 'BROW' in p else (x * 0.6 if 'EYE' in p else x * 15), 'smoothing': 0.1},
            3: {'params': ['PARAM_BROW_L_FORM', 'PARAM_BROW_R_FORM', 'PARAM_MOUTH_FORM', 'PARAM_4'], 'mapper': lambda p, x: -x * 0.8 if 'BROW' in p else (-x if 'MOUTH_F' in p else x), 'smoothing': 0.1},
            4: {'params': ['PARAM_BODY_ANGLE_X', 'PARAM_BREATH'], 'mapper': lambda p, x: x * 0.1, 'smoothing': 0.05},
            5: {'params': ['PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN', 'PARAM_MOUTH_OPEN_Y'], 'mapper': lambda p, x: min(1.2, x * 1.2), 'smoothing': 0.08},
            'cursor_x': {'params': ['PARAM_ANGLE_X', 'PARAM_EYE_BALL_X', 'PARAM_BODY_ANGLE_X', 'Param8', 'Param42'], 'mapper': lambda p, x: x * 30 if 'ANGLE' in p else (x * 0.8 if 'EYE' in p else x * 10), 'smoothing': 0.05},
            'cursor_y': {'params': ['PARAM_ANGLE_Y', 'PARAM_EYE_BALL_Y', 'PARAM_BODY_ANGLE_Y', 'Param12', 'Param45'], 'mapper': lambda p, x: x * 30 if 'ANGLE' in p else (x * 0.8 if 'EYE' in p else x * 10), 'smoothing': 0.05},
            'blush':     {'params': ['Param'], 'mapper': lambda p, x: x, 'smoothing': 0.1},
            'wings':     {'params': ['Param6'], 'mapper': lambda p, x: x, 'smoothing': 0.1},
            'mad':       {'params': ['Param2'], 'mapper': lambda p, x: x, 'smoothing': 0.1},
            'confusion': {'params': ['Param4'], 'mapper': lambda p, x: x, 'smoothing': 0.1}
        }
        all_params = set()
        for config_val in self.parameter_map.values():
            if isinstance(config_val, dict) and 'params' in config_val:
                all_params.update(p for p in config_val.get('params', []) if isinstance(p, str))
        all_params.update({'PARAM_BODY_ANGLE_Z', 'Param3', 'PARAM_BREATH'})

        logger.info(f"Initializing {len(all_params)} parameters identified from map/heuristics...")
        self._failed_params = set()
        self.default_values = {}
        self.target_values = {}
        self.current_values = {}
        self.parameter_velocities = {}
        model_param_ids = []
        if self.model:
            try: model_param_ids = self.model.GetParameterIds()
            except AttributeError: logger.warning("model.GetParameterIds() not available.")
            except Exception as e: logger.warning(f"Could not get parameter IDs from model: {e}.")

        for param_id in all_params:
            default_val = 0.0; found_in_model = False
            if param_id in model_param_ids and hasattr(self.model, 'GetParameterDefaultValue'):
                try: default_val = self.model.GetParameterDefaultValue(param_id); found_in_model = True
                except Exception as e: logger.warning(f"Could not get default for '{param_id}': {e}.")
            if not found_in_model:
                if param_id.endswith('_OPEN'): default_val = Config.Graphics.EYE_PARAM_DEFAULT
                elif param_id == 'PARAM_MOUTH_OPEN_Y': default_val = Config.Graphics.MOUTH_PARAM_DEFAULT
                elif param_id == 'PARAM_BREATH': default_val = 0.5

            self.default_values[param_id] = default_val
            self.target_values[param_id] = default_val
            self.current_values[param_id] = default_val
            self.parameter_velocities[param_id] = 0.0
            try:
                if hasattr(self.model, 'SetParameterValue'):
                     self.model.SetParameterValue(param_id, default_val)
            except Exception as e:
                 if param_id not in self._failed_params:
                     logger.debug(f"Initial set failed for param '{param_id}'. Err: {e}")
                     self._failed_params.add(param_id)
        logger.info("Parameter map and initial values set.")

    def resizeGL(self, width: int, height: int):
        if width <= 0 or height <= 0: return
        try: self.makeCurrent()
        except Exception as e: logger.error(f"Failed context in resizeGL: {e}"); return
        try:
            glViewport(0, 0, width, height)
            if self.model and self.model_loaded: self.model.Resize(width, height)
            if self.shader_program and self.shader_program.isLinked():
                self.shader_program.bind()
                self.shader_program.setUniformValue("resolution", float(width), float(height))
                self.shader_program.release()
        except Exception as e: logger.error(f"Error in resizeGL: {e}")
        finally: self.doneCurrent()

    def paintGL(self):
        try: self.makeCurrent()
        except Exception as e: logger.error(f"Failed context in paintGL: {e}"); return

        if not self.model_loaded or not self.model:
            bg = Config.Graphics.BACKGROUND_COLOR; glClearColor(bg[0], bg[1], bg[2], bg[3])
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            try: self.doneCurrent()
            except: pass
            return

        try:
            glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE)
            if self.shader_program and self.shader_program.isLinked() and self.quad_vao:
                self.shader_program.bind()
                self.shader_program.setUniformValue("time", self.time_elapsed)
                self.shader_program.setUniformValue("resolution", float(self.width()), float(self.height()))
                dominant_idx = 0; intensity = 0.5
                if self.emotions.numel() >= Config.Agent.EMOTION_DIM and is_safe(self.emotions):
                    try: emo_cpu = self.emotions.cpu(); dominant_idx = torch.argmax(emo_cpu).item(); intensity = emo_cpu[dominant_idx].item()
                    except Exception: pass
                colors_fallback = [(0.3, 1, 0.3), (1, 0.3, 0.3), (1, 1, 0.3), (1, 0.6, 0.3), (0.3, 0.8, 1), (1, 0.3, 1)]
                if len(colors_fallback) < Config.Agent.EMOTION_DIM:
                    colors_fallback.extend([(0.8, 0.8, 0.8)] * (Config.Agent.EMOTION_DIM - len(colors_fallback)))

                base_rgb = colors_fallback[dominant_idx % len(colors_fallback)]
                final_color = [max(0.1, c * (0.5 + intensity * 0.7)) for c in base_rgb]
                self.shader_program.setUniformValue("emotion_color", final_color[0], final_color[1], final_color[2])

                glBindVertexArray(self.quad_vao); glDrawArrays(GL_TRIANGLES, 0, 6); glBindVertexArray(0)
                self.shader_program.release()
            else:
                bg = Config.Graphics.BACKGROUND_COLOR; glClearColor(bg[0], bg[1], bg[2], bg[3]); glClear(GL_COLOR_BUFFER_BIT)

            if self.particle_system.max_count > 0 and self.particle_vao and self.particle_vbo:
                glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glEnable(GL_POINT_SMOOTH); glDepthMask(GL_FALSE)

                glBindVertexArray(self.particle_vao)
                glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
                if self.particle_system.particle_buffer is not None and self.particle_system.particle_buffer.size > 0:
                     glBufferSubData(GL_ARRAY_BUFFER, 0, self.particle_system.particle_buffer.nbytes, self.particle_system.particle_buffer)
                glBindBuffer(GL_ARRAY_BUFFER, 0)

                avg_size = np.mean([p.size for p in self.particle_system.particles]) if self.particle_system.particles else Config.Graphics.PARTICLE_MIN_SIZE
                point_pixel_size = np.clip(avg_size, Config.Graphics.PARTICLE_MIN_SIZE, Config.Graphics.PARTICLE_MAX_SIZE)
                glPointSize(point_pixel_size)

                glDrawArrays(GL_POINTS, 0, self.particle_system.max_count)

                glBindVertexArray(0)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_TRUE)

            glEnable(GL_DEPTH_TEST); glClear(GL_DEPTH_BUFFER_BIT)
            if self.model: self.model.Draw()

        except Exception as e:
            logger.error(f"OpenGL rendering error in paintGL: {e}", exc_info=True)
            self.error_occurred.emit(f"Render error: {e}")
            if hasattr(self, 'animation_timer') and self.animation_timer.isActive(): self.animation_timer.stop()
            glClearColor(0.3, 0.0, 0.0, 1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        finally:
            self.doneCurrent()

    def _tick(self):
        if not self.model_loaded or not self.model: return
        self.frame_count += 1
        interval_ms = self.animation_timer.interval()
        delta_time = interval_ms / 1000.0 if interval_ms > 0 else (1.0 / Config.Graphics.FPS)
        self.time_elapsed += delta_time

        self.particle_system.update(delta_time, self.emotions)
        self._update_animations(delta_time)

        # --- NEW: Trigger monologue analysis periodically ---
        if self.time_elapsed - self.last_monologue_update_time > self.monologue_update_interval:
             # In a real scenario, you might get the monologue from the orchestrator here
             # For now, let's just use the stored one if it exists
             if self.current_monologue:
                 self._analyze_monologue(self.current_monologue)
             self.last_monologue_update_time = self.time_elapsed

        self._update_model_parameters(delta_time) # This now includes monologue/micro-movements
        try:
            self.model.Update()
        except Exception as e: logger.error(f"Error during model.Update() in _tick: {e}", exc_info=True); return

        self.update()

    def _update_animations(self, delta_time: float):
        # ... (blink, breath, sway, idle logic remains the same) ...
        time_scale = delta_time * 60.0
        breath = self.animation_params['breath']
        breath['phase'] = (breath['phase'] + breath['speed'] * time_scale) % (2 * math.pi)
        blink = self.animation_params['blink']; blink['timer'] += delta_time
        blink_close_speed = 8.0; blink_open_speed = 6.0
        if blink['state'] == 'WAITING':
            if blink['timer'] >= blink['interval']: blink['state'] = 'CLOSING'; blink['phase'] = 0.0; blink['timer'] = 0.0
        elif blink['state'] == 'CLOSING':
            blink['phase'] = min(1.0, blink['phase'] + delta_time * blink_close_speed)
            if blink['phase'] >= 1.0: blink['state'] = 'OPENING'
        elif blink['state'] == 'OPENING':
            blink['phase'] = max(0.0, blink['phase'] - delta_time * blink_open_speed)
            if blink['phase'] <= 0.0: blink['state'] = 'WAITING'; blink['interval'] = random.uniform(1.5, 4.5); blink['timer'] = 0.0
        hair = self.animation_params['hair_sway']; hair['phase'] = (hair['phase'] + hair['speed'] * time_scale) % (2 * math.pi)
        idle = self.animation_params['idle']; idle['phase'] = (idle['phase'] + idle['speed'] * time_scale) % (2 * math.pi)
        if 'surprise' in self.animation_params:
             surprise = self.animation_params['surprise']
             surprise['timer'] = surprise.get('timer', 0.0) + delta_time
             if surprise['timer'] >= surprise.get('duration', 1.5):
                 self.animation_params.pop('surprise', None)
        fade_rate = 2.0
        if not self.is_mouse_over:
            self.toggle_states['confusion'] = max(0.0, self.toggle_states['confusion'] - delta_time * fade_rate)

        # --- NEW: Update Micro-movement Timer ---
        self.micro_movement_timer += delta_time
        if self.micro_movement_timer >= self.micro_movement_interval:
            self.micro_movement_timer = 0.0
            # Update target offsets less frequently than applying them
            if random.random() < 0.2: # 20% chance each micro-interval to pick new target
                self.micro_movement_target_offset_x = random.uniform(-0.5, 0.5) # Small angle target
                self.micro_movement_target_offset_y = random.uniform(-0.5, 0.5)

    # --- NEW: Method to receive and analyze monologue ---
    def update_monologue(self, monologue_text: str):
        """Receives the latest internal monologue and analyzes it."""
        self.current_monologue = monologue_text
        self._analyze_monologue(monologue_text)

    def _analyze_monologue(self, monologue: str):
        """Analyzes monologue sentiment and keywords."""
        if not monologue or not TEXTBLOB_AVAILABLE:
            self.monologue_sentiment = 0.0
            self.monologue_movement_type = None
            return

        # Reset movement type
        self.monologue_movement_type = None
        try:
            blob = TextBlob(monologue)
            self.monologue_sentiment = blob.sentiment.polarity # Value between -1.0 and 1.0

            # Keyword analysis
            lower_mono = monologue.lower()
            thinking_keywords = ["think", "wonder", "consider", "maybe", "perhaps", "hmm"]
            decision_keywords = ["decide", "choose", "will", "okay", "right"] # Simple examples
            positive_keywords = ["happy", "great", "good", "love", "like", "yay"]
            negative_keywords = ["sad", "bad", "worried", "nervous", "hate", "ugh"]

            if any(word in lower_mono for word in thinking_keywords):
                self.monologue_movement_type = "ponder"
            elif any(word in lower_mono for word in decision_keywords):
                 self.monologue_movement_type = "decide"
            # Use sentiment as fallback if no strong keywords match
            elif self.monologue_sentiment > 0.4: # Threshold for clear positivity
                 if any(word in lower_mono for word in positive_keywords): # Check for explicit positive words too
                      self.monologue_movement_type = "positive"
            elif self.monologue_sentiment < -0.4: # Threshold for clear negativity
                 if any(word in lower_mono for word in negative_keywords):
                      self.monologue_movement_type = "negative"

            # logger.debug(f"Monologue: '{monologue}' | Sentiment: {self.monologue_sentiment:.2f} | Type: {self.monologue_movement_type}")

        except Exception as e:
            logger.warning(f"Error analyzing monologue with TextBlob: {e}")
            self.monologue_sentiment = 0.0
            self.monologue_movement_type = None

    # --- NEW: Method to apply head movement based on monologue ---
    def _apply_monologue_movement(self, active_targets: Dict[str, float], delta_time: float) -> Dict[str, float]:
        """Calculates and applies head angle adjustments based on monologue analysis."""
        # Define target angles based on movement type
        target_angle_x = 0.0
        target_angle_y = 0.0
        current_time = self.time_elapsed # Use consistent time

        if self.monologue_movement_type == "ponder":
            target_angle_x = 3.0 * math.sin(current_time * 0.8) # Slow side tilt, smaller range
        elif self.monologue_movement_type == "decide":
            # Quick nod - use a short pulse shape? Or just brief sine
            target_angle_y = 4.0 * math.sin(current_time * 5.0) # Faster up/down
            # Fade this out quickly? For now, rely on next monologue changing it
        elif self.monologue_movement_type == "positive":
            target_angle_y = 5.0 * math.sin(current_time * 2.5) # Moderate nod
        elif self.monologue_movement_type == "negative":
            target_angle_x = -4.0 * math.sin(current_time * 1.8) # Moderate side tilt/shake

        # Apply these targets additively or selectively to relevant params
        # We add them to the *existing* target calculated from other layers (emotions, cursor)
        # Using PARAM_ANGLE_X and PARAM_ANGLE_Y
        if 'PARAM_ANGLE_X' in active_targets:
            active_targets['PARAM_ANGLE_X'] += target_angle_x
        if 'PARAM_ANGLE_Y' in active_targets:
            active_targets['PARAM_ANGLE_Y'] += target_angle_y
        # Maybe also a subtle Z rotation for pondering?
        if self.monologue_movement_type == "ponder" and 'PARAM_ANGLE_Z' in active_targets:
             active_targets['PARAM_ANGLE_Z'] += 2.0 * math.sin(current_time * 0.7)


        return active_targets

    # --- NEW: Method to apply micro-movements ---
    def _apply_micro_movements(self, active_targets: Dict[str, float], delta_time: float) -> Dict[str, float]:
        """Applies subtle random offsets for idle realism."""
        # Smoothly interpolate current offset towards target offset
        lerp_factor = 1.0 - math.exp(-delta_time / 0.1) # Faster smoothing for micro-movements
        self.micro_movement_current_offset_x += (self.micro_movement_target_offset_x - self.micro_movement_current_offset_x) * lerp_factor
        self.micro_movement_current_offset_y += (self.micro_movement_target_offset_y - self.micro_movement_current_offset_y) * lerp_factor

        # Add the current smoothed offset to target angles
        if 'PARAM_ANGLE_X' in active_targets:
             active_targets['PARAM_ANGLE_X'] += self.micro_movement_current_offset_x
        if 'PARAM_ANGLE_Y' in active_targets:
             active_targets['PARAM_ANGLE_Y'] += self.micro_movement_current_offset_y
        # Add tiny Z offset too?
        if 'PARAM_ANGLE_Z' in active_targets:
             active_targets['PARAM_ANGLE_Z'] += self.micro_movement_current_offset_x * 0.3 # Link Z slightly to X offset

        return active_targets


    # --- MODIFIED: Call monologue and micro-movement logic ---
    def _update_model_parameters(self, delta_time: float):
        if not self.model or not self.model_loaded or not self.parameter_map: return
        active_targets = self.default_values.copy()

        # --- Layer 1: Base Procedural Animations ---
        breath_value = 0.5 + 0.5 * math.sin(self.animation_params['breath']['phase'])
        if 'PARAM_BREATH' in active_targets: active_targets['PARAM_BREATH'] = breath_value
        hair_sway_val = math.sin(self.animation_params['hair_sway']['phase']) * self.animation_params['hair_sway']['magnitude']
        idle_tilt_val = math.sin(self.animation_params['idle']['phase']) * self.animation_params['idle']['magnitude']
        if 'PARAM_ANGLE_Z' in active_targets: active_targets['PARAM_ANGLE_Z'] = self.default_values.get('PARAM_ANGLE_Z', 0.0) + idle_tilt_val * 0.3
        if 'Param8' in active_targets: active_targets['Param8'] = self.default_values.get('Param8', 0.0) + hair_sway_val * 0.1


        # --- Layer 2: Emotion Influence ---
        if self.emotions.numel() >= Config.Agent.EMOTION_DIM and is_safe(self.emotions):
            emo_cpu = self.emotions.cpu()
            for emotion_idx, config_val in self.parameter_map.items():
                if not isinstance(emotion_idx, int) or not (0 <= emotion_idx < Config.Agent.EMOTION_DIM): continue
                emotion_value = emo_cpu[emotion_idx].item()
                if emotion_value > 0.05:
                     for param_id in config_val.get('params', []):
                         if param_id in active_targets:
                             active_targets[param_id] = config_val['mapper'](param_id, emotion_value)

        # --- Layer 3: Interaction Influence (Cursor) ---
        if self.is_mouse_over and self.cursor_history:
             if len(self.cursor_history) > 0:
                 avg_x = sum(x for x, _ in self.cursor_history) / len(self.cursor_history)
                 avg_y = sum(y for _, y in self.cursor_history) / len(self.cursor_history)
                 for key, loop_value in [('cursor_x', avg_x), ('cursor_y', avg_y)]:
                     config_val_lookup = self.parameter_map.get(key)
                     if config_val_lookup:
                         for param_id in config_val_lookup.get('params', []):
                             if param_id in active_targets:
                                 active_targets[param_id] = config_val_lookup['mapper'](param_id, loop_value)

        # --- Layer 4: Toggleable States ---
        for toggle_key, toggle_value in self.toggle_states.items():
             config_val = self.parameter_map.get(toggle_key)
             if config_val and toggle_value > 0.01:
                 for param_id in config_val.get('params', []):
                     if param_id in active_targets:
                          mapped_value = config_val['mapper'](param_id, toggle_value)
                          if toggle_key == 'confusion': active_targets[param_id] = active_targets.get(param_id, 0.0) + mapped_value
                          else: active_targets[param_id] = mapped_value

        # --- NEW: Layer 5: Monologue Head Movement (Additive) ---
        active_targets = self._apply_monologue_movement(active_targets, delta_time)

        # --- NEW: Layer 6: Micro-movements (Additive) ---
        active_targets = self._apply_micro_movements(active_targets, delta_time)

        # --- Layer 7: Eye Blinking (Multiplicative - Applied Last) ---
        blink_phase = self.animation_params['blink']['phase']; blink_state = self.animation_params['blink']['state']
        eye_open_value = 1.0
        if blink_state == 'CLOSING': eye_open_value = 1.0 - blink_phase
        elif blink_state == 'OPENING': eye_open_value = blink_phase
        eye_params = ['PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN']
        for param_id in eye_params:
             if param_id in active_targets:
                 active_targets[param_id] *= eye_open_value


        # --- Apply Smoothing and Set Parameter Values ---
        for param_id, target_val in active_targets.items():
            if param_id not in self.current_values: continue
            current_val = self.current_values[param_id]
            param_config = None
            for cfg in self.parameter_map.values():
                if isinstance(cfg, dict) and param_id in cfg.get('params', []): param_config = cfg; break
            smoothing_alpha = param_config.get('smoothing', 0.1) if param_config else 0.1
            lerp_factor = 1.0 - math.exp(-delta_time / max(0.001, smoothing_alpha))
            new_current = current_val + (target_val - current_val) * lerp_factor

            min_val, max_val = -1.0, 1.0
            if 'ANGLE' in param_id: min_val, max_val = -30.0, 30.0
            elif 'EYE_OPEN' in param_id or 'MOUTH_OPEN' in param_id or 'BREATH' in param_id or 'CHEEK' in param_id: min_val, max_val = 0.0, 1.0
            elif param_id in ['Param', 'Param6', 'Param2', 'Param4']: min_val, max_val = 0.0, 1.0

            clamped_new_current = max(min_val, min(max_val, new_current))
            self.current_values[param_id] = clamped_new_current

            try:
                 if hasattr(self.model, 'SetParameterValue'):
                    self.model.SetParameterValue(param_id, clamped_new_current)
                 self._failed_params.discard(param_id)
            except TypeError as te:
                 if param_id not in self._failed_params:
                     logger.warning(f"TypeError setting param '{param_id}' (val: {clamped_new_current:.3f}). SDK likely expects different arguments: {te}")
                     self._failed_params.add(param_id)
            except Exception as e:
                if param_id not in self._failed_params:
                    err_type = type(e).__name__
                    logger.warning(f"Failed set param '{param_id}' (val: {clamped_new_current:.3f}). SDK Method missing or other error: {err_type}")
                    self._failed_params.add(param_id)

    # ... (update_emotions, mouseMoveEvent, mousePressEvent, enterEvent, leaveEvent, cleanup remain the same) ...
    def update_emotions(self, emotions_tensor: torch.Tensor):
        if isinstance(emotions_tensor, torch.Tensor) and emotions_tensor.shape == (Config.Agent.EMOTION_DIM,) and is_safe(emotions_tensor):
            self.emotions = emotions_tensor.detach().to(DEVICE)
        else:
            shape_info = emotions_tensor.shape if isinstance(emotions_tensor, torch.Tensor) else 'N/A'
            logger.warning(f"Received invalid emotion data type/shape: {type(emotions_tensor)}, shape: {shape_info}")

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.model_loaded: return
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0: return
        try: pos = event.pos(); current_x = pos.x(); current_y = pos.y()
        except Exception as e: logger.error(f"Error accessing event coordinates: {e}"); return
        x_norm = (current_x / width) * 2.0 - 1.0
        y_norm = 1.0 - (current_y / height) * 2.0
        self.cursor_pos = (x_norm, y_norm)
        self.cursor_history.append(self.cursor_pos)
        self.is_mouse_over = True
        self.last_interaction_time = time.time()
        self.toggle_states['confusion'] = min(1.0, self.toggle_states['confusion'] + 0.1)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.model_loaded: return
        self.interaction_detected.emit()
        self.last_interaction_time = time.time()
        button = event.button()
        if button == Qt.MouseButton.LeftButton: self.toggle_states['blush'] = 1.0 - self.toggle_states['blush']
        elif button == Qt.MouseButton.RightButton: self.toggle_states['wings'] = 1.0 - self.toggle_states['wings']
        elif button == Qt.MouseButton.MiddleButton: self.toggle_states['mad'] = 1.0 - self.toggle_states['mad']
        self.animation_params['surprise'] = {'timer': 0.0, 'duration': 1.5}
        self.setFocus()

    def enterEvent(self, event): super().enterEvent(event); self.is_mouse_over = True
    def leaveEvent(self, event): super().leaveEvent(event); self.is_mouse_over = False; self.cursor_history.clear()

    def cleanup(self):
        logger.info("Cleaning up Live2DCharacter resources...")
        try: self.makeCurrent()
        except Exception as e: logger.error(f"Failed context for cleanup: {e}"); return
        try:
            if hasattr(self, 'animation_timer') and self.animation_timer: self.animation_timer.stop()

            if self.quad_vao: glDeleteVertexArrays(1, [self.quad_vao]); self.quad_vao = 0
            if self.quad_vbo: glDeleteBuffers(1, [self.quad_vbo]); self.quad_vbo = 0
            if self.particle_vao: glDeleteVertexArrays(1, [self.particle_vao]); self.particle_vao = 0
            if self.particle_vbo: glDeleteBuffers(1, [self.particle_vbo]); self.particle_vbo = 0

            if self.model:
                release_method = getattr(self.model, 'Release', None)
                if callable(release_method):
                    try: release_method()
                    except Exception as e_rel: logger.error(f"Error calling model.Release(): {e_rel}")
                self.model = None; self.model_loaded = False; logger.debug("Model reference released.")

            if self.shader_program: self.shader_program = None;
            if self.particle_shader: self.particle_shader = None;

        except Exception as e:
            logger.error(f"Error during OpenGL resource cleanup: {e}", exc_info=True)
        finally:
            try: self.doneCurrent()
            except Exception as e_done: logger.error(f"Error calling doneCurrent during cleanup: {e_done}")
            logger.info("Character cleanup finished.")


# --- END OF FILE graphics.py ---
