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

# Need Config, DEVICE, logger, is_safe
from config import Config, DEVICE, logger
from utils import is_safe

# Need Qt and OpenGL imports
try:
    from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QRect
    from PyQt5.QtGui import (QColor, QFont, QPainter, QPen, QOpenGLShader, QOpenGLShaderProgram,
                             QVector3D, QMouseEvent)
    from OpenGL.GL import *
    from OpenGL.GL import glGenBuffers, glBindBuffer, glBufferData, glBufferSubData, glDeleteBuffers
    from OpenGL.GL import glGenVertexArrays, glBindVertexArray, glEnableVertexAttribArray, glVertexAttribPointer, glDeleteVertexArrays
except ImportError as e:
    logger.critical(f"graphics.py: PyQt5 or PyOpenGL import failed: {e}. Please install PyQt5, PyOpenGL, PyOpenGL-accelerate.")
    raise # Re-raise to prevent silent failure

# Need Live2D imports
try:
    import live2d.v3 as live2d_v3
    from live2d.v3 import LAppModel as Live2Dv3AppModel
except ImportError as e:
    logger.critical(f"graphics.py: live2d-py import failed: {e}. Ensure live2d-py[cubism3] and native Core lib are installed/accessible.")
    raise
except Exception as core_err:
    logger.critical(f"graphics.py: An unexpected error occurred related to the Live2D Core library: {core_err}. Ensure native lib installed.")
    raise

# Need dataclass
from dataclasses import dataclass, field

# --- Particle System Components ---
@dataclass
class Particle:
    position: QVector3D = field(default_factory=lambda: QVector3D(0, 0, 0))
    velocity: QVector3D = field(default_factory=lambda: QVector3D(0, 0, 0))
    color: QVector3D = field(default_factory=lambda: QVector3D(1, 1, 1))
    lifetime: float = 0.0
    size: float = 1.0
    alpha: float = 1.0

class ParticleSystem:
    def __init__(self, count=Config.PARTICLE_COUNT):
        self.max_count = count
        if self.max_count <= 0:
            logger.warning(f"ParticleSystem count is {self.max_count}. Disabling particles."); self.max_count = 0
        self.particles = []
        self.particle_buffer = np.zeros(self.max_count * 7, dtype=np.float32) if self.max_count > 0 else np.array([], dtype=np.float32)
        self.initialize_particles()

    def initialize_particles(self):
        if self.max_count <= 0: return
        self.particles = [self.create_particle() for _ in range(self.max_count)]
        logger.debug(f"Initialized {len(self.particles)} particles.")

    def create_particle(self):
        size_factor = random.uniform(Config.PARTICLE_MIN_SIZE, Config.PARTICLE_MAX_SIZE)
        pos = QVector3D(random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2), random.uniform(-0.2, 0.2))
        vel = QVector3D(random.uniform(-0.005, 0.005), random.uniform(0.001, 0.008), random.uniform(-0.001, 0.001))
        col = QVector3D(random.uniform(0.6, 0.9), random.uniform(0.7, 1.0), random.uniform(0.8, 1.0))
        life = random.uniform(2.0, 5.0)
        return Particle(position=pos, velocity=vel, color=col, lifetime=life, size=size_factor, alpha=1.0)

    def update(self, delta_time, emotions):
        if self.max_count <= 0 or not self.particles: return

        dominant_emotion_idx = 0; intensity = 0.5
        if isinstance(emotions, torch.Tensor) and emotions.numel() >= Config.EMOTION_DIM and is_safe(emotions):
            try:
                emo_cpu = emotions.cpu()
                dominant_emotion_idx = torch.argmax(emo_cpu).item();
                intensity = emo_cpu[dominant_emotion_idx].item()
            except Exception as e:
                 logger.warning(f"Particle update error getting emotion: {e}")

        buffer_idx = 0
        time_scale = delta_time

        for i, particle in enumerate(self.particles):
            particle.position += particle.velocity * time_scale
            particle.lifetime -= delta_time
            particle.alpha = max(0.0, min(1.0, particle.lifetime / 2.0)) * 0.8

            if dominant_emotion_idx == 0: # Joy
                particle.velocity.setY(particle.velocity.y() + 0.005 * intensity * time_scale)
                particle.color.setX(min(1.0, particle.color.x() + 0.1 * intensity * time_scale))
                particle.color.setY(min(1.0, particle.color.y() + 0.05 * intensity * time_scale))
            elif dominant_emotion_idx == 1: # Fear
                particle.velocity += QVector3D(random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.005), 0) * intensity * time_scale
                particle.color.setX(min(1.0, particle.color.x() + 0.1 * intensity * time_scale))
            elif dominant_emotion_idx == 4: # Calm
                particle.velocity *= (1.0 - 0.1 * intensity * time_scale)
                particle.color.setZ(min(1.0, particle.color.z() + 0.1 * intensity * time_scale))

            needs_reset = (particle.lifetime <= 0 or
                           abs(particle.position.x()) > 1.5 or
                           particle.position.y() > 1.5 or
                           particle.position.y() < -1.5)
            if needs_reset:
                self.particles[i] = self.create_particle()
                particle = self.particles[i]

            if buffer_idx + 7 <= len(self.particle_buffer):
                self.particle_buffer[buffer_idx : buffer_idx+7] = [
                    particle.position.x(), particle.position.y(), particle.position.z(),
                    particle.color.x(), particle.color.y(), particle.color.z(), particle.alpha
                ]
            else:
                logger.warning("Particle buffer overflow detected during update.")
                break
            buffer_idx += 7

# --- Live2D Character Widget ---
class Live2DCharacter(QOpenGLWidget):
    """Widget to display the Live2D model using modern OpenGL."""
    character_initialized = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    interaction_detected = pyqtSignal()

    live2d_initialized_global = False

    def __init__(self, hud_widget=None, parent=None): # hud_widget passed in, but not used internally directly anymore
        super().__init__(parent)
        logger.info("Initializing Live2DCharacter...")
        self.model = None
        self.live2d_initialized = False
        self.model_loaded = False
        self.model_path = Config.MODEL_PATH
        self.emotions = torch.zeros(Config.EMOTION_DIM, device=DEVICE)
        self.frame_count = 0
        self.time_elapsed = 0.0
        # self.hud_widget = hud_widget # Removed direct ref, HUD color calculation is in HUDWidget itself

        self.cursor_pos = (0.0, 0.0)
        self.cursor_history = deque(maxlen=10)
        self.is_mouse_over = False
        self.last_interaction_time = time.time()
        self.toggle_states = {'blush': 0.0, 'wings': 0.0, 'mad': 0.0, 'confusion': 0.0}

        self.animation_params = {
            'breath': {'phase': 0.0, 'speed': 0.08, 'magnitude': 0.5},
            'blink': {'timer': 0.0, 'interval': random.uniform(2.0, 5.0), 'state': 'WAITING', 'phase': 0.0},
            'hair_sway': {'phase': 0.0, 'speed': 0.06, 'magnitude': 8.0},
            'idle': {'phase': 0.0, 'speed': 0.03, 'magnitude': 5.0}
        }
        self.target_values = {}
        self.current_values = {}
        self.default_values = {}
        self.parameter_velocities = {}
        self.parameter_map = {}
        self._failed_params = set()

        self.shader_program = None
        self.particle_shader = None
        self.particle_vao = None
        self.particle_vbo = None
        self.quad_vao = None
        self.quad_vbo = None

        self.particle_system = ParticleSystem()

        self.setMinimumSize(400, 600)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._init_live2d_library()
        self._setup_animation_timer()

    # --- Core Initialization Methods ---
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
        else:
            self.live2d_initialized = True

    def _setup_animation_timer(self):
        self.animation_timer = QTimer(self)
        interval = max(10, int(1000.0 / Config.FPS))
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
            bg = Config.BACKGROUND_COLOR; glClearColor(bg[0], bg[1], bg[2], bg[3])
            glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL)
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_PROGRAM_POINT_SIZE); glEnable(GL_POINT_SMOOTH)
            try:
                live2d_v3.glInit(); logger.debug("Live2D OpenGL extensions initialized.")
            except Exception as e:
                logger.critical(f"live2d_v3.glInit() FAILED: {e}", exc_info=True)
                raise RuntimeError("Live2D GL Initialization failed.") from e

            if not self._setup_background_shader(): logger.warning("Failed to set up BG shader.")
            elif not self._setup_gl_buffers(): logger.warning("Failed to set up BG buffers.")

            if not self._setup_particle_buffers():
                logger.warning("Failed to set up particle buffers. Particles disabled.")

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
            self.doneCurrent()

    # --- Resource Setup Methods ---
    def _setup_background_shader(self):
        try:
            self.shader_program = QOpenGLShaderProgram(self)
            vs_code = """#version 330 core
                layout (location = 0) in vec2 position; out vec2 fragCoord;
                void main() { gl_Position = vec4(position.x, position.y, 0.0, 1.0); fragCoord = position * 0.5 + 0.5; }"""
            fs_code = """#version 330 core
                uniform vec2 resolution; uniform float time; uniform vec3 emotion_color;
                in vec2 fragCoord; out vec4 FragColor;
                float random(vec2 st) { return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123); }
                void main() {
                    vec2 uv = fragCoord; vec3 finalColor = vec3(0.1, 0.1, 0.15);
                    float noise = random(uv * 5.0 + vec2(time * 0.1)) * 0.05; finalColor += emotion_color * noise * 2.0;
                    float vignette = smoothstep(0.8, 0.2, length(uv - 0.5)); finalColor *= vignette;
                    float dist = length(uv - 0.5); float pulse = sin(time * 1.5 + dist * 5.0) * 0.5 + 0.5;
                    float glow = smoothstep(0.5, 0.0, dist) * pulse * 0.1; finalColor += emotion_color * glow;
                    FragColor = vec4(clamp(finalColor, 0.0, 1.0), 1.0);
                } """
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
            if hasattr(self, 'quad_vbo') and self.quad_vbo: glDeleteBuffers(1, [self.quad_vbo]); self.quad_vbo = None
            if hasattr(self, 'quad_vao') and self.quad_vao: glDeleteVertexArrays(1, [self.quad_vao]); self.quad_vao = None
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
            glEnableVertexAttribArray(1); glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * ctypes.sizeof(GLfloat)))
            glBindBuffer(GL_ARRAY_BUFFER, 0); glBindVertexArray(0)
            logger.debug("Particle VAO/VBO created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error setting up particle GL buffers: {e}", exc_info=True)
            if hasattr(self, 'particle_vbo') and self.particle_vbo: glDeleteBuffers(1, [self.particle_vbo]); self.particle_vbo = None
            if hasattr(self, 'particle_vao') and self.particle_vao: glDeleteVertexArrays(1, [self.particle_vao]); self.particle_vao = None
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
            if not load_success: logger.warning("Proceeding despite LoadModelJson returning False")
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
        for config in self.parameter_map.values():
            if isinstance(config, dict) and 'params' in config:
                all_params.update(p for p in config.get('params', []) if isinstance(p, str))
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
            except Exception as e: logger.warning(f"Could not get parameter IDs from model: {e}.")

        for param_id in all_params:
            default_val = 0.0; found_in_model = False
            if param_id in model_param_ids:
                try: default_val = self.model.GetParameterDefaultValue(param_id); found_in_model = True
                except Exception as e: logger.warning(f"Could not get default for '{param_id}': {e}.")
            if not found_in_model: # Heuristic guessing
                if param_id.endswith('_OPEN'): default_val = Config.EYE_PARAM_DEFAULT
                elif param_id == 'PARAM_MOUTH_OPEN_Y': default_val = Config.MOUTH_PARAM_DEFAULT
                elif param_id == 'PARAM_BREATH': default_val = 0.5
                # ... (other heuristics omitted for brevity, assumed 0.0)

            self.default_values[param_id] = default_val
            self.target_values[param_id] = default_val
            self.current_values[param_id] = default_val
            self.parameter_velocities[param_id] = 0.0
            try: self.model.SetParameterValue(param_id, default_val)
            except Exception as e:
                 if param_id not in self._failed_params:
                     logger.debug(f"Initial set failed for param '{param_id}'. Err: {e}")
                     self._failed_params.add(param_id)
        logger.info("Parameter map and initial values set.")

    # --- Rendering and Update Methods ---
    def resizeGL(self, width, height):
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
            bg = Config.BACKGROUND_COLOR; glClearColor(bg[0], bg[1], bg[2], bg[3])
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.doneCurrent(); return
        try:
            # 1. Background
            glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE)
            if self.shader_program and self.shader_program.isLinked() and self.quad_vao:
                self.shader_program.bind()
                self.shader_program.setUniformValue("time", self.time_elapsed)
                self.shader_program.setUniformValue("resolution", float(self.width()), float(self.height()))
                # Dominant emotion color calculation (simplified for graphics.py)
                dominant_idx = 0; intensity = 0.5
                if self.emotions.numel() >= Config.EMOTION_DIM and is_safe(self.emotions):
                    try: emo_cpu = self.emotions.cpu(); dominant_idx = torch.argmax(emo_cpu).item(); intensity = emo_cpu[dominant_idx].item()
                    except Exception: pass
                # Simplified color logic without direct HUD dependency
                colors_fallback = [(0.3, 1, 0.3), (1, 0.3, 0.3), (1, 1, 0.3), (1, 0.6, 0.3), (0.3, 0.8, 1), (1, 0.3, 1)] # Normalized RGB
                base_rgb = colors_fallback[dominant_idx % len(colors_fallback)]
                final_color = [max(0.1, c * (0.5 + intensity * 0.7)) for c in base_rgb]
                self.shader_program.setUniformValue("emotion_color", final_color[0], final_color[1], final_color[2])

                glBindVertexArray(self.quad_vao); glDrawArrays(GL_TRIANGLES, 0, 6); glBindVertexArray(0)
                self.shader_program.release()
            else:
                bg = Config.BACKGROUND_COLOR; glClearColor(bg[0], bg[1], bg[2], bg[3]); glClear(GL_COLOR_BUFFER_BIT)

            # 2. Particles
            if self.particle_system.max_count > 0 and self.particle_vao and self.particle_vbo:
                glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glEnable(GL_POINT_SMOOTH); glDepthMask(GL_FALSE)
                glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
                if self.particle_system.particle_buffer is not None and self.particle_system.particle_buffer.size > 0:
                     glBufferSubData(GL_ARRAY_BUFFER, 0, self.particle_system.particle_buffer.nbytes, self.particle_system.particle_buffer)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glBindVertexArray(self.particle_vao)
                glPointSize(5.0)
                glDrawArrays(GL_POINTS, 0, self.particle_system.max_count)
                glBindVertexArray(0)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_TRUE)

            # 3. Avatar
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
        delta_time = interval_ms / 1000.0 if interval_ms > 0 else (1.0 / Config.FPS)
        self.time_elapsed += delta_time
        self.particle_system.update(delta_time, self.emotions)
        self._update_animations(delta_time)
        try: self.model.Update()
        except Exception as e: logger.error(f"Error during model.Update() in _tick: {e}", exc_info=True); return
        self._update_model_parameters(delta_time)
        self.update() # Request repaint

    def _update_animations(self, delta_time):
        time_scale = delta_time * 60.0 # Keep scaling for now

        # Breath
        breath = self.animation_params['breath']
        breath['phase'] = (breath['phase'] + breath['speed'] * time_scale) % (2 * math.pi)

        # Blink
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

        # Hair Sway / Idle Tilt
        hair = self.animation_params['hair_sway']; hair['phase'] = (hair['phase'] + hair['speed'] * time_scale) % (2 * math.pi)
        idle = self.animation_params['idle']; idle['phase'] = (idle['phase'] + idle['speed'] * time_scale) % (2 * math.pi)

        # Surprise Timer
        if 'surprise' in self.animation_params:
             surprise = self.animation_params['surprise']
             surprise['timer'] = surprise.get('timer', 0.0) + delta_time
             if surprise['timer'] >= surprise.get('duration', 1.5):
                 self.animation_params.pop('surprise', None)

        # Confusion Fade
        fade_rate = 2.0
        if not self.is_mouse_over:
            self.toggle_states['confusion'] = max(0.0, self.toggle_states['confusion'] - delta_time * fade_rate)

    def _update_model_parameters(self, delta_time):
        if not self.model or not self.model_loaded or not self.parameter_map: return
        active_targets = self.default_values.copy()

        # Layer 1: Procedural
        breath_value = 0.5 + 0.5 * math.sin(self.animation_params['breath']['phase'])
        if 'PARAM_BREATH' in active_targets: active_targets['PARAM_BREATH'] = breath_value
        hair_sway_val = math.sin(self.animation_params['hair_sway']['phase']) * self.animation_params['hair_sway']['magnitude']
        idle_tilt_val = math.sin(self.animation_params['idle']['phase']) * self.animation_params['idle']['magnitude']
        if 'PARAM_ANGLE_Z' in active_targets: active_targets['PARAM_ANGLE_Z'] = self.default_values.get('PARAM_ANGLE_Z', 0.0) + idle_tilt_val * 0.3
        if 'Param8' in active_targets: active_targets['Param8'] = self.default_values.get('Param8', 0.0) + hair_sway_val * 0.1

        # Layer 2: Emotions
        if self.emotions.numel() >= Config.EMOTION_DIM and is_safe(self.emotions):
            emo_cpu = self.emotions.cpu()
            for emotion_idx, config in self.parameter_map.items():
                if not isinstance(emotion_idx, int) or not (0 <= emotion_idx < Config.EMOTION_DIM): continue
                emotion_value = emo_cpu[emotion_idx].item()
                if emotion_value > 0.05:
                     for param_id in config.get('params', []):
                         if param_id in active_targets:
                             active_targets[param_id] = config['mapper'](param_id, emotion_value)

        # Layer 3: Interaction
        if self.is_mouse_over and self.cursor_history:
             if len(self.cursor_history) > 0:
                 avg_x = sum(x for x, _ in self.cursor_history) / len(self.cursor_history)
                 avg_y = sum(y for _, y in self.cursor_history) / len(self.cursor_history)
                 for key, value in [('cursor_x', avg_x), ('cursor_y', avg_y)]:
                     config = self.parameter_map.get(key)
                     if config:
                         for param_id in config.get('params', []):
                             if param_id in active_targets:
                                 active_targets[param_id] = config['mapper'](param_id, value)

        # Layer 4: Toggles
        for toggle_key, toggle_value in self.toggle_states.items():
             config = self.parameter_map.get(toggle_key)
             if config and toggle_value > 0.01:
                 for param_id in config.get('params', []):
                     if param_id in active_targets:
                          mapped_value = config['mapper'](param_id, toggle_value)
                          if toggle_key == 'confusion': active_targets[param_id] = active_targets.get(param_id, 0.0) + mapped_value
                          else: active_targets[param_id] = mapped_value

        # Layer 5: Blinking
        blink_phase = self.animation_params['blink']['phase']; blink_state = self.animation_params['blink']['state']
        eye_open_value = 1.0
        if blink_state == 'CLOSING': eye_open_value = 1.0 - blink_phase
        elif blink_state == 'OPENING': eye_open_value = blink_phase
        eye_params = ['PARAM_EYE_L_OPEN', 'PARAM_EYE_R_OPEN']
        for param_id in eye_params:
             if param_id in active_targets:
                 active_targets[param_id] *= eye_open_value

        # Apply Smoothing & Set Values
        for param_id, target_val in active_targets.items():
            if param_id not in self.current_values: continue
            current_val = self.current_values[param_id]
            param_config = None
            for cfg in self.parameter_map.values():
                if isinstance(cfg, dict) and param_id in cfg.get('params', []): param_config = cfg; break
            smoothing_alpha = param_config.get('smoothing', 0.1) if param_config else 0.1
            lerp_factor = min(1.0, delta_time / max(0.01, smoothing_alpha * 2.0))
            new_current = current_val + (target_val - current_val) * lerp_factor
            self.current_values[param_id] = new_current
            try:
                 self.model.SetParameterValue(param_id, new_current) # FIXED API CALL
                 self._failed_params.discard(param_id)
            except Exception as e:
                if param_id not in self._failed_params:
                    err_type = type(e).__name__
                    logger.warning(f"Failed set param '{param_id}' (val: {new_current:.3f}). Err: {err_type}")
                    self._failed_params.add(param_id)

    # --- External Update and Interaction Handlers ---
    def update_emotions(self, emotions_tensor):
        if isinstance(emotions_tensor, torch.Tensor) and emotions_tensor.shape == (Config.EMOTION_DIM,) and is_safe(emotions_tensor):
            self.emotions = emotions_tensor.detach().to(DEVICE)
        else:
            logger.warning(f"Received invalid emotion data type/shape")

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

    # --- Cleanup ---
    def cleanup(self):
        logger.info("Cleaning up Live2DCharacter resources...")
        try: self.makeCurrent()
        except Exception as e: logger.error(f"Failed context for cleanup: {e}"); return
        try:
            if hasattr(self, 'animation_timer') and self.animation_timer: self.animation_timer.stop()
            if hasattr(self, 'quad_vao') and self.quad_vao: glDeleteVertexArrays(1, [self.quad_vao]); self.quad_vao = None
            if hasattr(self, 'quad_vbo') and self.quad_vbo: glDeleteBuffers(1, [self.quad_vbo]); self.quad_vbo = None
            if hasattr(self, 'particle_vao') and self.particle_vao: glDeleteVertexArrays(1, [self.particle_vao]); self.particle_vao = None
            if hasattr(self, 'particle_vbo') and self.particle_vbo: glDeleteBuffers(1, [self.particle_vbo]); self.particle_vbo = None
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