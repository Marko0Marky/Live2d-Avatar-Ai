# --- START OF FILE gui_widgets.py ---

import torch
import math
import html

# Need Config, logger
from config import Config, logger
from typing import Optional, Dict

# Need Qt imports
try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
                                 QGroupBox, QTextEdit, QPushButton, QMessageBox)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect
    from PyQt5.QtGui import QFont, QPainter, QPen, QColor
except ImportError as e:
    logger.critical(f"gui_widgets.py: PyQt5 import failed: {e}. Please install PyQt5.")
    raise

# Need is_safe (optional, but good practice)
from utils import is_safe

# Forward declaration for type hinting
class EnhancedConsciousAgent: pass

class HUDWidget(QWidget):
    """Overlays key information (emotions, metrics, response) directly on the avatar view."""
    def __init__(self, agent_orchestrator: 'EnhancedConsciousAgent', parent=None):
        super().__init__(parent)
        # Directly access env from orchestrator to get names
        if not hasattr(agent_orchestrator, 'env'):
             raise AttributeError("HUDWidget requires agent_orchestrator to have an 'env' attribute.")
        self.agent_orchestrator = agent_orchestrator # Keep ref if needed later, though env used directly now
        self.response = "Initializing..."
        try:
            self.emotion_names = self.agent_orchestrator.env.emotion_names
            if len(self.emotion_names) != Config.EMOTION_DIM:
                 logger.warning(f"HUD: Emotion name/DIM mismatch. Using defaults.")
                 self.emotion_names = [f"Emo{i+1}" for i in range(Config.EMOTION_DIM)]
        except Exception as e:
            logger.warning(f"HUD: Could not get emotion names from env: {e}. Using defaults.");
            self.emotion_names = [f"Emo{i+1}" for i in range(Config.EMOTION_DIM)]

        self.emotion_values: torch.Tensor = torch.zeros(Config.EMOTION_DIM)
        self.metrics: Dict[str, float] = {}

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground);
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.font = QFont("Consolas", 9);
        self.glow_pen = QPen(QColor(0,0,0,0), 1); self.glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap); self.glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self.text_pen = QPen(QColor(230, 230, 255, 210), 1);
        self.response_pen = QPen(QColor(255, 220, 100, 230), 1)

        # Base glow colors (copied from original)
        self._glow_colors_rgb = [(76, 255, 76), (255, 76, 76), (255, 255, 76), (255, 153, 76), (76, 200, 255), (255, 76, 255)];
        num_colors_needed = Config.EMOTION_DIM
        if len(self._glow_colors_rgb) < num_colors_needed: self._glow_colors_rgb.extend([(200,200,200)]*(num_colors_needed - len(self._glow_colors_rgb)))
        elif len(self._glow_colors_rgb) > num_colors_needed: self._glow_colors_rgb = self._glow_colors_rgb[:num_colors_needed]

        self.pulse_phase = 0.0;
        self.pulse_timer = QTimer(self);
        self.pulse_timer.timeout.connect(self.update_pulse);
        self.pulse_timer.start(1000 // 60)

    def update_pulse(self):
        self.pulse_phase = (self.pulse_phase + 0.08) % (2 * math.pi);
        self.update()

    def update_hud(self, emotions, response, metrics_dict):
        self.response = response if response else ""
        if emotions is not None and isinstance(emotions, torch.Tensor) and emotions.shape == (Config.EMOTION_DIM,) and is_safe(emotions):
            self.emotion_values = emotions.detach().cpu()
        else:
            if not torch.equal(self.emotion_values, torch.zeros(Config.EMOTION_DIM)):
                logger.warning(f"HUD received invalid emotions. Resetting display.")
                self.emotion_values = torch.zeros(Config.EMOTION_DIM)

        self.metrics = {
            "Att": metrics_dict.get("att_score", 0.0),
            "Rho": metrics_dict.get("rho_score", 0.0),
            "Loss": metrics_dict.get("loss_value", 0.0),
            "Box": metrics_dict.get("box_score", 0.0)
            }

    def paintEvent(self, event):
        painter = QPainter(self);
        painter.setRenderHint(QPainter.RenderHint.Antialiasing);
        painter.setFont(self.font)
        padding = 10
        rect = self.rect().adjusted(padding, padding, -padding, -padding);
        if not rect.isValid(): return

        x_pos = rect.left(); y_pos = rect.top(); line_height = 14

        dominant_idx = 0; intensity = 0.0
        if self.emotion_values.numel() >= Config.EMOTION_DIM:
            try: dominant_idx = torch.argmax(self.emotion_values).item(); intensity = self.emotion_values[dominant_idx].item()
            except Exception: pass

        pulse = 1.0 + 0.3 * math.sin(self.pulse_phase) * intensity;
        glow_alpha = int(min(1.0, intensity * Config.GLOW_INTENSITY * pulse) * 80)

        if glow_alpha > 10:
            glow_color_rgb = self._glow_colors_rgb[dominant_idx % len(self._glow_colors_rgb)];
            base_glow_color = QColor(*glow_color_rgb, glow_alpha);
            self.glow_pen.setColor(base_glow_color);
            self.glow_pen.setWidth(int(3 * pulse))
        else:
            self.glow_pen.setColor(QColor(0,0,0,0)); self.glow_pen.setWidth(1)

        def draw_text_with_glow(p, x, y, text, base_pen):
            if self.glow_pen.color().alpha() > 10:
                p.setPen(self.glow_pen); p.drawText(x + 1, y + 1, text)
            p.setPen(base_pen); p.drawText(x, y, text)

        # Draw Emotion Bars
        bar_width = min(70, rect.width() - 50); bar_height = 5; bar_spacing = 2; y_emo_start = y_pos
        for i, name in enumerate(self.emotion_names):
            value = self.emotion_values[i].item() if i < self.emotion_values.numel() else 0.0
            current_y = y_emo_start + i * (bar_height + bar_spacing + line_height)
            label_y = current_y + line_height - 4
            bar_x = x_pos + 30; bar_y = label_y + 3
            if bar_y + bar_height > rect.bottom(): break
            draw_text_with_glow(painter, x_pos, label_y, f"{name[:3].upper()}", self.text_pen)
            painter.setPen(Qt.PenStyle.NoPen);
            painter.setBrush(QColor(50, 50, 70, 150));
            painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 1, 1)
            fill_width = int(value * bar_width);
            fill_color_rgb = self._glow_colors_rgb[i % len(self._glow_colors_rgb)]
            fill_color = QColor(*fill_color_rgb, 200);
            painter.setBrush(fill_color);
            painter.drawRoundedRect(bar_x, bar_y, fill_width, bar_height, 1, 1)
        y_pos = bar_y + bar_height + int(line_height * 0.8)

        # Draw Metrics
        if y_pos < rect.bottom() - line_height:
            metrics_text = " | ".join([f"{k}:{v:.2f}" for k, v in self.metrics.items()])
            draw_text_with_glow(painter, x_pos, y_pos, metrics_text, self.text_pen);
            y_pos += line_height

        # Draw Agent Response
        if self.response and y_pos < rect.bottom():
            response_text = f"> {self.response}";
            response_rect = QRect(x_pos, y_pos, rect.width(), rect.bottom() - y_pos)
            response_flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap
            if self.glow_pen.color().alpha() > 10:
                 painter.setPen(self.glow_pen)
                 painter.drawText(response_rect.adjusted(1, 1, 1, 1), response_flags, response_text)
            painter.setPen(self.response_pen)
            painter.drawText(response_rect, response_flags, response_text)


class AIStateWidget(QWidget):
    """Displays agent's core stats like emotions (bars) and aggregate metrics (text)."""
    request_completeness_test = pyqtSignal()

    def __init__(self, agent_orchestrator: 'EnhancedConsciousAgent', parent=None):
        super().__init__(parent)
        if not hasattr(agent_orchestrator, 'env'):
             raise AttributeError("AIStateWidget requires agent_orchestrator to have an 'env' attribute.")
        self.agent_orchestrator = agent_orchestrator;
        self.setMinimumWidth(300); self.setMaximumWidth(450)

        layout = QVBoxLayout(); layout.setSpacing(8); layout.setContentsMargins(10, 10, 10, 10)

        self.status_label = QLabel("Status: Initializing...");
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter);
        layout.addWidget(self.status_label)

        emotion_group = QGroupBox("Emotions");
        emotion_layout = QVBoxLayout(); emotion_layout.setSpacing(4);
        emotion_group.setLayout(emotion_layout)
        self.emotion_labels = []; self.emotion_bars = []

        try: emotions_list = self.agent_orchestrator.env.emotion_names
        except Exception as e:
            logger.warning(f"AIState: Could not get emotion names: {e}");
            emotions_list = [f"Emo{i+1}" for i in range(Config.EMOTION_DIM)]
        if len(emotions_list) != Config.EMOTION_DIM:
             logger.warning(f"AIState: Emotion name/DIM mismatch. Using defaults.")
             emotions_list = [f"Emo{i+1}" for i in range(Config.EMOTION_DIM)]

        hud_colors_rgb = [(76, 255, 76), (255, 76, 76), (255, 255, 76), (255, 153, 76), (76, 200, 255), (255, 76, 255)];
        num_colors_needed = Config.EMOTION_DIM
        if len(hud_colors_rgb) < num_colors_needed: hud_colors_rgb.extend([(200,200,200)]*(num_colors_needed - len(hud_colors_rgb)))
        elif len(hud_colors_rgb) > num_colors_needed: hud_colors_rgb = hud_colors_rgb[:num_colors_needed]
        colors_hex = [QColor(*rgb).name() for rgb in hud_colors_rgb]

        for i, emotion in enumerate(emotions_list):
            hlayout = QHBoxLayout();
            label = QLabel(f"{emotion}:"); label.setFixedWidth(75);
            bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(0);
            bar.setTextVisible(True); bar.setFormat(f"%p%")
            bar_color = colors_hex[i % len(colors_hex)]
            bar.setStyleSheet(f""" QProgressBar {{ border: 1px solid #555; border-radius: 4px; background: #2a2a3a; height: 14px; text-align: center; color: #FFFFFF; font-size: 9px; font-weight: bold; }} QProgressBar::chunk {{ background-color: {bar_color}; border-radius: 3px; margin: 1px; }} """)
            self.emotion_labels.append(label); self.emotion_bars.append(bar);
            hlayout.addWidget(label); hlayout.addWidget(bar);
            emotion_layout.addLayout(hlayout)
        layout.addWidget(emotion_group)

        stats_group = QGroupBox("Agent Metrics");
        stats_layout = QVBoxLayout(); stats_group.setLayout(stats_layout)
        self.stats_label = QLabel("Initializing metrics...");
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop);
        self.stats_label.setWordWrap(True);
        stats_font = QFont("Consolas", 9); self.stats_label.setFont(stats_font);
        stats_layout.addWidget(self.stats_label);
        layout.addWidget(stats_group)

        completeness_layout = QHBoxLayout();
        self.completeness_label = QLabel("Completeness: N/A");
        self.completeness_button = QPushButton("Run Test");
        self.completeness_button.setToolTip("Manually run Syntrometrie completeness test");
        self.completeness_button.clicked.connect(self.trigger_completeness_test)
        completeness_layout.addWidget(self.completeness_label);
        completeness_layout.addStretch();
        completeness_layout.addWidget(self.completeness_button);
        layout.addLayout(completeness_layout)

        self.setLayout(layout)
        self.setStyleSheet(""" AIStateWidget { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #282838, stop:1 #181828); border-radius: 8px; border: 1px solid #4a4a5a; } QGroupBox { font-family: "Segoe UI", Arial; font-size: 11px; font-weight: bold; color: #00bcd4; background: rgba(40, 40, 60, 0.7); border: 1px solid #3a3a4a; border-radius: 5px; margin-top: 8px; padding: 12px 5px 5px 5px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 1px 4px 1px 4px; background-color: #303040; border-radius: 3px; } QLabel { color: #cccccc; font-size: 10px; background-color: transparent; border: none; } QLabel#stats_label { color: #d0d0d0; font-size: 9px; padding: 4px; } QLabel#status_label { font-weight: bold; color: #FFA500; font-family: 'Segoe UI', Arial; font-size: 13px; padding: 5px; } QLabel#completeness_label { font-weight: bold; color: #ccc; } QPushButton { font-size: 10px; padding: 3px 8px; background-color: #007B8C; color: white; border-radius: 3px; border: 1px solid #005f6b; } QPushButton:hover { background-color: #009CB0; } QPushButton:pressed { background-color: #005f6b; } """)
        self.stats_label.setObjectName("stats_label")
        self.status_label.setObjectName("status_label")
        self.completeness_label.setObjectName("completeness_label")

        self.completeness_result = None
        self.completeness_details = ""

    def trigger_completeness_test(self):
        logger.info("Completeness test button clicked, emitting request signal.");
        self.request_completeness_test.emit()

    def update_display(self, stats, emotions, loss=0.0):
        if emotions is not None and isinstance(emotions, torch.Tensor) and emotions.shape == (Config.EMOTION_DIM,) and is_safe(emotions):
             emo_cpu = emotions.detach().cpu();
             for i, bar in enumerate(self.emotion_bars):
                 if i < len(emo_cpu): bar.setValue(int(emo_cpu[i].item() * 100))
                 else: bar.setValue(0)
        else:
             if not all(bar.value() == 0 for bar in self.emotion_bars):
                 logger.warning("AIStateWidget received invalid emotions. Resetting bars.")
                 for bar in self.emotion_bars: bar.setValue(0)

        episodes = stats.get("episode", 0); total_steps = stats.get("total_steps", 0);
        avg_reward = stats.get("avg_reward_last20", 0.0)
        I_S = stats.get("I_S", 0.0); rho_struct = stats.get("rho_struct", 0.0); att_score = stats.get("att_score", 0.0);
        self_consistency = stats.get("self_consistency", 0.0); rho_score = stats.get("rho_score", 0.0); tau_t = stats.get("tau_t", 0.0);
        box_score = stats.get("box_score", 0.0); R_acc = stats.get("R_acc", 0.0)
        stats_text = (
            f"Steps: {total_steps:<7,} Episode: {episodes:<5} Avg Rew: {avg_reward:<+8.3f}\n"
            f"--------------------------------------------\n"
            f" I_S    : {I_S:<8.3f} | Att    : {att_score:<8.3f}\n"
            f" RhoStr : {rho_struct:<8.3f} | SelfC  : {self_consistency:<+8.3f}\n"
            f" RhoScr : {rho_score:<8.3f} | Tau(t) : {tau_t:<8.3f}\n"
            f" BoxScr : {box_score:<8.3f} | R_Acc  : {R_acc:<8.3f}\n"
            f" Loss   : {loss:<+8.4f}" )
        self.stats_label.setText(stats_text)

    def update_completeness_display(self, result, details):
        self.completeness_result = result;
        self.completeness_details = details if details else "No details provided."
        if self.completeness_result is True:
            self.completeness_label.setText("Completeness: ✅ Yes");
            self.completeness_label.setStyleSheet("#completeness_label { font-weight: bold; color: #4CAF50; }");
        elif self.completeness_result is False:
            self.completeness_label.setText("Completeness: ❌ No");
            self.completeness_label.setStyleSheet("#completeness_label { font-weight: bold; color: #F44336; }");
        else:
            self.completeness_label.setText("Completeness: N/A");
            self.completeness_label.setStyleSheet("#completeness_label { font-weight: bold; color: #ccc; }");
        self.completeness_label.setToolTip(self.completeness_details)

    def set_status(self, status_text, color_hex="#FFA500"):
        self.status_label.setText(f"Status: {status_text}")
        self.status_label.setStyleSheet(f"""QLabel#status_label {{ font-weight: bold; color: {color_hex}; font-family: 'Segoe UI', Arial; font-size: 13px; border: none; background: transparent; padding: 5px; }}""")

# --- END OF FILE gui_widgets.py ---