# --- START OF FILE main_gui.py ---
import html
import logging
import torch
from typing import Dict, List, Optional, TYPE_CHECKING, Union
import os # Added for save/load path checks
import time # Added for sleep in save/load handlers

from config import MasterConfig as Config
from config import logger, DEVICE, log_file
# Import save/load paths
from config import AGENT_SAVE_PATH, GPT_SAVE_PATH, OPTIMIZER_SAVE_PATH, REPLAY_BUFFER_SAVE_PATH
try:
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
                                 QSlider, QLabel, QTextEdit, QSizePolicy, QMessageBox,
                                 QPushButton, QLineEdit, QApplication) # Added QApplication
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QEvent
    from PyQt5.QtGui import QKeyEvent, QCloseEvent
except ImportError as e: logger.critical(f"main_gui.py: PyQt5 import failed: {e}."); raise

if TYPE_CHECKING:
    from orchestrator import EnhancedConsciousAgent, ReflectReturnType

from graphics import Live2DCharacter
from gui_widgets import HUDWidget, AIStateWidget
from utils import is_safe

class EnhancedGameGUI(QMainWindow):
    def __init__(self, agent_orchestrator: 'EnhancedConsciousAgent'):
        super().__init__()
        if not hasattr(agent_orchestrator, 'env') or not hasattr(agent_orchestrator, 'avatar'): raise AttributeError("Requires agent with 'env' and 'avatar'.")
        self.agent = agent_orchestrator
        self.last_reward: float = 0.0; self.last_loss: float = 0.0; self.paused: bool = True
        try: self.intensities: List[float] = [float(i) for i in self.agent.env.base_intensities.cpu().tolist()]; self.event_freq: float = float(Config.Env.EVENT_FREQ)
        except Exception as e: logger.error(f"Failed get init env params: {e}"); self.intensities=[0.5]*Config.Agent.EMOTION_DIM; self.event_freq=0.3
        self.completeness_check_interval: int = 200; self.steps_since_last_check: int = 0
        window_title = f"Syntrometrie Conscious Agent (VrAvatar Ai5 - Chat - {DEVICE})"; self.setWindowTitle(window_title); self.setGeometry(100, 100, 1400, 800); self.setStyleSheet("QMainWindow { background-color: #181828; }")
        # Layout setup
        main_widget=QWidget(); self.setCentralWidget(main_widget); main_layout=QHBoxLayout(main_widget); main_layout.setContentsMargins(5,5,5,5); main_layout.setSpacing(8)
        avatar_container=QWidget(); avatar_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); avatar_layout=QVBoxLayout(avatar_container); avatar_layout.setContentsMargins(0,0,0,0)
        self.avatar_widget = self.agent.avatar
        if not isinstance(self.avatar_widget, Live2DCharacter): raise TypeError("agent_orchestrator did not provide Live2DCharacter.")
        self.avatar_widget.setParent(avatar_container); avatar_layout.addWidget(self.avatar_widget)
        self.hud_widget=HUDWidget(self.agent, avatar_container); self.agent.set_hud_widget(self.hud_widget); self.hud_widget.setGeometry(15, 15, 250, 280); self.hud_widget.raise_()
        main_layout.addWidget(avatar_container, 7)
        right_panel_widget=QWidget(); right_panel_layout=QVBoxLayout(right_panel_widget); right_panel_layout.setContentsMargins(5, 0, 5, 0); right_panel_layout.setSpacing(10)
        self.state_widget=AIStateWidget(self.agent); right_panel_layout.addWidget(self.state_widget)
        self.control_group=self._create_control_group(); right_panel_layout.addWidget(self.control_group)
        # --- ADDED: Save/Load Button Layout ---
        save_load_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Agent")
        self.load_button = QPushButton("Load Agent")
        self.save_button.setStyleSheet("QPushButton { font-size: 10px; padding: 4px 10px; background-color: #2a5d65; color: white; border-radius: 3px; border: 1px solid #1a3a40; } QPushButton:hover { background-color: #3a7d85; } QPushButton:pressed { background-color: #1a3a40; }")
        self.load_button.setStyleSheet("QPushButton { font-size: 10px; padding: 4px 10px; background-color: #5d4037; color: white; border-radius: 3px; border: 1px solid #3e2723; } QPushButton:hover { background-color: #7d5a4f; } QPushButton:pressed { background-color: #3e2723; }")
        self.save_button.clicked.connect(self._save_agent_state)
        self.load_button.clicked.connect(self._load_agent_state)
        save_load_layout.addWidget(self.save_button)
        save_load_layout.addWidget(self.load_button)
        right_panel_layout.addLayout(save_load_layout) # Add below controls
        # --- END ADDED ---
        self.chat_group=self._create_chat_group(); right_panel_layout.addWidget(self.chat_group, 1) # Chat group takes remaining space
        right_panel_widget.setMinimumWidth(340); right_panel_widget.setMaximumWidth(480); right_panel_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding); main_layout.addWidget(right_panel_widget, 3)
        self.timer=QTimer(self); self.timer.timeout.connect(self.update_game)
        # Signal connections
        self.avatar_widget.character_initialized.connect(self._handle_avatar_init); self.avatar_widget.error_occurred.connect(self._handle_avatar_error); self.state_widget.request_completeness_test.connect(self._run_manual_completeness_test)
        if hasattr(self, 'chat_input'): self.chat_input.returnPressed.connect(self.send_message); logger.debug("Connected chat_input returnPressed signal.")
        else: logger.error("CRITICAL: chat_input widget not found after _create_chat_group!")
        logger.info("EnhancedGameGUI initialized. Waiting for character init signal."); self.state_widget.set_status("Initializing Character...", "#FFA500")

    def _handle_avatar_init(self, success: bool):
        """Handles the signal indicating avatar initialization is complete."""
        if success:
            logger.info("Character initialized successfully. GUI Ready (Paused).")
            self.state_widget.set_status("Ready (Paused)", "#FFA500")
            # Start the main timer only after successful init
            if not self.timer.isActive():
                interval=max(10, int(1000.0/Config.Graphics.FPS))
                self.timer.start(interval)
                logger.info(f"Main update timer started ({interval}ms). Simulation initially paused.")
        else:
            logger.error("Character initialization failed.");
            self.state_widget.set_status("Character Init Failed!", "#F44336")
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize Live2D character.\nCheck model path '{Config.Graphics.MODEL_PATH}', Live2D Core library installation, and log file '{log_file}'.");
            self.paused=True; # Ensure paused if init fails
            if self.timer.isActive(): self.timer.stop()

    def _handle_avatar_error(self, error_message: str):
        """Handles critical errors reported by the avatar widget."""
        logger.critical(f"Live2D Character runtime error: {error_message}");
        self._pause_simulation(force_pause=True); # Force pause
        self.state_widget.set_status(f"Character Error!", "#F44336");
        QMessageBox.critical(self, "Runtime Error", f"A critical error occurred in the Live2D character:\n{error_message}\nSimulation paused. Check log file '{log_file}' for details.")

    def _create_control_group(self) -> QGroupBox:
        """Creates the QGroupBox containing environment control sliders."""
        group = QGroupBox("Environment Control"); group.setStyleSheet(""" QGroupBox { font-size: 11px; color: #00bcd4; border: 1px solid #3a3a4a; border-radius: 5px; margin-top: 6px; padding: 8px 5px 5px 5px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px 0 3px; background-color: #303040; border-radius: 3px; } QLabel { color: #ccc; font-size: 10px; background: transparent; border:none; } QSlider::groove:horizontal { height: 4px; background: #444; border-radius: 2px; margin: 2px 5px; } QSlider::handle:horizontal { background: #00bcd4; border: 1px solid #008C9E; width: 10px; height: 10px; border-radius: 5px; margin: -4px 0; } QSlider::sub-page:horizontal { background: #008C9E; border-radius: 2px; } QLabel#ValueLabel { font-weight: bold; color: #e0e0e0; font-size: 10px; min-width: 35px; qproperty-alignment: AlignRight; padding-right: 3px; } """)
        layout = QVBoxLayout(); layout.setSpacing(5);
        # Event Frequency
        freq_layout = QHBoxLayout(); freq_label = QLabel("Event Freq:"); freq_label.setFixedWidth(80); self.freq_slider = QSlider(Qt.Orientation.Horizontal); self.freq_slider.setRange(0, 100); self.freq_slider.setValue(int(self.event_freq*100)); self.freq_slider.valueChanged.connect(self._update_event_freq); self.freq_value_label = QLabel(f"{self.event_freq:.2f}"); self.freq_value_label.setObjectName("ValueLabel"); freq_layout.addWidget(freq_label); freq_layout.addWidget(self.freq_slider); freq_layout.addWidget(self.freq_value_label); layout.addLayout(freq_layout)
        # Emotion Intensities
        self.intensity_sliders: List[QSlider] =[]; self.intensity_labels: List[QLabel] =[]
        try: emotion_names=self.agent.env.emotion_names
        except Exception: emotion_names=[f"Emo{i+1}" for i in range(Config.Agent.EMOTION_DIM)]
        for i, emotion in enumerate(emotion_names):
             if i >= Config.Agent.EMOTION_DIM: break # Safety break if env has more names than config dim
             emo_layout = QHBoxLayout(); emo_label = QLabel(f"{emotion} Int:"); emo_label.setFixedWidth(80); emo_slider = QSlider(Qt.Orientation.Horizontal); emo_slider.setRange(0, 100); initial_intensity = int(self.intensities[i]*100) if i < len(self.intensities) else 50; emo_slider.setValue(initial_intensity); emo_slider.valueChanged.connect(lambda value, idx=i: self._update_intensity(idx, value)); intensity_val_str = f"{self.intensities[i]:.2f}" if i < len(self.intensities) else "0.50"; emo_value_label = QLabel(intensity_val_str); emo_value_label.setObjectName("ValueLabel"); emo_layout.addWidget(emo_label); emo_layout.addWidget(emo_slider); emo_layout.addWidget(emo_value_label); layout.addLayout(emo_layout); self.intensity_sliders.append(emo_slider); self.intensity_labels.append(emo_value_label)
        group.setLayout(layout); return group

    def _create_chat_group(self) -> QGroupBox:
        """Creates the QGroupBox containing the chat display and input."""
        group = QGroupBox("Agent Dialogue"); group.setStyleSheet(""" QGroupBox { font-size: 11px; color: #00bcd4; border: 1px solid #3a3a4a; border-radius: 5px; margin-top: 6px; padding-top: 10px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px 0 3px; background-color: #303040; border-radius: 3px; } """)
        layout = QVBoxLayout(); layout.setContentsMargins(0,0,0,0); layout.setSpacing(5);
        self.chat_display = QTextEdit(); self.chat_display.setReadOnly(True); self.chat_display.setStyleSheet(""" QTextEdit { background-color: rgba(20, 20, 35, 0.8); color: #e8e8e8; border: 1px solid #444; border-radius: 4px; font-family: "Segoe UI", Consolas, monospace; font-size: 10px; padding: 5px; } """); self.chat_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded); layout.addWidget(self.chat_display, 1);
        self.chat_input = QLineEdit(); self.chat_input.setPlaceholderText("Type message and press Enter..."); self.chat_input.setStyleSheet("""QLineEdit { background-color: #2a2a3a; color: #e0e0e0; border: 1px solid #555; border-radius: 4px; padding: 4px; font-size: 10px; }"""); layout.addWidget(self.chat_input, 0);
        group.setLayout(layout); return group

    def _update_event_freq(self, value: int):
        """Updates event frequency when the slider changes."""
        self.event_freq=value/100.0; self.freq_value_label.setText(f"{self.event_freq:.2f}");
        self.agent.update_environment(self.event_freq, self.intensities)

    def _update_intensity(self, index: int, value: int):
        """Updates emotion intensity when a slider changes."""
        if 0 <= index < len(self.intensities) and index < len(self.intensity_labels):
            self.intensities[index]=value/100.0;
            self.intensity_labels[index].setText(f"{self.intensities[index]:.2f}");
            self.agent.update_environment(self.event_freq, self.intensities)
        else: logger.warning(f"Intensity update invalid index {index} or label mismatch.")

    def send_message(self):
        """Handles sending a user message from the chat input."""
        if not hasattr(self, 'chat_input') or not hasattr(self, 'chat_display'): logger.error("Chat widgets not initialized properly."); return
        user_text = self.chat_input.text().strip();
        if not user_text: return # Ignore empty input
        self.chat_display.append(f"<font color='#90CAF9'><b>You:</b> {html.escape(user_text)}</font>"); self.chat_input.clear()
        try:
            ai_response = self.agent.handle_user_chat(user_text);
            self.chat_display.append(f"<font color='#FFF59D'><b>AI:</b> {html.escape(ai_response)}</font>")
            # Update state/HUD after chat interaction
            stats = self.agent.reflect(); response_emotions = self.agent.last_response_emotions
            if not isinstance(response_emotions, torch.Tensor) or not is_safe(response_emotions): logger.warning("Invalid resp emo send_msg."); response_emotions = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
            self.state_widget.update_display(stats, response_emotions, self.agent.last_reported_loss)
            # handle_user_chat updates avatar movement internally now
            hud_metrics = { "att_score": stats.get("att_score", 0.0), "rho_score": stats.get("rho_score", 0.0), "loss": self.agent.last_reported_loss, "box_score": stats.get("box_score", 0.0) }
            self.hud_widget.update_hud(response_emotions, ai_response, hud_metrics)
        except Exception as e: logger.error(f"Error handle_user_chat call: {e}", exc_info=True); self.chat_display.append("<font color='#FF8A80'><b>AI:</b> [Error processing message]</font>")
        # Scroll chat display to the bottom
        sb = self.chat_display.verticalScrollBar(); sb.setValue(sb.maximum())

    def update_game(self):
        """Main update loop: steps simulation, triggers learning, updates GUI."""
        if self.paused: return
        try:
            step_results = self.agent.train_step()

            if isinstance(step_results, dict) and step_results.get('error'):
                 logger.error(f"Orchestrator error: {step_results.get('message', 'Unknown')}. Pausing."); self._pause_simulation(force_pause=True); self.state_widget.set_status(step_results.get("message", "Error!"), "#F44336"); return
            # Check for 7 items
            if not isinstance(step_results, tuple) or len(step_results) != 7:
                 logger.error(f"Orchestrator returned invalid results ({len(step_results)} items, expected 7). Pausing.")
                 self._pause_simulation(force_pause=True); self.state_widget.set_status("Step Return Error!", "#F44336")
                 return

            # Unpack 7 items
            metrics, reward, done, response, loss_value, monologue, predicted_hm_label = step_results
            self.last_reward = reward; self.last_loss = loss_value

            # --- Update Avatar with Predicted Head Movement (from internal step) ---
            if self.avatar_widget and hasattr(self.avatar_widget, 'update_predicted_movement'):
                # Use the movement predicted during this train_step, NOT the chat one
                self.avatar_widget.update_predicted_movement(predicted_hm_label)
            # ---

            # Update Avatar Emotions
            chat_emotions = self.agent.last_response_emotions # Use emotions tied to last response for display
            if not isinstance(chat_emotions, torch.Tensor) or not is_safe(chat_emotions):
                 logger.warning("Invalid response emotions update_game."); chat_emotions = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
            if self.avatar_widget and hasattr(self.avatar_widget, 'update_emotions'):
                self.avatar_widget.update_emotions(chat_emotions)

            # Update State/HUD Widgets
            self.state_widget.update_display(metrics, chat_emotions, loss_value) # metrics includes monologue now
            hud_metrics = { "att_score": metrics.get("att_score", 0.0), "rho_score": metrics.get("rho_score", 0.0), "loss": loss_value, "box_score": metrics.get("box_score", 0.0) }
            # HUD shows the agent's *latest* response (which could be from internal step or chat)
            self.hud_widget.update_hud(chat_emotions, self.agent.current_response, hud_metrics)

            # Periodic checks and episode handling
            self.steps_since_last_check += 1
            if self.steps_since_last_check >= self.completeness_check_interval: self._run_periodic_completeness_test()
            if done: logger.info(f"--- Episode {self.agent.episode_count} ended ---") # Orchestrator handles reset

        except Exception as e: logger.error(f"Critical error in GUI update_game loop: {e}", exc_info=True); self._pause_simulation(force_pause=True); self.state_widget.set_status("Runtime Error!", "#F44336")

    def _run_periodic_completeness_test(self):
        """Runs the completeness test periodically."""
        try: result, details = self.agent.test_completeness(); self.state_widget.update_completeness_display(result, details)
        except Exception as e: logger.error(f"Error periodic completeness test: {e}", exc_info=True)
        finally: self.steps_since_last_check = 0 # Reset counter regardless of success

    def _run_manual_completeness_test(self):
        """Runs the completeness test when the button is clicked."""
        logger.info("Manual Completeness Test triggered.");
        if self.paused: logger.info("Note: Simulation is paused during manual test.")
        try:
            result, details = self.agent.test_completeness()
            self.state_widget.update_completeness_display(result, details);
            self.steps_since_last_check = 0 # Reset periodic counter
            QMessageBox.information(self, "Completeness Test Result", f"Result: {'PASS ✅' if result else 'FAIL ❌'}\n\nDetails:\n{details}")
        except Exception as e:
            logger.error(f"Error during manual completeness test: {e}", exc_info=True);
            QMessageBox.warning(self, "Completeness Test Error", f"Failed to run test:\n{e}")

    def _pause_simulation(self, force_pause: bool = False):
        """Toggles or forces the pause state of the simulation."""
        can_run = self.agent and self.agent.avatar and self.agent.avatar.model_loaded
        if force_pause:
            if not self.paused:
                self.paused=True;
                if self.timer.isActive(): self.timer.stop();
                self.state_widget.set_status("Paused (Forced)", "#E67E22"); # Darker orange
                logger.info("Simulation Paused (Forced).")
            elif self.paused: # Already paused, just update status if needed
                self.state_widget.set_status("Paused (Forced)", "#E67E22")
            return # Exit after forcing pause

        # Toggle pause state
        if self.paused:
            if can_run: # Can only resume if avatar is ready
                self.paused=False;
                if not self.timer.isActive(): # Start timer if not running
                     interval=max(10, int(1000.0/Config.Graphics.FPS)); self.timer.start(interval);
                self.state_widget.set_status("Running", "#4CAF50"); # Green
                logger.info("Simulation Resumed.")
            else:
                logger.warning("Cannot resume: Character not ready.");
                QMessageBox.warning(self, "Cannot Resume", "Character model is not loaded or ready. Cannot resume simulation.");
                self.state_widget.set_status("Cannot Run (Char Error)", "#F44336") # Red
        else: # If running, pause it
            self.paused=True;
            if self.timer.isActive(): self.timer.stop();
            self.state_widget.set_status("Paused", "#FFA500"); # Orange
            logger.info("Simulation Paused.")

    def keyPressEvent(self, event: QKeyEvent):
        """Handles key presses for pausing, quitting, and chat input focus."""
        key = event.key()
        # If chat input has focus, prioritize Enter key for sending message
        if hasattr(self, 'chat_input') and self.chat_input.hasFocus():
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.send_message()
                event.accept() # Consume event
                return
            else:
                super().keyPressEvent(event) # Allow other keys in chat input
            return # Don't process global keys if chat has focus

        # Global key bindings
        if key == Qt.Key.Key_Space:
            self._pause_simulation()
            event.accept()
        elif key == Qt.Key.Key_Q or key == Qt.Key.Key_Escape:
            self.close() # Trigger closeEvent for cleanup
            event.accept()
        elif key == Qt.Key.Key_C:
            self._run_manual_completeness_test()
            event.accept()
        else:
            super().keyPressEvent(event) # Pass unhandled keys up

    # --- ADDED: Save/Load Methods ---
    def _save_agent_state(self):
        """Handles the Save Agent button click."""
        was_paused = self.paused
        if not was_paused:
            self._pause_simulation(force_pause=True) # Pause before saving
            QApplication.processEvents() # Process pause event
            time.sleep(0.1) # Brief pause

        logger.info("Save button clicked. Attempting to save agent state...")
        try:
            self.agent.save_agent()
            QMessageBox.information(self, "Save Successful", f"Agent state saved successfully.")
        except Exception as e:
            logger.error(f"Error saving agent state via GUI: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Failed", f"Could not save agent state:\n{e}")

        if not was_paused:
            self._pause_simulation() # Resume if it was running before

    def _load_agent_state(self):
        """Handles the Load Agent button click."""
        was_paused = self.paused
        if not was_paused:
            self._pause_simulation(force_pause=True) # Pause before loading
            QApplication.processEvents() # Process pause event
            time.sleep(0.1) # Brief pause

        logger.info("Load button clicked. Attempting to load agent state...")
        try:
            # Check if files/dirs exist before attempting load
            # Agent state is critical, GPT less so (can use base model)
            agent_path_exists = os.path.exists(AGENT_SAVE_PATH)
            gpt_path_exists = os.path.isdir(GPT_SAVE_PATH) # Check if directory exists for HF

            if not agent_path_exists:
                QMessageBox.warning(self, "Load Failed", f"Cannot load: Agent state file not found.\nExpected: {AGENT_SAVE_PATH}")
                if not was_paused: self._pause_simulation() # Resume if paused for load
                return
            if not gpt_path_exists:
                logger.warning(f"GPT save directory not found ({GPT_SAVE_PATH}). Will use base HF model.")
                # No critical error, agent.load_agent handles missing GPT load

            if self.agent.load_agent():
                 QMessageBox.information(self, "Load Successful", "Agent state loaded successfully.")
                 # Reset some runtime state after loading
                 self.agent.last_response_emotions = self.agent.model.prev_emotions.clone().detach()
                 self.agent.mood = torch.ones_like(self.agent.model.prev_emotions) * 0.5 # Reset mood
                 self.agent.conversation_history.clear()
                 self.agent.last_internal_monologue = ""
                 self.agent.current_response = "Loaded state."
                 # Update display immediately
                 stats = self.agent.reflect()
                 self.state_widget.update_display(stats, self.agent.last_response_emotions, self.agent.last_reported_loss)
                 self.hud_widget.update_hud(self.agent.last_response_emotions, self.agent.current_response, stats)
                 self.chat_display.clear()
                 self.chat_display.append("<font color='#FFCC80'><i>Agent state loaded.</i></font>")

            else:
                 QMessageBox.critical(self, "Load Failed", "Failed to load agent state. Check logs for details.")
        except Exception as e:
            logger.error(f"Error loading agent state via GUI: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Failed", f"An unexpected error occurred during loading:\n{e}")

        if not was_paused:
            self._pause_simulation() # Resume if it was running before
    # --- END ADDED ---

    def closeEvent(self, event: QCloseEvent):
        """Handles cleanup when the main window is closed."""
        logger.info("Close event triggered. Initiating cleanup...")
        # Stop the main timer
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()
            logger.debug("Main GUI timer stopped.")

        # Call orchestrator cleanup (which should call avatar cleanup)
        if self.agent and hasattr(self.agent, 'cleanup'):
             try:
                 logger.debug("Calling orchestrator cleanup via closeEvent...")
                 self.agent.cleanup()
                 logger.debug("Orchestrator cleanup completed.")
             except Exception as e:
                 logger.error(f"Error during orchestrator cleanup on close: {e}", exc_info=True)
        else:
             logger.warning("Orchestrator cleanup method not found or agent not available.")
             # Attempt direct avatar cleanup as a fallback
             if hasattr(self, 'avatar_widget') and self.avatar_widget and hasattr(self.avatar_widget, 'cleanup'):
                logger.warning("Attempting direct avatar cleanup as fallback...")
                try: self.avatar_widget.cleanup()
                except Exception as e: logger.error(f"Error during direct avatar cleanup on close: {e}")

        logger.info("GUI Cleanup finished. Accepting close event.")
        event.accept() # Allow the window to close

# --- END OF FILE main_gui.py ---
