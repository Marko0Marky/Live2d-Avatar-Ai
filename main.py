# --- START OF FILE main.py ---

import sys
import os
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Tuple

import time # Import time for fallback loop timing

# Use qasync library for integrating asyncio with PyQt
import qasync

# --- Initial Setup (Logging, Path, Config Import, DEVICE) ---
try:
    logger = logging.getLogger(__name__) # Create a preliminary logger
    script_path = os.path.dirname(os.path.realpath(__file__))
    logger.debug(f"Script path initially determined as: {script_path}")
    os.chdir(script_path)
    logger.info(f"CWD changed to script directory: {os.getcwd()}")
    if script_path not in sys.path: sys.path.insert(0, script_path)
    from config import MasterConfig as Config, logger as config_logger
    from config import DEVICE # Import DEVICE
    logger = config_logger # Use the configured logger from now on
    logger.info("Configuration loaded successfully.")
except ImportError as e: logger.critical(f"CRITICAL ERROR: Failed to import 'config': {e}", exc_info=True); sys.exit(1)
except FileNotFoundError: logger.critical(f"CRITICAL ERROR: CWD issue. CWD: {os.getcwd()}", exc_info=True); sys.exit(1)
except Exception as e: logger.critical(f"CRITICAL ERROR: Unexpected init error: {e}", exc_info=True); sys.exit(1)

# --- Check PyTorch, Transformers, Datasets ---
try: import torch; logger.info(f"PyTorch version {torch.__version__} available.")
except ImportError: logger.critical("CRITICAL ERROR: PyTorch is not installed."); sys.exit(1)
try: import transformers, datasets; logger.info("Hugging Face libs found.")
except ImportError: logger.warning("transformers/datasets not found.")

# --- Core Application Imports ---
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox, QTextEdit, QPushButton, QLineEdit, QLabel # Added QLabel
    from PyQt5.QtCore import Qt, QTimer, QSize
    from PyQt5.QtGui import QIcon
    from orchestrator import EnhancedConsciousAgent
    from graphics import Live2DCharacter
    from gui_widgets import AIStateWidget, HUDWidget
    logger.info("Core modules imported successfully.")
except ImportError as e: logger.critical(f"CRITICAL ERROR: Failed core import: {e}", exc_info=True); sys.exit(1)
except Exception as e: logger.critical(f"CRITICAL ERROR: Unexpected core import error: {e}", exc_info=True); sys.exit(1)

# --- Forward Declaration ---
if TYPE_CHECKING:
    from asyncio import AbstractEventLoop


class EnhancedGameGUI(QMainWindow):
    """Main GUI window integrating Live2D, HUD, AI controls, and Chat."""
    cli_args: Optional[argparse.Namespace] = None

    def __init__(self, agent_orchestrator: EnhancedConsciousAgent, loop: 'AbstractEventLoop'):
        super().__init__()
        self.agent_orchestrator = agent_orchestrator
        self.loop = loop
        self.setWindowTitle("Syntrometric Conscious Agent - v5 + Chat")
        self.setGeometry(100, 100, 1300, 850)
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        if os.path.exists(icon_path): self.setWindowIcon(QIcon(icon_path))
        else: logger.warning(f"Icon file not found at {icon_path}")

        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #181828;")
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left Pane: Live2D Character and HUD
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane); left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(0)

        # --- FIX: Remove conditional check based on imported flags ---
        # Assume Live2DCharacter handles internal checks for library availability
        try:
            self.live2d_widget = Live2DCharacter() # Directly instantiate
            self.live2d_widget.error_occurred.connect(self.show_error_message)
            self.live2d_widget.character_initialized.connect(self.on_character_init)
            self.live2d_widget.interaction_detected.connect(self.on_character_interaction)
            left_layout.addWidget(self.live2d_widget, stretch=1)

            # Create HUD only if Live2D widget was likely created successfully
            self.hud_widget = HUDWidget(self.agent_orchestrator);
            self.agent_orchestrator.set_hud_widget(self.hud_widget)
            self.hud_widget.setParent(self.live2d_widget);
            self.hud_widget.hide()
        except Exception as e: # Catch potential errors during Live2DCharacter init
             logger.error(f"Failed to create Live2D/HUD widgets: {e}", exc_info=True)
             # Provide a placeholder if graphics failed
             placeholder = QLabel("Graphics/Live2D Unavailable\nCheck Dependencies/Logs");
             placeholder.setAlignment(Qt.AlignCenter);
             placeholder.setStyleSheet("color: white; font-size: 16px;");
             left_layout.addWidget(placeholder)
             self.live2d_widget = None # Ensure attributes exist but are None
             self.hud_widget = None
        # --- END FIX ---
        splitter.addWidget(left_pane)

        # --- Right Pane: AI State Widget and Chat Interface ---
        # ... (Rest of __init__ remains the same) ...
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane); right_pane.setMinimumWidth(400); right_pane.setMaximumWidth(600)
        self.ai_state_widget = AIStateWidget(self.agent_orchestrator)
        self.ai_state_widget.request_completeness_test.connect(self.run_completeness_test)
        right_layout.addWidget(self.ai_state_widget, stretch=2)
        chat_group = QWidget(); chat_layout = QVBoxLayout(chat_group); chat_layout.setContentsMargins(5, 5, 5, 5); chat_layout.setSpacing(4)
        self.chat_history = QTextEdit(); self.chat_history.setReadOnly(True); self.chat_history.setStyleSheet("background-color: #1e1e2e; color: #cccccc; border: 1px solid #444; border-radius: 4px; font-size: 11px;")
        chat_layout.addWidget(self.chat_history, stretch=3)
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit(); self.chat_input.setPlaceholderText("Enter message..."); self.chat_input.setStyleSheet("background-color: #2a2a3a; color: #dddddd; border: 1px solid #555; border-radius: 3px; padding: 4px;")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        input_layout.addWidget(self.chat_input)
        send_button = QPushButton("Send"); send_button.setStyleSheet("QPushButton { font-size: 10px; padding: 4px 8px; background-color: #007B8C; color: white; border-radius: 3px; border: 1px solid #005f6b; } QPushButton:hover { background-color: #009CB0; } QPushButton:pressed { background-color: #005f6b; }")
        send_button.clicked.connect(self.send_chat_message)
        input_layout.addWidget(send_button)
        chat_layout.addLayout(input_layout)
        right_layout.addWidget(chat_group, stretch=1)
        splitter.addWidget(right_pane)
        splitter.setSizes([700, 600])

        self.timer = QTimer(self); timer_interval_ms = int(1000 / Config.Graphics.FPS) if Config.Graphics.FPS > 0 else 16
        self.timer.setInterval(timer_interval_ms); self.timer.timeout.connect(self.update_game); self.is_running = False; self.last_update_time = 0.0
        logger.debug(f"GUI Init: Live2D exists: {self.live2d_widget is not None}, HUD exists: {self.hud_widget is not None}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.hud_widget and self.live2d_widget and hasattr(self.live2d_widget, 'height'):
            hud_margin=10; hud_width=200; hud_height=300; live2d_h = self.live2d_widget.height()
            if live2d_h > 0: self.hud_widget.setGeometry(hud_margin, live2d_h - hud_height - hud_margin, hud_width, hud_height)
            else: self.hud_widget.setGeometry(hud_margin, hud_margin, hud_width, hud_height)

    def start_simulation(self):
        if not self.is_running:
            if not (hasattr(self, 'live2d_widget') and self.live2d_widget): # Check if widget exists
                 logger.error("Cannot start simulation: Live2D widget not initialized.")
                 self.show_error_message("Cannot start simulation: Graphics failed.")
                 return
            logger.info("Starting simulation loop..."); self.last_update_time = self.loop.time() if self.loop else time.time() # Use loop time if available
            self.timer.start(); self.is_running = True; self.ai_state_widget.set_status("Running", "#4CAF50")
        else: logger.warning("Simulation already running.")

    def stop_simulation(self):
        if self.is_running: logger.info("Stopping simulation loop..."); self.timer.stop(); self.is_running = False; self.ai_state_widget.set_status("Stopped", "#F44336")
        else: logger.warning("Simulation already stopped.")

    def update_game(self):
        live2d_ready = self.live2d_widget and hasattr(self.live2d_widget, 'model_loaded') and self.live2d_widget.model_loaded
        if not self.is_running or not live2d_ready: return
        current_time = self.loop.time() if self.loop else time.time()
        delta_time = current_time - self.last_update_time; self.last_update_time = current_time
        try:
            metrics, reward, done, response, loss, monologue = self.agent_orchestrator.train_step() # Expect 6 values
            response_emotions_list = metrics.get('current_emotions_response', [0.0]*Config.Agent.EMOTION_DIM)
            response_emotions_tensor = torch.tensor(response_emotions_list, device=DEVICE, dtype=torch.float32)
            if self.live2d_widget: self.live2d_widget.update_emotions(response_emotions_tensor); self.live2d_widget.update_predicted_movement("idle")
            if self.hud_widget: self.hud_widget.update_hud(response_emotions_tensor, self.agent_orchestrator.current_response, metrics)
            self.ai_state_widget.update_display(metrics, response_emotions_tensor, loss)
            if done: logger.info("Episode finished.")
        except Exception as e: logger.error(f"Game update loop error: {e}", exc_info=True); self.show_error_message(f"Runtime Error: {e}"); self.stop_simulation()

    def send_chat_message(self):
        user_message = self.chat_input.text().strip()
        if not user_message: return
        self.append_chat_message("You", user_message); self.chat_input.clear()
        try: ai_response = self.agent_orchestrator.handle_user_chat(user_message); self.append_chat_message("AI", ai_response)
        except Exception as e: logger.error(f"Chat response error: {e}", exc_info=True); self.append_chat_message("System", "[Response Error]")

    def append_chat_message(self, sender: str, message: str):
        import html; safe_message = html.escape(message)
        formatted_message = f"<b>{sender}:</b> {safe_message}<br>"; self.chat_history.append(formatted_message)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def on_character_init(self, success: bool):
        """Callback when Live2D character finishes initialization."""
        if success:
            logger.info("Live2D Character Initialized Successfully.")
            if self.hud_widget:
                self.hud_widget.show()
                self.resizeEvent(None) # Position HUD
            self.start_simulation() # Start simulation loop *after* successful init
        else:
            logger.error("Live2D Character Initialization Failed.")
            self.show_error_message("Failed to initialize the Live2D character. Check logs and model path.")
            if hasattr(self, 'ai_state_widget') and self.ai_state_widget:
                self.ai_state_widget.set_status("Error: Live2D Failed", "#FF0000")

    def on_character_interaction(self): logger.debug("Character interaction.")
    def run_completeness_test(self):
        logger.info("GUI requesting RIH test..."); self.stop_simulation()
        try: is_complete, details = self.agent_orchestrator.test_completeness(); self.ai_state_widget.update_completeness_display(is_complete, details); QMessageBox.information(self, "RIH Test", f"Result: {'Met' if is_complete else 'Not Met'}\nDetails: {details}")
        except Exception as e: logger.error(f"RIH test error: {e}"); self.show_error_message(f"Test Error: {e}")
        finally: self.start_simulation()
    def _save_agent_state(self):
        logger.info("Save request..."); self.stop_simulation()
        try: self.agent_orchestrator.save_agent(); QMessageBox.information(self, "Save", "Agent state saved.")
        except Exception as e: logger.error(f"Save error: {e}"); self.show_error_message(f"Save failed: {e}")
        finally: self.start_simulation()
    def _load_agent_state(self):
        logger.info("Load request..."); self.stop_simulation()
        reply = QMessageBox.question(self, 'Load', "Overwrite state?", QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                if self.agent_orchestrator.load_agent():
                    emotions = self.agent_orchestrator.last_response_emotions.to(DEVICE); metrics = self.agent_orchestrator.reflect()
                    if self.live2d_widget: self.live2d_widget.update_emotions(emotions)
                    if self.hud_widget: self.hud_widget.update_hud(emotions, "Loaded.", metrics)
                    self.ai_state_widget.update_display(metrics, emotions, self.agent_orchestrator.last_reported_loss)
                    self.ai_state_widget.set_status("Loaded", "#03A9F4"); QMessageBox.information(self, "Load", "State loaded.")
                    self.start_simulation()
                else: self.show_error_message("Load failed."); self.ai_state_widget.set_status("Load Failed", "#FF0000")
            except Exception as e: logger.error(f"Load error: {e}"); self.show_error_message(f"Load error: {e}"); self.ai_state_widget.set_status("Load Error", "#FF0000")
        else: logger.info("Load cancelled."); self.start_simulation()
    def show_error_message(self, message: str): logger.error(f"GUI Error: {message}"); QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        logger.info("Close event triggered. Initiating cleanup...")
        self.stop_simulation()
        if EnhancedGameGUI.cli_args and EnhancedGameGUI.cli_args.save_on_exit:
             logger.info("Attempting save on exit (from closeEvent)...")
             try:
                 if hasattr(self,'agent_orchestrator') and self.agent_orchestrator and not self.agent_orchestrator.cleaned_up: self.agent_orchestrator.save_agent(); logger.info("Agent state saved.")
                 else: logger.warning("Skip save: Orchestrator unavailable/cleaned.")
             except Exception as e: logger.error(f"Save on exit error: {e}", exc_info=True)
        if hasattr(self, 'live2d_widget') and self.live2d_widget: self.live2d_widget.cleanup()
        if hasattr(self, 'agent_orchestrator') and self.agent_orchestrator and not self.agent_orchestrator.cleaned_up: self.agent_orchestrator.cleanup()
        else: logger.debug("Orchestrator cleanup skipped in closeEvent.")
        logger.info("GUI Cleanup finished.")
        event.accept()
        instance = QApplication.instance();
        if instance: logger.info("Requesting app quit from closeEvent."); instance.quit()


# --- Main Application Logic ---
async def main_async_loop(args: argparse.Namespace, loop: 'AbstractEventLoop', app: QApplication) -> Optional[EnhancedGameGUI]:
    """ Sets up the GUI and returns the main window instance. """
    # (Implementation remains the same - sets up GUI and returns window)
    logger.info("--- Application Starting (Async Setup) ---")
    logger.info(f"Using PyTorch device: {DEVICE}")
    logger.info(f"Target Live2D Model Path: {Config.Graphics.MODEL_PATH}")
    logger.info(f"Log file: {Config.LOG_FILE}")
    logger.info(f"Agent State Dimension: {Config.Agent.STATE_DIM} (LM: {Config.NLP.HUGGINGFACE_MODEL})")

    agent_orchestrator = None # Initialize to None
    main_window = None      # Initialize to None
    try:
        # Initialize Orchestrator
        agent_orchestrator = EnhancedConsciousAgent()
        logger.info("Orchestrator initialized.")

        # Load Agent if requested (only if orchestrator was successfully initialized)
        if args.load:
            logger.info("Attempting --load agent state...")
            # Make sure agent_orchestrator exists before calling load_agent
            if agent_orchestrator:
                if not agent_orchestrator.load_agent():
                    logger.error("Load failed, starting fresh.")
                else:
                    logger.info("Load successful via --load.")
            else:
                # This case should technically not be reached if init fails earlier,
                # but added for robustness.
                logger.error("Cannot load agent state: Orchestrator failed to initialize.")
        EnhancedGameGUI.cli_args = args
        main_window = EnhancedGameGUI(agent_orchestrator, loop)
        main_window.show(); logger.info("GUI shown.")
        logger.info(f"Is main_window visible after show() in main_async_loop? {main_window.isVisible()}")
        # --- This coroutine now just does setup and returns the window ---
        return main_window
    except Exception as e:
        logger.critical(f"Error during setup in main_async_loop: {e}", exc_info=True)
        if agent_orchestrator and hasattr(agent_orchestrator,'cleaned_up') and not agent_orchestrator.cleaned_up: agent_orchestrator.cleanup()
        if app: app.quit()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Syntrometric Conscious Agent GUI")
    parser.add_argument('--load', action='store_true', help='Load saved agent state.')
    parser.add_argument('--save-on-exit', action='store_true', help='Save agent state on exit.')
    cli_args = parser.parse_args()
    logger.info(f"Command line arguments: {cli_args}")

    exit_code = 1; app = None; loop = None; setup_task = None
    try:
        app = QApplication.instance();
        if app is None: app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)

        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        logger.info("Starting application with qasync event loop...")

        # --- FIX: Run setup task, then run app.exec_() ---
        # 1. Run the setup coroutine to create the GUI
        logger.info("Running setup task...")
        main_window_ref = loop.run_until_complete(main_async_loop(cli_args, loop, app))
        logger.info("Setup task finished.")

        # 2. Check if setup succeeded and window exists
        if main_window_ref is not None:
            logger.info("Setup successful. Starting Qt event loop (app.exec_)...")
            # 3. Start the blocking Qt event loop. qasync allows asyncio tasks
            #    (like the QTimer in the GUI) to run concurrently.
            exit_code = app.exec_()
            logger.info(f"Qt application event loop finished. Exit code: {exit_code}")
        else:
            logger.error("Setup failed (main_window is None), application will not run main event loop.")
            exit_code = 1 # Ensure non-zero exit code if setup failed
        # --- END FIX ---

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...");
        if app: app.quit(); # Ask Qt app to quit gracefully
        exit_code = 0
    except SystemExit as sysexit:
        logger.info(f"SystemExit called with code: {sysexit.code}")
        exit_code = sysexit.code or 0
    except Exception as e:
        logger.critical(f"CRITICAL Unhandled Exception at top level: {e}", exc_info=True)
        exit_code = 1
    finally:
        logger.info("Main execution block's finally reached.")
        # asyncio loop cleanup (important if loop was started but exec_ failed/interrupted)
        if loop is not None and loop.is_running() and not loop.is_closed():
             logger.info("Closing asyncio event loop in main finally.")
             try: # Cancel any remaining asyncio tasks
                 tasks = [t for t in asyncio.all_tasks(loop=loop) if not t.done()]
                 if tasks:
                      logger.debug(f"Cancelling {len(tasks)} outstanding asyncio tasks...")
                      for task in tasks: task.cancel()
                      try: loop.run_until_complete(asyncio.sleep(0.1)) # Allow cancellations
                      except RuntimeError as e: logger.warning(f"Ignoring RT error during task cancel sleep: {e}")
                 # Ensure loop is closed
                 if not loop.is_closed(): loop.close(); logger.info("Asyncio loop closed.")
             except Exception as task_cancel_err: logger.error(f"Error finalizing asyncio loop: {task_cancel_err}")
        elif loop is not None and loop.is_closed():
             logger.info("Asyncio event loop was already closed.")
        else:
             logger.info("No active asyncio event loop to close.")


        logger.info(f"--- Application Exiting (Final Code: {exit_code}) ---")
        logging.shutdown()
        sys.exit(exit_code)

# --- END OF FILE main.py ---
