# --- START OF FILE main.py ---
import time
import sys
import os
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Tuple

# Use qasync library for integrating asyncio with PyQt
import qasync

# --- Initial Setup ---
try:
    logger = logging.getLogger(__name__)
    script_path = os.path.dirname(os.path.realpath(__file__))
    logger.debug(f"Script path initially determined as: {script_path}")
    os.chdir(script_path)
    logger.info(f"CWD changed to script directory: {os.getcwd()}")
    if script_path not in sys.path: sys.path.insert(0, script_path)
    from config import MasterConfig as Config, logger as config_logger
    from config import DEVICE
    logger = config_logger
    logger.info("Configuration loaded successfully.")
except ImportError as e: logger.critical(f"CRITICAL ERROR: Failed to import 'config': {e}", exc_info=True); sys.exit(1)
except FileNotFoundError: logger.critical(f"CRITICAL ERROR: CWD issue. CWD: {os.getcwd()}", exc_info=True); sys.exit(1)
except Exception as e: logger.critical(f"CRITICAL ERROR: Unexpected init error: {e}", exc_info=True); sys.exit(1)

# --- Check Libs ---
try: import torch; logger.info(f"PyTorch version {torch.__version__} available.")
except ImportError: logger.critical("CRITICAL ERROR: PyTorch is not installed."); sys.exit(1)
try: import transformers, datasets; logger.info("Hugging Face libs found.")
except ImportError: logger.warning("transformers/datasets not found.")

# --- Core App Imports ---
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox, QTextEdit, QPushButton, QLineEdit
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

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane); left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(0)
        self.live2d_widget = Live2DCharacter()
        self.live2d_widget.error_occurred.connect(self.show_error_message)
        self.live2d_widget.character_initialized.connect(self.on_character_init)
        self.live2d_widget.interaction_detected.connect(self.on_character_interaction)
        left_layout.addWidget(self.live2d_widget, stretch=1)
        self.hud_widget = HUDWidget(self.agent_orchestrator); self.agent_orchestrator.set_hud_widget(self.hud_widget)
        self.hud_widget.setParent(self.live2d_widget); self.hud_widget.hide()
        splitter.addWidget(left_pane)

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

        self.timer = QTimer(self)
        timer_interval_ms = int(1000 / Config.Graphics.FPS) if Config.Graphics.FPS > 0 else 16
        self.timer.setInterval(timer_interval_ms)
        self.timer.timeout.connect(self.update_game)
        self.is_running = False
        self.last_update_time = 0.0

        logger.debug(f"GUI Init: Live2D parent: {self.live2d_widget.parent()}")
        logger.debug(f"GUI Init: HUD parent: {self.hud_widget.parent()}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.hud_widget and self.live2d_widget:
            hud_margin = 10; hud_width = 200; hud_height = 300
            live2d_h = self.live2d_widget.height()
            if live2d_h > 0: self.hud_widget.setGeometry(hud_margin, live2d_h - hud_height - hud_margin, hud_width, hud_height)
            else: self.hud_widget.setGeometry(hud_margin, hud_margin, hud_width, hud_height)

    def start_simulation(self):
        if not self.is_running:
            logger.info("Starting simulation loop...")
            self.last_update_time = asyncio.get_event_loop().time()
            self.timer.start()
            self.is_running = True
            self.ai_state_widget.set_status("Running", "#4CAF50")
        else: logger.warning("Simulation already running.")

    def stop_simulation(self):
        if self.is_running:
            logger.info("Stopping simulation loop...")
            self.timer.stop()
            self.is_running = False
            self.ai_state_widget.set_status("Stopped", "#F44336")
        else: logger.warning("Simulation already stopped.")

    def update_game(self):
        if not self.is_running or not hasattr(self.live2d_widget, 'model_loaded') or not self.live2d_widget.model_loaded: return
        current_time = self.loop.time() if self.loop else time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        try:
            metrics, reward, done, response, loss, monologue, predicted_hm_label = self.agent_orchestrator.train_step()
            response_emotions_list = metrics.get('current_emotions_response', [0.0]*Config.Agent.EMOTION_DIM)
            response_emotions_tensor = torch.tensor(response_emotions_list, device=DEVICE, dtype=torch.float32)
            self.live2d_widget.update_emotions(response_emotions_tensor)
            self.live2d_widget.update_predicted_movement(predicted_hm_label)
            self.hud_widget.update_hud(response_emotions_tensor, response, metrics)
            self.ai_state_widget.update_display(metrics, response_emotions_tensor, loss)
            if done: logger.info("Episode finished. Resetting environment (handled by orchestrator).")
        except Exception as e: logger.error(f"Error during game update loop: {e}", exc_info=True); self.show_error_message(f"Runtime Error: {e}"); self.stop_simulation()

    def send_chat_message(self):
        user_message = self.chat_input.text().strip()
        if not user_message: return
        self.append_chat_message("You", user_message)
        self.chat_input.clear()
        try: ai_response = self.agent_orchestrator.handle_user_chat(user_message); self.append_chat_message("AI", ai_response)
        except Exception as e: logger.error(f"Error getting AI chat response: {e}", exc_info=True); self.append_chat_message("System", "[Error getting response]")

    def append_chat_message(self, sender: str, message: str):
        import html
        safe_message = html.escape(message)
        formatted_message = f"<b>{sender}:</b> {safe_message}<br>"
        self.chat_history.append(formatted_message)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def on_character_init(self, success: bool):
        if success: logger.info("Live2D Character Initialized Successfully."); self.hud_widget.show(); self.resizeEvent(None); self.start_simulation()
        else: logger.error("Live2D Character Initialization Failed."); self.show_error_message("Failed to initialize Live2D character."); self.ai_state_widget.set_status("Error: Live2D Init Failed", "#FF0000")

    def on_character_interaction(self): logger.debug("Character interaction detected.")
    def run_completeness_test(self):
        logger.info("GUI requesting completeness test...")
        self.stop_simulation()
        try: is_complete, details = self.agent_orchestrator.test_completeness(); self.ai_state_widget.update_completeness_display(is_complete, details); QMessageBox.information(self, "Completeness Test", f"Result: {'Complete' if is_complete else 'Incomplete'}\nDetails: {details}")
        except Exception as e: logger.error(f"Error running completeness test: {e}", exc_info=True); self.show_error_message(f"Error: {e}")
        finally: self.start_simulation()
    def _save_agent_state(self):
        logger.info("Save agent state requested...")
        self.stop_simulation()
        try: self.agent_orchestrator.save_agent(); QMessageBox.information(self, "Save Agent", "Agent state saved.")
        except Exception as e: logger.error(f"Error saving state: {e}", exc_info=True); self.show_error_message(f"Save failed: {e}")
        finally: self.start_simulation()
    def _load_agent_state(self):
        logger.info("Load agent state requested...")
        self.stop_simulation()
        reply = QMessageBox.question(self, 'Load Agent', "Overwrite current state?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if self.agent_orchestrator.load_agent():
                    emotions = self.agent_orchestrator.last_response_emotions.to(DEVICE); metrics = self.agent_orchestrator.reflect()
                    self.live2d_widget.update_emotions(emotions); self.hud_widget.update_hud(emotions, "Loaded.", metrics); self.ai_state_widget.update_display(metrics, emotions, self.agent_orchestrator.last_reported_loss)
                    self.ai_state_widget.set_status("Loaded", "#03A9F4"); QMessageBox.information(self, "Load Agent", "Agent state loaded."); self.start_simulation()
                else: self.show_error_message("Load failed. Check logs."); self.ai_state_widget.set_status("Load Failed", "#FF0000")
            except Exception as e: logger.error(f"Error loading state: {e}", exc_info=True); self.show_error_message(f"Load error: {e}"); self.ai_state_widget.set_status("Load Error", "#FF0000")
        else: logger.info("Load cancelled."); self.start_simulation()
    def show_error_message(self, message: str): logger.error(f"GUI Error Display: {message}"); QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        logger.info("Close event triggered. Initiating cleanup...")
        self.stop_simulation()
        if EnhancedGameGUI.cli_args and EnhancedGameGUI.cli_args.save_on_exit:
             logger.info("Attempting to save agent state on exit (from closeEvent)...")
             try:
                 if hasattr(self, 'agent_orchestrator') and self.agent_orchestrator and not self.agent_orchestrator.cleaned_up: self.agent_orchestrator.save_agent(); logger.info("Agent state saved successfully on exit.")
                 else: logger.warning("Skipping save on exit: Orchestrator unavailable/cleaned.")
             except Exception as e: logger.error(f"Failed to save agent state on exit: {e}", exc_info=True)
        if hasattr(self, 'live2d_widget') and self.live2d_widget: self.live2d_widget.cleanup()
        if hasattr(self, 'agent_orchestrator') and self.agent_orchestrator and not self.agent_orchestrator.cleaned_up: self.agent_orchestrator.cleanup()
        else: logger.debug("Orchestrator cleanup skipped in closeEvent.")
        logger.info("GUI Cleanup finished.")
        event.accept()
        instance = QApplication.instance()
        if instance:
            logger.info("Requesting application quit from closeEvent.")
            instance.quit()


# --- Main Application Logic ---
async def main_async_loop(args: argparse.Namespace, loop: 'AbstractEventLoop', app: QApplication) -> Optional[EnhancedGameGUI]:
    """ Sets up the GUI and keeps the asyncio loop alive. """
    logger.info("--- Application Starting (Async Loop) ---")
    logger.info(f"Using PyTorch device: {DEVICE}")
    logger.info(f"Target Live2D Model Path: {Config.Graphics.MODEL_PATH}")
    logger.info(f"Log file: {Config.LOG_FILE}")
    logger.info(f"Agent State Dimension: {Config.Agent.STATE_DIM} (LM: {Config.NLP.HUGGINGFACE_MODEL})")

    agent_orchestrator = None
    main_window = None
    try:
        agent_orchestrator = EnhancedConsciousAgent()
        logger.info("Orchestrator initialized.")
        if args.load:
            logger.info("Attempting --load agent state...")
            if not agent_orchestrator.load_agent(): logger.error("Load failed, starting fresh.")
            else: logger.info("Load successful via --load.")
        EnhancedGameGUI.cli_args = args
        main_window = EnhancedGameGUI(agent_orchestrator, loop)
        main_window.show()
        logger.info("GUI shown.")

        # --- Keep coroutine alive indefinitely ---
        await asyncio.Future() # This waits forever until cancelled

        # Code below this await will likely not run if loop is stopped by app.quit()
        logger.info("Async loop's indefinite wait finished (likely cancelled).")
        return main_window # Should ideally not be reached in normal exit

    except asyncio.CancelledError:
         logger.info("main_async_loop cancelled.")
         # Cleanup might be needed here if cancellation happens before closeEvent
         if main_window: main_window.close() # Try to trigger closeEvent cleanup
         elif agent_orchestrator and not agent_orchestrator.cleaned_up: agent_orchestrator.cleanup()
         return main_window # Return reference if it exists
    except Exception as e:
        logger.critical(f"Error during setup/run in main_async_loop: {e}", exc_info=True)
        if agent_orchestrator and hasattr(agent_orchestrator, 'cleaned_up') and not agent_orchestrator.cleaned_up: agent_orchestrator.cleanup()
        if app: app.quit()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Syntrometric Conscious Agent GUI")
    parser.add_argument('--load', action='store_true', help='Load saved agent state.')
    parser.add_argument('--save-on-exit', action='store_true', help='Save agent state on exit.')
    cli_args = parser.parse_args()
    logger.info(f"Command line arguments: {cli_args}")

    exit_code = 1
    app = None
    loop = None
    main_task = None # Keep track of the main task
    try:
        app = QApplication.instance()
        if app is None: app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)

        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        logger.info("Starting application with qasync event loop...")

        # Create the main task but don't run it with qasync.run immediately
        main_task = loop.create_task(main_async_loop(cli_args, loop, app))

        # Start the Qt event loop using qasync's integrated method
        loop.run_forever()

        # Code here runs AFTER the Qt loop has exited
        logger.info("Event loop finished.")
        exit_code = app.exitCode() if hasattr(app, 'exitCode') and app.exitCode() is not None else 0

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Attempting graceful shutdown...")
        if app: app.quit()
        if main_task and not main_task.done(): main_task.cancel()
        exit_code = 0
    except Exception as e:
        logger.critical(f"CRITICAL Unhandled Exception at top level: {e}", exc_info=True)
        if main_task and not main_task.done(): main_task.cancel()
        exit_code = 1
    finally:
        logger.info("Main execution block's finally reached.")
        # Ensure the main task is cancelled if it's still pending
        if main_task and not main_task.done():
            logger.info("Cancelling main async task...")
            main_task.cancel()
            try:
                # Give cancellation a moment
                loop.run_until_complete(asyncio.sleep(0.1))
            except (RuntimeError, asyncio.CancelledError): pass # Ignore errors if loop closed/task cancelled

        if loop is not None and not loop.is_closed():
             logger.info("Closing asyncio event loop in main finally.")
             # Cancel any other remaining tasks (should ideally be none)
             try:
                 tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not main_task and not t.done()]
                 if tasks:
                      logger.debug(f"Cancelling {len(tasks)} other outstanding asyncio tasks...")
                      for task in tasks: task.cancel()
                      loop.run_until_complete(asyncio.sleep(0.1))
             except Exception as task_cancel_err: logger.error(f"Error cancelling other tasks: {task_cancel_err}")
             finally:
                 if not loop.is_closed(): loop.close(); logger.info("Asyncio loop closed.")

        logger.info(f"--- Application Exiting (Final Code: {exit_code}) ---")
        logging.shutdown()
        sys.exit(exit_code)
# --- END OF FILE main.py ---
