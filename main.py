# --- START OF FILE main.py ---

import sys
import os
import asyncio
import logging
import concurrent.futures
import time
from typing import Optional # Added Optional

# --- PyQt5 / QAsync Imports ---
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget
    import qasync
    from qasync import QEventLoop
except ImportError as e:
     print(f"CRITICAL ERROR: PyQt5 or qasync import failed: {e}")
     print("Please install PyQt5 and qasync: pip install PyQt5 qasync")
     sys.exit(1)

# --- Setup logging ---
# NOTE: Logger is configured within config.py now, but we get the logger instance here.
log_file = "vr_avatar_ai5_run.log" # Define log file name for potential error messages before config load
logger = logging.getLogger(__name__) # Get root logger initially

# --- Import Configuration and Core Components ---
# IMPORTANT: Importing config now also initializes the tokenizer
try:
    from config import MasterConfig as Config # Use instantiated config
    from config import DEVICE, log_file # Get definitive log_file name from config
    # load_train_data and train_or_load_tokenizer are used internally by config/agent now
    from orchestrator import EnhancedConsciousAgent
    from main_gui import EnhancedGameGUI
except ImportError as e:
     # Catch import errors that might happen if config itself fails early
     initial_msg = f"CRITICAL ERROR: Failed to import core modules: {e}. Check configuration and dependencies."
     print(initial_msg)
     try:
         # Attempt to use basic logging if possible
         logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
         logging.critical(initial_msg, exc_info=True)
     except Exception:
         pass # Ignore logging errors if logging itself failed
     sys.exit(1)
except Exception as e:
    # Catch other potential errors during import/config execution
    initial_msg = f"CRITICAL ERROR: Unexpected error during initial imports/config setup: {e}"
    print(initial_msg)
    try:
         logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
         logging.critical(initial_msg, exc_info=True)
    except Exception:
        pass
    sys.exit(1)


async def main_async_loop():
    """Sets up the QApplication, Agent Orchestrator, GUI, integrates with asyncio,
       and runs the main event loop. Handles initialization and cleanup.
    """
    app: Optional[QApplication] = QApplication.instance() # Use Optional type hint
    if app is None:
        logger.debug("Creating new QApplication instance.")
        app = QApplication(sys.argv);
    else:
        logger.debug("Using existing QApplication instance.")
    app.setStyle("Fusion")

    logger.info("--- Application Starting ---")
    logger.info(f"Using PyTorch device: {DEVICE}")
    logger.info(f"Target Live2D Model Path: {Config.Graphics.MODEL_PATH}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Tokenizer Vocab Size: {Config.NLP.VOCAB_SIZE}") # Log vocab size after init

    window: Optional[EnhancedGameGUI] = None # Type hint window
    agent_orchestrator: Optional[EnhancedConsciousAgent] = None # Type hint orchestrator
    loop: Optional[QEventLoop] = None # Type hint loop
    exit_code = 0

    try:
        # Tokenizer is initialized automatically when config.py is imported.

        # Initialize agent (which depends on tokenizer being ready) and GUI
        logger.info("Initializing Agent Orchestrator and GUI...")
        agent_orchestrator = EnhancedConsciousAgent()
        window = EnhancedGameGUI(agent_orchestrator)
        window.show()
        logger.info("Agent orchestrator and GUI initialization complete.")

        # Setup and run the event loop
        loop = QEventLoop(app);
        asyncio.set_event_loop(loop);
        logger.info("Starting Qt event loop integrated with asyncio...")
        loop.run_forever() # Blocks until loop.stop()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown.")
        exit_code = 0
    except SystemExit as e:
        logger.info(f"System exit called with code: {e.code}");
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        logger.critical(f"Unhandled exception during application run: {e}", exc_info=True);
        try:
            # Use window as parent if available, otherwise None
            parent_widget = window if window and isinstance(window, QWidget) else None
            QMessageBox.critical(parent_widget, "Fatal Runtime Error", f"An unexpected error occurred:\n{e}\nSee log '{log_file}'.")
        except Exception as mb_err: logger.error(f"Failed to show error message box: {mb_err}")
        exit_code = 1
    finally:
        logger.info("--- Initiating Application Shutdown ---")

        # --- Cleanup Resources ---
        # 1. Close Window (triggers GUI closeEvent -> orchestrator.cleanup -> avatar.cleanup)
        if window and window.isVisible():
            logger.debug("Closing main window...")
            try:
                window.close()
                # Allow Qt to process the close event if the loop is somehow still usable
                if loop and not loop.is_closed() and loop.is_running():
                     logger.debug("Processing pending Qt events after window close request...")
                     app.processEvents()
                     time.sleep(0.1) # Brief pause to allow processing
                else:
                     logger.warning("Cannot process Qt events post-close, loop may be stopped/closed.")
            except Exception as e:
                 logger.error(f"Error during window close: {e}", exc_info=True)


        # Orchestrator cleanup should be called via window.closeEvent()
        # Avatar cleanup should be called via orchestrator cleanup / window close


        # 3. Stop and close asyncio loop
        if loop:
            try:
                if loop.is_running():
                    logger.info("Stopping asyncio event loop...")
                    loop.stop()

                # Short pause allows pending tasks/callbacks scheduled by stop() to potentially run
                time.sleep(0.2)

                if not loop.is_closed():
                    logger.info("Closing asyncio loop object.")
                    loop.close()
                logger.info("Asyncio loop stopped and closed.")
            except RuntimeError as re:
                logger.warning(f"RuntimeError during loop stop/close (possibly already stopped): {re}")
            except Exception as e:
                logger.error(f"Error stopping/closing asyncio loop: {e}", exc_info=True)

        logger.info(f"Shutdown complete. Returning exit code: {exit_code}.")

    return exit_code

if __name__ == "__main__":
    # --- Path and CWD Setup ---
    script_dir = os.getcwd()
    try:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            # logger.debug(f"Added script directory to sys.path: {script_dir}") # Logger might not be fully configured yet
    except NameError:
        print("Warning: __file__ not defined, using CWD for path setup.")
        # logger.warning("__file__ not defined, using CWD for path setup.") # Logger might not be ready
        script_dir = os.getcwd()
    try:
        os.chdir(script_dir)
        print(f"Info: Changed CWD to script directory: {script_dir}")
        # logger.info(f"Changed CWD to script directory: {script_dir}") # Logger might not be ready
    except Exception as e:
         print(f"ERROR: Failed to change CWD to script directory '{script_dir}': {e}")
         print("Warning: Proceeding without changing CWD. Relative paths might fail.")
         # logger.error(f"Failed to change CWD to script directory '{script_dir}': {e}") # Logger might not be ready
         # logger.warning("Proceeding without changing CWD. Relative paths might fail.")


    # --- Run Main Async Function ---
    final_exit_code = 1
    try:
        # asyncio.run handles loop creation, running the future, and closing the loop
        final_exit_code = asyncio.run(main_async_loop())

    except RuntimeError as e:
        # Handle common asyncio loop errors, especially in IDEs like Spyder
        if "Cannot run the event loop while another loop is running" in str(e) or "no running event loop" in str(e).lower():
             print(f"Warning: Asyncio loop issue ('{e}'). Attempting fallback Qt execution.");
             # logger.warning(f"Asyncio loop issue ('{e}'). Attempting fallback Qt execution."); # Logger might not be fully ready
             app_fb = QApplication.instance(); app_fb = QApplication(sys.argv) if app_fb is None else app_fb; app_fb.setStyle("Fusion")
             agent_orchestrator_fb = None
             window_fb = None
             try:
                 # Fallback doesn't need explicit tokenizer init here, happens on config import
                 agent_orchestrator_fb = EnhancedConsciousAgent(); window_fb = EnhancedGameGUI(agent_orchestrator_fb); window_fb.show();
                 final_exit_code = app_fb.exec_()
             except Exception as gui_err:
                 print(f"CRITICAL ERROR: Fallback Init Error: {gui_err}")
                 # logger.critical(f"Fallback Init Error: {gui_err}"); # Logger might not be ready
                 try: QMessageBox.critical(None, "Fatal Init Error", f"Failed fallback init:\n{gui_err}");
                 except: pass # Ignore errors showing message box if Qt itself is broken
                 final_exit_code=1
             finally:
                  if window_fb and window_fb.isVisible(): window_fb.close() # Ensure window close is called in fallback
                  # Cleanup might have already been called by closeEvent, but check again
                  if agent_orchestrator_fb and hasattr(agent_orchestrator_fb, 'cleanup'):
                       try: agent_orchestrator_fb.cleanup()
                       except Exception as clean_err: print(f"Error: Fallback cleanup error: {clean_err}") # logger might be unreliable
        else: # Re-raise other RuntimeErrors
             print(f"CRITICAL ERROR: Unhandled RuntimeError: {e}")
             # logger.critical(f"Unhandled RuntimeError: {e}", exc_info=True);
             final_exit_code = 1
    except KeyboardInterrupt:
        print("Info: KeyboardInterrupt caught at top level. Exiting cleanly.")
        # logger.info("KeyboardInterrupt caught at top level. Exiting cleanly.")
        final_exit_code = 0
    except Exception as e:
        print(f"CRITICAL ERROR: Unhandled Exception at top level: {e}")
        # logger.critical(f"Unhandled Exception at top level: {e}", exc_info=True);
        final_exit_code = 1

    print(f"Info: Exiting application with final code: {final_exit_code}")
    # logger.info(f"Exiting application with final code: {final_exit_code}")
    logging.shutdown() # Ensure logs are flushed
    sys.exit(final_exit_code)


# --- END OF FILE main.py ---
