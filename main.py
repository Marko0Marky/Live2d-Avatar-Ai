# --- START OF FILE main.py ---

import sys
import os
import asyncio
import logging
import concurrent.futures # Keep for potential future use, though not directly used here
from typing import Optional

# Need Qt and qasync
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    import qasync
    from qasync import QEventLoop
except ImportError as e:
     # Use basic print before logger might be fully configured
     print(f"CRITICAL ERROR: PyQt5 or qasync import failed: {e}")
     print("Please install PyQt5 and qasync: pip install PyQt5 qasync")
     sys.exit(1)

# --- Setup logging ---
# Assuming log_file is defined globally for error messages if logger fails
log_file = "vr_avatar_ai5_run.log"
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
# Try basic config first, handlers can be adjusted later
try:
    logging.basicConfig(level=logging.DEBUG, format=log_format, handlers=[
        logging.FileHandler(log_file, mode='w'),
        # logging.StreamHandler(sys.stdout) # Add StreamHandler later if needed
    ])
    logger = logging.getLogger(__name__) # Get logger instance

    # Ensure console handler exists and level is INFO
    console_handler = None
    for handler in logging.root.handlers: # Check root logger handlers
        if isinstance(handler, logging.StreamHandler):
            console_handler = handler
            break
    if console_handler:
        console_handler.setLevel(logging.INFO)
        logger.debug("Console handler found and level set to INFO.")
    else: # Add one if missing
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        logging.root.addHandler(ch) # Add to root logger
        logger.info("StreamHandler added manually.")

    # Reduce noise from libraries known to be verbose
    logging.getLogger('OpenGL').setLevel(logging.WARNING)
    logging.getLogger('PyQt5').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tokenizers').setLevel(logging.WARNING)

except Exception as log_e:
    print(f"CRITICAL ERROR: Failed to configure logging: {log_e}")
    print(f"Check write permissions for the log file: {log_file}")
    # Continue without logging if setup fails? Or exit? For now, continue.
    logger = logging.getLogger(__name__) # Get a basic logger anyway
    logger.error(f"Logging setup failed: {log_e}", exc_info=True)


# --- Import Core Application Components ---
try:
    # Use MasterConfig object and tokenizer functions
    from config import MasterConfig as Config # Use instantiated config
    from config import DEVICE, TRAIN_DATA, train_or_load_tokenizer
    from orchestrator import EnhancedConsciousAgent
    from main_gui import EnhancedGameGUI
except ImportError as import_err:
    logger.critical(f"Failed to import core modules: {import_err}. Check file structure and dependencies.", exc_info=True)
    try:
        # Try showing message box even if imports failed
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Import Error", f"Failed to import core modules:\n{import_err}\nCheck logs and environment.")
    except Exception:
        pass # Ignore if GUI components aren't available
    sys.exit(1)
except Exception as config_err:
    logger.critical(f"Error during config import or processing: {config_err}", exc_info=True)
    try:
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Config Error", f"Error loading configuration:\n{config_err}\nCheck config.py and logs.")
    except Exception:
        pass
    sys.exit(1)

async def main_async_loop():
    """
    Sets up the QApplication, Agent Orchestrator, GUI, integrates with asyncio,
    and runs the main event loop. Handles initialization and cleanup.
    """
    app = QApplication.instance()
    if app is None:
        logger.debug("Creating new QApplication instance.")
        app = QApplication(sys.argv);
    else:
        logger.debug("Using existing QApplication instance.")
    app.setStyle("Fusion") # Optional: Set a consistent style

    logger.info("Application starting...")
    logger.info(f"Using PyTorch device: {DEVICE}")
    logger.info(f"Target Live2D Model Path: {Config.Graphics.MODEL_PATH}")
    logger.info(f"Log file: {log_file}")

    window: Optional[EnhancedGameGUI] = None
    agent_orchestrator: Optional[EnhancedConsciousAgent] = None
    loop: Optional[QEventLoop] = None
    exit_code: int = 0

    try:
        # 1. Initialize Tokenizer (Crucial step before Agent)
        logger.info("Initializing Tokenizer...")
        try:
            train_or_load_tokenizer(TRAIN_DATA, Config.NLP)
        except Exception as tok_err:
            logger.critical(f"Failed to initialize tokenizer: {tok_err}", exc_info=True)
            QMessageBox.critical(None, "Fatal Error", f"Failed to initialize tokenizer:\n{tok_err}")
            return 1 # Return error code

        # 2. Initialize Orchestrator (depends on tokenizer's vocab size via Config)
        logger.info("Initializing Agent Orchestrator...")
        agent_orchestrator = EnhancedConsciousAgent()

        # 3. Initialize GUI (depends on Orchestrator)
        logger.info("Initializing Main GUI...")
        window = EnhancedGameGUI(agent_orchestrator)
        window.show()
        logger.info("Agent orchestrator and GUI initialized and shown.")

        # 4. Setup and Run Async Event Loop
        loop = QEventLoop(app);
        asyncio.set_event_loop(loop);
        logger.info("Starting Qt event loop integrated with asyncio...")

        # Run the loop until Application quits or loop is stopped
        await loop.run_forever()
        # Code here runs AFTER loop.stop() or app.quit()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown.")
        exit_code = 0
    except SystemExit as e:
        logger.info(f"System exit called with code: {e.code}");
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        logger.critical(f"Unhandled exception during application run: {e}", exc_info=True);
        try: QMessageBox.critical(None, "Fatal Runtime Error", f"An unexpected error occurred:\n{e}\nSee log '{log_file}'.")
        except Exception as mb_err: logger.error(f"Failed to show error message box: {mb_err}")
        exit_code = 1
    finally:
        logger.info("Main execution block's finally clause reached.")

        # --- Cleanup Resources ---
        # 1. Close Window (This should trigger GUI closeEvent -> orchestrator.cleanup -> avatar.cleanup)
        if window and window.isVisible():
            logger.debug("Closing main window...")
            window.close()
            # Allow Qt to process close events if the loop is still available
            if loop and not loop.is_closed() and loop.is_running():
                 logger.debug("Processing pending Qt events after window close request...")
                 # Give Qt a brief moment to process the close event
                 # This might not be strictly necessary with qasync but can help ensure cleanup signals propagate
                 await asyncio.sleep(0.1) # Short async sleep
                 # app.processEvents() # Alternative sync processing, less ideal in async context
            else:
                 logger.warning("Cannot process Qt events after close request, loop may be stopped/closed.")

        # 2. Orchestrator cleanup is expected to be called by window.closeEvent()
        # No explicit call here unless absolutely necessary as a fallback

        # 3. Stop and close asyncio loop explicitly
        if loop:
            try:
                if loop.is_running():
                    logger.info("Stopping asyncio event loop...")
                    loop.stop()
                # Give tasks spawned by stop() a chance to run
                await asyncio.sleep(0.1) # Short async sleep
                if not loop.is_closed():
                    logger.info("Closing asyncio loop object.")
                    loop.close()
            except RuntimeError as loop_err:
                logger.error(f"Error occurred during asyncio loop cleanup: {loop_err}")
            except Exception as e:
                logger.error(f"Unexpected error stopping/closing asyncio loop: {e}", exc_info=True)

        logger.info(f"Returning exit code from main_async_loop: {exit_code}.")

    return exit_code # Return exit code for asyncio.run

# --- Main Execution Entry Point ---
if __name__ == "__main__":
    # --- Path and CWD Setup ---
    script_dir = os.getcwd()
    try:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            logger.debug(f"Added script directory to sys.path: {script_dir}")
    except NameError:
        logger.warning("__file__ not defined, using CWD for path setup.")
        script_dir = os.getcwd()
    try:
        os.chdir(script_dir)
        logger.info(f"Changed CWD to script directory: {script_dir}")
    except Exception as e:
         logger.error(f"Failed to change CWD to script directory '{script_dir}': {e}")
         logger.warning("Proceeding without changing CWD. Relative paths might fail.")

    # --- Run Main Async Function ---
    final_exit_code: int = 1 # Default error code

    try:
        # Use asyncio.run() to manage the event loop lifecycle
        final_exit_code = asyncio.run(main_async_loop())

    except RuntimeError as e:
        # Handle common asyncio loop issues, especially in IDEs like Spyder
        if "Cannot run the event loop while another loop is running" in str(e) or "no running event loop" in str(e).lower():
             logger.warning(f"Asyncio loop issue ('{e}'). Attempting fallback Qt execution.");
             app_fb = QApplication.instance(); app_fb = QApplication(sys.argv) if app_fb is None else app_fb; app_fb.setStyle("Fusion")
             agent_orchestrator_fb = None
             window_fb = None
             try:
                 # --- Fallback needs tokenizer init too ---
                 logger.info("Fallback: Initializing Tokenizer...")
                 try: train_or_load_tokenizer(TRAIN_DATA, Config.NLP)
                 except Exception as tok_err: logger.critical(f"Fallback: Failed Tokenizer init: {tok_err}"); raise tok_err from tok_err
                 # --- End Fallback Tokenizer Init ---

                 logger.info("Fallback: Initializing Agent and GUI...")
                 agent_orchestrator_fb = EnhancedConsciousAgent();
                 window_fb = EnhancedGameGUI(agent_orchestrator_fb);
                 window_fb.show();
                 logger.info("Fallback: Starting standard Qt event loop (app.exec_)...")
                 final_exit_code = app_fb.exec_() # Run standard Qt loop
             except Exception as gui_err:
                 logger.critical(f"Fallback initialization or execution Error: {gui_err}", exc_info=True);
                 QMessageBox.critical(None, "Fatal Fallback Error", f"Failed fallback execution:\n{gui_err}")
                 final_exit_code=1
             finally: # Ensure cleanup in fallback mode
                  logger.info("Fallback: Cleaning up resources...")
                  # Explicitly close window if it exists
                  if window_fb and window_fb.isVisible():
                      window_fb.close()
                  # Explicitly cleanup orchestrator if it exists
                  if agent_orchestrator_fb and hasattr(agent_orchestrator_fb, 'cleanup'):
                       try: agent_orchestrator_fb.cleanup()
                       except Exception as clean_err: logger.error(f"Fallback orchestrator cleanup error: {clean_err}")
        else: # Re-raise other RuntimeErrors
             logger.critical(f"Unhandled RuntimeError: {e}", exc_info=True); final_exit_code = 1
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught at top level. Exiting cleanly.")
        final_exit_code = 0
    except Exception as e:
        logger.critical(f"Unhandled Exception at top level: {e}", exc_info=True); final_exit_code = 1

    logger.info(f"Exiting application with final code: {final_exit_code}")
    logging.shutdown() # Ensure logs are flushed
    sys.exit(final_exit_code)


# --- END OF FILE main.py ---
