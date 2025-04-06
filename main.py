# --- START OF FILE main.py ---

import sys
import os
import asyncio
import logging
import concurrent.futures # Need this for the executor cleanup check

# Need Qt and qasync
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    import qasync
    from qasync import QEventLoop
except ImportError as e:
     print(f"CRITICAL ERROR: PyQt5 or qasync import failed: {e}")
     print("Please install PyQt5 and qasync: pip install PyQt5 qasync")
     sys.exit(1)

# --- Setup logging ---
log_file = "vr_avatar_ai5_run.log"
# Use a slightly more concise format
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format, handlers=[
    logging.FileHandler(log_file, mode='w'),
    logging.StreamHandler(sys.stdout) # Initial stream handler
])
logger = logging.getLogger(__name__) # Get the root logger configured by basicConfig

# Ensure console handler level is INFO
console_handler = None
for handler in logging.root.handlers: # Check root logger handlers
    if isinstance(handler, logging.StreamHandler):
        console_handler = handler
        break
if console_handler:
    console_handler.setLevel(logging.INFO)
    # Optionally re-apply formatter if needed, though basicConfig should handle it
    # console_handler.setFormatter(logging.Formatter(log_format))
    logger.debug("Console handler found and level set to INFO.")
else:
    # Add handler if basicConfig somehow failed to add one (less likely)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)
    logging.root.addHandler(ch) # Add to root logger
    logger.warning("StreamHandler not found by basicConfig, added one manually.")

# Reduce noise from libraries
logging.getLogger('OpenGL').setLevel(logging.WARNING)
logging.getLogger('PyQt5').setLevel(logging.WARNING)


# Need Config, DEVICE (defined in config)
from config import Config, DEVICE

# Need Orchestrator and Main GUI Window
from orchestrator import EnhancedConsciousAgent
from main_gui import EnhancedGameGUI

async def main_async_loop():
    """Sets up the QApplication, Agent Orchestrator, GUI, integrates with asyncio,
       and runs the main event loop. Handles initialization and cleanup.
    """
    # Use QApplication.instance() to ensure only one instance exists
    app = QApplication.instance()
    if app is None:
        logger.debug("Creating new QApplication instance.")
        app = QApplication(sys.argv);
    else:
        logger.debug("Using existing QApplication instance.")
    app.setStyle("Fusion")

    logger.info("Application starting...")
    logger.info(f"Using PyTorch device: {DEVICE}")
    logger.info(f"Target Live2D Model Path: {Config.MODEL_PATH}")
    logger.info(f"Log file: {log_file}")

    window = None
    agent_orchestrator = None # Define outside try for finally block access
    loop = None # Define loop outside try for finally block access
    exit_code = 0 # Default exit code

    try:
        agent_orchestrator = EnhancedConsciousAgent() # Orchestrator created here
        window = EnhancedGameGUI(agent_orchestrator) # GUI needs orchestrator
        window.show()
        logger.info("Agent orchestrator and GUI initialized.")

        # --- Setup qasync event loop ---
        loop = QEventLoop(app);
        asyncio.set_event_loop(loop);
        logger.info("Starting Qt event loop integrated with asyncio...")

        # --- Run the loop (NO await) ---
        loop.run_forever()
        # Code here runs after loop.stop() is called

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown.")
        exit_code = 0 # Often considered normal exit
    except SystemExit as e:
        logger.info(f"System exit called with code: {e.code}");
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        # Catch initialization errors or unhandled loop errors
        logger.critical(f"Unhandled exception during application run: {e}", exc_info=True);
        # Try to show message box if GUI elements exist
        try: QMessageBox.critical(None, "Fatal Runtime Error", f"An unexpected error occurred:\n{e}\nSee log '{log_file}'.")
        except Exception as mb_err: logger.error(f"Failed to show error message box: {mb_err}")
        exit_code = 1
    finally:
        logger.info("Main execution block's finally clause reached.")

        # --- Cleanup Resources ---
        # 1. Close Window (This triggers GUI closeEvent -> orchestrator.cleanup -> avatar.cleanup)
        if window and window.isVisible():
            logger.debug("Closing main window...")
            window.close()
            # Allow Qt to process the close event if the loop is somehow still usable
            if loop and not loop.is_closed() and loop.is_running(): # Check if loop state allows event processing
                 logger.debug("Processing pending Qt events...")
                 app.processEvents() # Process close event etc.
            else:
                 logger.warning("Cannot process Qt events, loop may be stopped/closed.")


        # 2. Explicitly call orchestrator cleanup if it wasn't called via window close (e.g., early exit)
        # Note: window.close() *should* have called orchestrator.cleanup() already. This is a fallback.
        # However, calling cleanup twice might cause issues (e.g., executor shutdown twice).
        # It might be better to rely solely on the closeEvent chain.
        # Let's comment out the direct call here to avoid potential double-cleanup.
        # if agent_orchestrator and hasattr(agent_orchestrator, 'cleanup'):
        #    logger.warning("Calling orchestrator cleanup directly from main finally (fallback).")
        #    try: agent_orchestrator.cleanup()
        #    except Exception as e: logger.error(f"Error during fallback orchestrator cleanup: {e}", exc_info=True)


        # 3. Stop and close asyncio loop
        if loop: # Check if loop was successfully created
            try:
                if loop.is_running():
                    logger.info("Stopping asyncio event loop...")
                    loop.stop()

                # Allow some time for tasks scheduled by stop to run
                # This requires the loop to potentially process events briefly after stop()
                # If using asyncio.run, it handles this better. Here, we might need a small sync pause.
                # await asyncio.sleep(0.1) # Can't await here

                if not loop.is_closed():
                    logger.info("Closing asyncio loop object.")
                    loop.close()
            except Exception as e:
                logger.error(f"Error stopping/closing asyncio loop: {e}", exc_info=True)

        logger.info(f"Returning exit code: {exit_code}.")
        # Let asyncio.run handle the final sys.exit

    return exit_code # Return exit code for asyncio.run

if __name__ == "__main__":
    # --- Path and CWD Setup ---
    script_dir = os.getcwd() # Default to CWD
    try:
        # Use __file__ to get the script's directory reliably
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        # Add to sys.path if not already there
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            logger.debug(f"Added script directory to sys.path: {script_dir}")
    except NameError:
        # __file__ is not defined, e.g., in interactive interpreter
        logger.warning("__file__ not defined, using CWD for path setup.")
        script_dir = os.getcwd() # Use CWD as fallback base

    try:
        # Change CWD *after* getting script_dir based on __file__
        os.chdir(script_dir)
        logger.info(f"Changed CWD to script directory: {script_dir}")
    except Exception as e:
         logger.error(f"Failed to change CWD to script directory '{script_dir}': {e}")
         logger.warning("Proceeding without changing CWD. Relative paths might fail.")

    # --- Run Main Async Function ---
    final_exit_code = 1 # Default error code
    try:
        # Use asyncio.run() to manage the event loop lifecycle for main_async_loop
        final_exit_code = asyncio.run(main_async_loop())

    except RuntimeError as e:
        # Handle common asyncio loop issues, especially in IDEs like Spyder
        if "Cannot run the event loop while another loop is running" in str(e) or "no running event loop" in str(e).lower():
             logger.warning(f"Asyncio loop issue ('{e}'). Attempting fallback Qt execution.");
             app = QApplication.instance(); app = QApplication(sys.argv) if app is None else app; app.setStyle("Fusion")
             agent_orchestrator_fb = None
             try:
                 agent_orchestrator_fb = EnhancedConsciousAgent(); window = EnhancedGameGUI(agent_orchestrator_fb); window.show();
                 final_exit_code = app.exec_() # Run standard Qt loop
             except Exception as gui_err: logger.critical(f"Fallback Init Error: {gui_err}"); QMessageBox.critical(None, "Fatal Init Error", f"Failed fallback init:\n{gui_err}"); final_exit_code=1
             finally: # Ensure cleanup in fallback
                  if agent_orchestrator_fb and hasattr(agent_orchestrator_fb, 'cleanup'):
                       try: agent_orchestrator_fb.cleanup()
                       except Exception as clean_err: logger.error(f"Fallback cleanup error: {clean_err}")
        else: # Re-raise other RuntimeErrors
             logger.critical(f"Unhandled RuntimeError: {e}", exc_info=True); final_exit_code = 1
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught at top level. Exiting cleanly.")
        final_exit_code = 0
    except Exception as e:
        logger.critical(f"Unhandled Exception at top level: {e}", exc_info=True); final_exit_code = 1

    logger.info(f"Exiting application with final code: {final_exit_code}")
    sys.exit(final_exit_code)


# --- END OF FILE main.py ---