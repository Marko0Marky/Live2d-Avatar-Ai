# --- START OF FILE main.py ---

import sys
import os
import asyncio
import logging
import concurrent.futures
import time
from typing import Optional
import argparse # Import argparse

# --- PyQt5 / QAsync Imports ---
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget
    import qasync
    from qasync import QEventLoop, QApplication # Import QApplication from qasync as well
except ImportError as e:
     print(f"CRITICAL ERROR: PyQt5 or qasync import failed: {e}")
     print("Please install PyQt5, qasync, and potentially sentence-transformers:")
     print("  pip install PyQt5 qasync sentence-transformers")
     sys.exit(1)

# --- Setup logging ---
# Logger is configured within config.py now
log_file = "vr_avatar_ai5_run.log" # Define log file name for potential error messages before config load
logger = logging.getLogger(__name__) # Get root logger initially

# --- Import Configuration and Core Components ---
try:
    from config import MasterConfig as Config # Use instantiated config
    from config import DEVICE, log_file, TRAIN_DATA # Get definitive log_file name and TRAIN_DATA
    from orchestrator import EnhancedConsciousAgent
    from main_gui import EnhancedGameGUI
    from utils import initialize_bpe_tokenizer
except ImportError as e:
     initial_msg = f"CRITICAL ERROR: Failed to import core modules: {e}. Check configuration and dependencies (PyQt5, qasync, sentence-transformers, transformers, datasets, accelerate)."
     print(initial_msg)
     try:
         logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
         logging.critical(initial_msg, exc_info=True)
     except Exception: pass
     sys.exit(1)
except Exception as e:
    initial_msg = f"CRITICAL ERROR: Unexpected error during initial imports/config setup: {e}"
    print(initial_msg)
    try:
         logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
         logging.critical(initial_msg, exc_info=True)
    except Exception: pass
    sys.exit(1)

# --- Main Async Loop (Accepts args) ---
async def main_async_loop(args): # Accept args
    """Sets up the Agent Orchestrator and GUI.
       Relies on qasync.run to manage the QApplication and event loop.
    """
    # qasync.run handles QApplication instance and loop setup implicitly

    logger.info("--- Application Starting (within async loop) ---")
    logger.info(f"Using PyTorch device: {DEVICE}")
    logger.info(f"Target Live2D Model Path: {Config.Graphics.MODEL_PATH}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Agent State Dimension: {Config.Agent.STATE_DIM} (Language Embedding: {'Enabled' if Config.Agent.USE_LANGUAGE_EMBEDDING else 'Disabled'})")

    window: Optional[EnhancedGameGUI] = None # Type hint window
    agent_orchestrator: Optional[EnhancedConsciousAgent] = None # Type hint orchestrator
    exit_code = 0 # Keep track of potential errors

    try:
        initialize_bpe_tokenizer(TRAIN_DATA, Config.NLP)

        logger.info("Initializing Agent Orchestrator and GUI...")
        agent_orchestrator = EnhancedConsciousAgent()

        if args.load:
            logger.info("Load argument provided. Attempting to load saved agent state...")
            if agent_orchestrator.load_agent():
                 logger.info("Agent state loaded successfully.")
            else:
                 logger.error("Failed to load agent state. Continuing with fresh agent.")

        window = EnhancedGameGUI(agent_orchestrator)
        window.show()
        logger.info("Agent orchestrator and GUI initialization complete.")

        # --- NO loop.run_forever() needed here ---
        # qasync.run() manages the main execution loop externally.
        # We need a way to keep this coroutine alive until the app quits.
        # We can wait for the window to close.

        # Create a future that completes when the window is closed
        closed = asyncio.Future()
        original_closeEvent = window.closeEvent
        def close_event_wrapper(event):
            # Call original handler first (which might call agent cleanup)
            original_closeEvent(event)
            # If the event was accepted (window is closing), set the future result
            if event.isAccepted() and not closed.done():
                closed.set_result(True)
        window.closeEvent = close_event_wrapper # Monkey-patch closeEvent

        logger.info("Async loop waiting for window close...")
        await closed # Keep the coroutine alive until the window closes

        logger.info("Window closed, proceeding to shutdown.")


    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown.")
        exit_code = 0
    except SystemExit as e:
        logger.info(f"System exit called with code: {e.code}");
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        logger.critical(f"Unhandled exception during application run: {e}", exc_info=True);
        try:
            # Get app instance for message box parent
            app = QApplication.instance()
            parent_widget = window if window and isinstance(window, QWidget) else None
            if app: # Only show if app exists
                 QMessageBox.critical(parent_widget, "Fatal Runtime Error", f"An unexpected error occurred:\n{e}\nSee log '{log_file}'.")
        except Exception as mb_err: logger.error(f"Failed to show error message box: {mb_err}")
        exit_code = 1
    finally:
        logger.info("--- Initiating Application Shutdown (within async loop finally) ---")

        # --- Save agent state on exit (optional) ---
        # This should happen *before* orchestrator cleanup
        if args.save_on_exit and agent_orchestrator:
            logger.info("Save on exit requested. Saving agent state...")
            agent_orchestrator.save_agent()

        # --- Explicit Cleanup (Triggered AFTER window closed normally) ---
        # Note: window.closeEvent already called orchestrator.cleanup()
        # If we reach here due to an exception *before* window close,
        # we might need explicit cleanup. However, the closeEvent patch
        # ensures cleanup is tied to the window closing properly.
        if agent_orchestrator and hasattr(agent_orchestrator, 'cleanup'):
            # Double-check if cleanup was already done (e.g., add a flag in orchestrator)
            # For simplicity, we assume calling it again is safe or handled internally.
             try:
                 logger.debug("Ensuring orchestrator cleanup is called...")
                 # agent_orchestrator.cleanup() # Already called by closeEvent wrapper
             except Exception as e:
                 logger.error(f"Error during explicit orchestrator cleanup: {e}", exc_info=True)


        # --- Loop stopping/closing is handled by qasync.run implicitly ---
        logger.info(f"Async loop finished. Shutdown should be handled by qasync.run.")

    # qasync.run will return the application exit code
    # We track 'exit_code' mainly for logging exceptions
    return exit_code


if __name__ == "__main__":
    # --- Path and CWD Setup ---
    script_dir = os.getcwd()
    try:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            print(f"Debug: Added script directory to sys.path: {script_dir}")
    except NameError:
        print("Warning: __file__ not defined, using CWD for path setup.")
        script_dir = os.getcwd()
    try:
        os.chdir(script_dir)
        print(f"Info: Changed CWD to script directory: {script_dir}")
    except Exception as e:
         print(f"ERROR: Failed to change CWD to script directory '{script_dir}': {e}")
         print("Warning: Proceeding without changing CWD. Relative paths might fail.")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Live2D Avatar AI Agent")
    parser.add_argument("--load", action="store_true", help="Load saved agent state on startup.")
    parser.add_argument("--save-on-exit", action="store_true", help="Save agent state automatically on exit.")
    cli_args = parser.parse_args()

    # --- Run Main Async Function using qasync.run ---
    final_exit_code = 1 # Default error code
    try:
        # qasync.run handles creating QApplication and integrating the loop
        final_exit_code = qasync.run(main_async_loop(cli_args))

    # Fallback logic remains the same...
    except RuntimeError as e:
        if "Cannot run the event loop while another loop is running" in str(e) or "no running event loop" in str(e).lower():
             print(f"Warning: Asyncio loop issue ('{e}'). Trying fallback Qt execution (async learning may be limited).");
             # Use standard QApplication directly for fallback
             app_fb = QApplication.instance(); app_fb = QApplication(sys.argv) if app_fb is None else app_fb; app_fb.setStyle("Fusion")
             agent_orchestrator_fb = None
             window_fb = None
             try:
                 initialize_bpe_tokenizer(TRAIN_DATA, Config.NLP)
                 agent_orchestrator_fb = EnhancedConsciousAgent();
                 if cli_args.load: agent_orchestrator_fb.load_agent()
                 window_fb = EnhancedGameGUI(agent_orchestrator_fb); window_fb.show();
                 final_exit_code = app_fb.exec_() # Standard Qt execution
                 if cli_args.save_on_exit and agent_orchestrator_fb:
                     print("Info: Fallback mode - Saving agent state on exit...")
                     agent_orchestrator_fb.save_agent()
             except Exception as gui_err:
                 print(f"CRITICAL ERROR: Fallback Init/Execution Error: {gui_err}")
                 try: QMessageBox.critical(None, "Fatal Fallback Error", f"Failed fallback init/run:\n{gui_err}");
                 except: pass
                 final_exit_code=1
             finally:
                  if window_fb and window_fb.isVisible(): window_fb.close()
                  if agent_orchestrator_fb:
                      print("Info: Fallback mode - Cleaning up orchestrator...")
                      agent_orchestrator_fb.cleanup() # Explicit cleanup
        else:
             print(f"CRITICAL ERROR: Unhandled RuntimeError: {e}")
             try: logger.critical("Unhandled RuntimeError", exc_info=True)
             except: pass
             final_exit_code = 1
    except KeyboardInterrupt:
        print("Info: KeyboardInterrupt caught at top level. Exiting.")
        final_exit_code = 0
    except Exception as e:
        print(f"CRITICAL ERROR: Unhandled Exception at top level: {e}")
        try: logger.critical("Unhandled Exception at top level", exc_info=True)
        except: pass
        final_exit_code = 1

    print(f"Info: Exiting application with final code: {final_exit_code}")
    logging.shutdown() # Ensure all logs are flushed
    sys.exit(final_exit_code)


# --- END OF FILE main.py ---
