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
    from qasync import QEventLoop
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
# IMPORTANT: Importing config now also initializes the tokenizer and validates config
try:
    from config import MasterConfig as Config # Use instantiated config
    from config import DEVICE, log_file, TRAIN_DATA # Get definitive log_file name and TRAIN_DATA
    from orchestrator import EnhancedConsciousAgent
    from main_gui import EnhancedGameGUI
    # --- ADDED: Import BPE initializer ---
    from utils import initialize_bpe_tokenizer
    # ---
except ImportError as e:
     initial_msg = f"CRITICAL ERROR: Failed to import core modules: {e}. Check configuration and dependencies (PyQt5, qasync, sentence-transformers, transformers, datasets)." # Added transformers/datasets
     print(initial_msg)
     try:
         # Attempt basic logging if config failed early
         logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
         logging.critical(initial_msg, exc_info=True)
     except Exception: pass
     sys.exit(1)
except Exception as e:
    # Catch other potential errors during import/config execution (e.g., model loading fail in orchestrator)
    initial_msg = f"CRITICAL ERROR: Unexpected error during initial imports/config setup: {e}"
    print(initial_msg)
    try:
         logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
         logging.critical(initial_msg, exc_info=True)
    except Exception: pass
    sys.exit(1)

# --- Main Async Loop (Accepts args) ---
async def main_async_loop(args): # Accept args
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
    logger.info(f"Agent State Dimension: {Config.Agent.STATE_DIM} (Language Embedding: {'Enabled' if Config.Agent.USE_LANGUAGE_EMBEDDING else 'Disabled'})")


    window: Optional[EnhancedGameGUI] = None # Type hint window
    agent_orchestrator: Optional[EnhancedConsciousAgent] = None # Type hint orchestrator
    loop: Optional[QEventLoop] = None # Type hint loop
    exit_code = 0

    try:
        # --- ADDED: Initialize BPE Tokenizer (before Agent/Orchestrator) ---
        # This ensures it's ready if any component *other* than TransformerGPT needs it.
        # It uses TRAIN_DATA and Config.NLP which should be loaded by now.
        initialize_bpe_tokenizer(TRAIN_DATA, Config.NLP)
        # --- END ADDED ---

        # Initialize agent orchestrator (which loads models) and GUI
        logger.info("Initializing Agent Orchestrator and GUI...")
        agent_orchestrator = EnhancedConsciousAgent() # Handles ST model loading

        # --- ADDED: Load agent state if requested ---
        if args.load:
            logger.info("Load argument provided. Attempting to load saved agent state...")
            if agent_orchestrator.load_agent():
                 logger.info("Agent state loaded successfully.")
            else:
                 logger.error("Failed to load agent state. Continuing with fresh agent.")
        # --- END ADDED ---

        window = EnhancedGameGUI(agent_orchestrator)
        window.show()
        logger.info("Agent orchestrator and GUI initialization complete.")

        # Setup and run the event loop
        loop = QEventLoop(app);
        asyncio.set_event_loop(loop);
        logger.info("Starting Qt event loop integrated with asyncio...")
        await loop.run_forever() # Correct way to run in async function


    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown.")
        exit_code = 0
    except SystemExit as e:
        logger.info(f"System exit called with code: {e.code}");
        exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        logger.critical(f"Unhandled exception during application run: {e}", exc_info=True);
        try:
            parent_widget = window if window and isinstance(window, QWidget) else None
            QMessageBox.critical(parent_widget, "Fatal Runtime Error", f"An unexpected error occurred:\n{e}\nSee log '{log_file}'.")
        except Exception as mb_err: logger.error(f"Failed to show error message box: {mb_err}")
        exit_code = 1
    finally:
        logger.info("--- Initiating Application Shutdown ---")

        # --- ADDED: Save agent state on exit (optional) ---
        if args.save_on_exit and agent_orchestrator:
            logger.info("Save on exit requested. Saving agent state...")
            agent_orchestrator.save_agent()
        # --- END ADDED ---

        # --- Cleanup Resources ---
        # 1. Stop Asyncio loop FIRST
        if loop:
            try:
                if loop.is_running():
                    logger.info("Stopping asyncio event loop...")
                    loop.stop()
                await asyncio.sleep(0.2)
                if not loop.is_closed():
                    logger.info("Closing asyncio loop object.")
                    loop.close()
                logger.info("Asyncio loop stopped and closed.")
            except RuntimeError as re:
                logger.warning(f"RuntimeError during loop stop/close: {re}")
            except Exception as e:
                logger.error(f"Error stopping/closing asyncio loop: {e}", exc_info=True)


        # 2. Close Window (triggers GUI closeEvent -> orchestrator.cleanup -> avatar.cleanup)
        if window and window.isVisible():
            logger.debug("Closing main window...")
            try:
                window.close()
                app.processEvents()
                await asyncio.sleep(0.1)
            except Exception as e:
                 logger.error(f"Error during window close: {e}", exc_info=True)

        # 3. Explicit Orchestrator cleanup (as a fallback)
        # Note: window.close() should trigger agent_orchestrator.cleanup()
        # This is just an extra safety net if the GUI close event fails.
        # if agent_orchestrator and hasattr(agent_orchestrator, 'cleanup'):
        #     try:
        #         logger.info("Explicitly calling orchestrator cleanup (fallback)...")
        #         # agent_orchestrator.cleanup() # Already called via closeEvent normally
        #     except Exception as e:
        #         logger.error(f"Error during explicit orchestrator cleanup: {e}", exc_info=True)


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
            print(f"Debug: Added script directory to sys.path: {script_dir}") # Log might not be ready
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

    # --- Run Main Async Function ---
    final_exit_code = 1
    try:
        # Pass parsed arguments to the main loop
        final_exit_code = asyncio.run(main_async_loop(cli_args))
    # ... (Error handling / Fallback logic remains the same) ...
    except RuntimeError as e:
        if "Cannot run the event loop while another loop is running" in str(e) or "no running event loop" in str(e).lower():
             print(f"Warning: Asyncio loop issue ('{e}'). This can happen in some IDEs (like older Spyder versions). Trying fallback Qt execution (async learning may be limited).");
             app_fb = QApplication.instance(); app_fb = QApplication(sys.argv) if app_fb is None else app_fb; app_fb.setStyle("Fusion")
             agent_orchestrator_fb = None
             window_fb = None
             try:
                 initialize_bpe_tokenizer(TRAIN_DATA, Config.NLP) # Init BPE in fallback too
                 agent_orchestrator_fb = EnhancedConsciousAgent();
                 if cli_args.load: agent_orchestrator_fb.load_agent()
                 window_fb = EnhancedGameGUI(agent_orchestrator_fb); window_fb.show();
                 final_exit_code = app_fb.exec_()
                 if cli_args.save_on_exit and agent_orchestrator_fb: agent_orchestrator_fb.save_agent()
             except Exception as gui_err:
                 print(f"CRITICAL ERROR: Fallback Init/Execution Error: {gui_err}")
                 try: QMessageBox.critical(None, "Fatal Fallback Error", f"Failed fallback init/run:\n{gui_err}");
                 except: pass
                 final_exit_code=1
             finally:
                  if window_fb and window_fb.isVisible(): window_fb.close()
        else:
             print(f"CRITICAL ERROR: Unhandled RuntimeError: {e}")
             final_exit_code = 1
    except KeyboardInterrupt:
        print("Info: KeyboardInterrupt caught at top level. Exiting.")
        final_exit_code = 0
    except Exception as e:
        print(f"CRITICAL ERROR: Unhandled Exception at top level: {e}")
        final_exit_code = 1

    print(f"Info: Exiting application with final code: {final_exit_code}")
    logging.shutdown()
    sys.exit(final_exit_code)


# --- END OF FILE main.py ---
