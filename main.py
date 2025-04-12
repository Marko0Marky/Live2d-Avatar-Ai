# --- START OF FILE main.py ---

import sys
import os
import asyncio
import logging
import concurrent.futures
import time
import argparse
import signal # Added for signal handling
from pathlib import Path # Added for path checking
from typing import Optional, List # Added typing

# --- PyQt5 / QAsync Imports ---
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget
    import qasync
    from qasync import QEventLoop, QApplication as QAsyncApplication
    from PyQt5.QtWidgets import QApplication as StandardQApplication
    PYQT_AVAILABLE = True
except ImportError as e:
    print(f"CRITICAL ERROR: PyQt5 or qasync import failed: {e}")
    print("Please install required GUI libraries:")
    print("  pip install PyQt5 qasync")
    PYQT_AVAILABLE = False
    sys.exit(1) # Exit early if core GUI components are missing

# --- Setup logging ---
logger = logging.getLogger(__name__)
log_file = "vr_avatar_ai5_run.log" # Default log file name

# --- Import Configuration and Core Components ---
try:
    from config import MasterConfig as Config
    from config import DEVICE, log_file as configured_log_file
    log_file = configured_log_file # Use configured log file name

    if not hasattr(Config, 'Graphics') or not hasattr(Config.Graphics, 'MODEL_PATH'):
        raise ValueError("Configuration is missing critical Graphics settings (e.g., MODEL_PATH)")
    if not hasattr(Config, 'Agent'):
        raise ValueError("Configuration is missing critical Agent settings")
    logger.info("Configuration loaded successfully.")

    from orchestrator import EnhancedConsciousAgent
    from main_gui import EnhancedGameGUI
    # --- REMOVED BPE IMPORTS ---

    logger.info("Core modules imported successfully.")

except ImportError as e:
    initial_msg = f"CRITICAL ERROR: Failed to import core modules (config, orchestrator, main_gui, utils): {e}. Check configuration and dependencies (e.g., transformers, sentence-transformers, datasets, accelerate)."
    print(initial_msg)
    try:
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
        logging.critical(initial_msg, exc_info=True)
    except Exception: pass
    sys.exit(1)
except ValueError as ve:
    initial_msg = f"CRITICAL ERROR: Invalid or incomplete configuration: {ve}"
    print(initial_msg)
    try:
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
        logging.critical(initial_msg, exc_info=True)
    except Exception: pass
    sys.exit(1)
except Exception as e:
    initial_msg = f"CRITICAL ERROR: Unexpected error during initial imports or config setup: {e}"
    print(initial_msg)
    try:
        logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)])
        logging.critical(initial_msg, exc_info=True)
    except Exception: pass
    sys.exit(1)


def setup_paths_and_cwd():
    """Sets the current working directory to the script's directory."""
    script_dir = os.getcwd() # Default to CWD
    try:
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        logger.debug(f"Script directory identified as: {script_dir}")
    except NameError:
        logger.warning("__file__ not defined, using current working directory for path setup.")
        script_dir = os.getcwd()

    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        logger.debug(f"Added script directory to sys.path: {script_dir}")

    try:
        os.chdir(script_dir)
        logger.info(f"Changed CWD to script directory: {script_dir}")
    except Exception as e:
        logger.error(f"Failed to change CWD to script directory '{script_dir}': {e}", exc_info=True)
        logger.warning("Proceeding without changing CWD. Relative paths might fail.")


def check_live2d_sdk_availability():
    """
    Performs a basic check for the likely location of the Live2D SDK native library.
    This is a heuristic check and might need adjustment.
    """
    sdk_path_env = os.environ.get('LIVE2D_SDK_NATIVE_PATH')
    if sdk_path_env:
        logger.info(f"Checking Live2D SDK path from environment variable: {sdk_path_env}")
        if Path(sdk_path_env).is_file():
            logger.info("Live2D SDK native library found via environment variable.")
            return True
        else:
            logger.warning(f"Live2D SDK path from environment variable '{sdk_path_env}' does not point to a valid file.")

    possible_relative_paths = [
        "Live2D_SDK/Core/dll/linux/x86_64/libLive2DCubismCore.so", # Linux example
        "Live2D_SDK/Core/dll/windows/x86_64/Live2DCubismCore.dll", # Windows example
        "Live2D_SDK/Core/dll/macos/libLive2DCubismCore.dylib",     # macOS example
        "libLive2DCubismCore.so", # Maybe directly in CWD/bundle root
        "Live2DCubismCore.dll",
        "libLive2DCubismCore.dylib",
    ]
    for rel_path in possible_relative_paths:
        # Check relative to current CWD (which should be script dir or bundle dir)
        abs_path = Path(os.getcwd()) / rel_path
        if abs_path.is_file():
            logger.info(f"Found potential Live2D SDK native library at conventional path: {abs_path}")
            return True

    logger.warning("Could not find Live2D SDK native library based on environment variable or common relative paths.")
    logger.warning("Live2D functionality will likely fail. Ensure the native SDK is correctly placed and accessible.")
    return False


async def main_async_loop(args: argparse.Namespace, loop: asyncio.AbstractEventLoop):
    """
    Main asynchronous application loop managed by qasync.
    Initializes and runs the agent orchestrator and GUI.
    """
    logger.info("--- Application Starting (Async Loop) ---")
    logger.info(f"Using PyTorch device: {DEVICE}")
    logger.info(f"Target Live2D Model Path: {Config.Graphics.MODEL_PATH}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Agent State Dimension: {Config.Agent.STATE_DIM} (Language Embedding: {'Enabled' if Config.Agent.USE_LANGUAGE_EMBEDDING else 'Disabled'})")

    window: Optional[EnhancedGameGUI] = None
    agent_orchestrator: Optional[EnhancedConsciousAgent] = None
    app_exit_code = 0
    tasks_to_cancel: List[asyncio.Future] = []

    try:
        check_live2d_sdk_availability()
        # --- REMOVED BPE Tokenizer initialization call ---

        logger.info("Initializing Agent Orchestrator and GUI...")
        agent_orchestrator = EnhancedConsciousAgent()

        if args.load:
             logger.info("Load argument provided. Attempting to load saved agent state...")
             if agent_orchestrator.load_agent():
                 logger.info("Agent state loaded successfully.")
             else:
                 logger.warning("Failed to fully load agent state. Continuing.")

        window = EnhancedGameGUI(agent_orchestrator)
        window.show()
        logger.info("Agent orchestrator and GUI initialization complete.")

        window_closed_future = asyncio.Future()
        tasks_to_cancel.append(window_closed_future)
        original_closeEvent = window.closeEvent

        def close_event_wrapper(event):
            original_closeEvent(event)
            if event.isAccepted() and not window_closed_future.done():
                logger.debug("Window close event accepted, setting future result.")
                window_closed_future.set_result(True)
            else:
                logger.debug("Window close event ignored, future not set.")
        window.closeEvent = close_event_wrapper

        logger.info("Async loop running. Waiting for window close or termination signal...")
        await window_closed_future
        logger.info("Window close signal received or loop interrupted.")

    except asyncio.CancelledError:
        logger.info("Main async loop cancelled (likely due to signal). Initiating shutdown.")
        app_exit_code = 0
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received during async loop. Initiating shutdown.")
        app_exit_code = 0
    except SystemExit as e:
        logger.info(f"System exit called during async loop with code: {e.code}")
        app_exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        logger.critical(f"Unhandled exception during application run: {e}", exc_info=True)
        try:
            app = QAsyncApplication.instance()
            parent_widget = window if window and isinstance(window, QWidget) else None
            if app:
                QMessageBox.critical(parent_widget, "Fatal Runtime Error",
                                     f"An unexpected error occurred:\n{e}\n\n"
                                     f"Please check the log file for details:\n'{log_file}'")
        except Exception as mb_err:
            logger.error(f"Failed to show critical error message box: {mb_err}")
        app_exit_code = 1
    finally:
        logger.info("--- Initiating Application Shutdown (Async Loop Finally) ---")

        logger.debug(f"Cancelling {len(tasks_to_cancel)} tracked tasks...")
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        if app_exit_code == 0 and args.save_on_exit and agent_orchestrator:
            logger.info("Save on exit requested. Saving agent state...")
            try:
                agent_orchestrator.save_agent()
                logger.info("Agent state saved successfully on exit.")
            except Exception as save_err:
                logger.error(f"Failed to save agent state on exit: {save_err}", exc_info=True)
        elif args.save_on_exit:
             logger.warning(f"Skipping save on exit due to non-zero exit code ({app_exit_code}) or missing orchestrator.")

        # Check if cleanup already happened via closeEvent
        orchestrator_needs_cleanup = agent_orchestrator and not getattr(agent_orchestrator, 'cleaned_up', False)
        if orchestrator_needs_cleanup:
             logger.warning("Orchestrator cleanup might not have run via closeEvent. Calling directly.")
             try:
                 agent_orchestrator.cleanup()
             except Exception as direct_cleanup_err:
                 logger.error(f"Error during direct orchestrator cleanup: {direct_cleanup_err}", exc_info=True)
        elif agent_orchestrator:
             logger.debug("Skipping redundant cleanup call (already run or orchestrator missing).")


    logger.info(f"Async loop finished with internal exit code: {app_exit_code}")
    return app_exit_code

# --- Signal Handling ---
_shutdown_requested = False
def handle_signal(sig, frame):
    global _shutdown_requested
    if _shutdown_requested:
        logger.warning("Shutdown already requested, ignoring signal.")
        return
    _shutdown_requested = True
    logger.info(f"Received signal {sig}. Requesting graceful shutdown...")
    try:
        loop = asyncio.get_running_loop()
        for task in asyncio.all_tasks(loop):
            task.cancel()
        logger.info("All asyncio tasks cancelled.")
    except RuntimeError:
        logger.warning("No running asyncio loop found during signal handling.")
    app = StandardQApplication.instance() # Use standard Qt app instance for quit
    if app:
        logger.info("Requesting Qt application quit.")
        app.quit()


if __name__ == "__main__":
    if not PYQT_AVAILABLE:
        print("PyQt5/qasync not found. Cannot run the application.")
        sys.exit(1)

    setup_paths_and_cwd()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run Live2D Avatar AI Agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--load", action="store_true",
                        help="Load saved agent state on startup.")
    parser.add_argument("--save-on-exit", action="store_true",
                        help="Save agent state automatically on graceful exit.")
    cli_args = parser.parse_args()
    logger.info(f"Command line arguments: {cli_args}")

    # --- Event Loop and Signal Setup ---
    final_exit_code = 1
    loop = None
    try:
        if sys.platform != "win32":
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        else:
             signal.signal(signal.SIGINT, handle_signal)

        logger.info("Starting application with qasync event loop...")
        loop = asyncio.get_event_loop()
        final_exit_code = qasync.run(main_async_loop(cli_args, loop))
        logger.info("qasync run finished.")

    # --- Fallback Logic ---
    except RuntimeError as e:
         if "Cannot run the event loop while another loop is running" in str(e) or \
            "no running event loop" in str(e).lower():
             logger.warning(f"Asyncio loop issue detected ('{e}'). Attempting fallback Qt execution.")
             logger.warning("--- RUNNING IN FALLBACK QT MODE ---")

             app_fb = StandardQApplication.instance()
             if app_fb is None:
                 logger.debug("Creating new StandardQApplication for fallback.")
                 app_fb = StandardQApplication(sys.argv)
             else:
                 logger.debug("Using existing StandardQApplication instance for fallback.")
             app_fb.setStyle("Fusion")

             agent_orchestrator_fb = None
             window_fb = None
             try:
                 logger.info("Fallback Mode: Initializing Orchestrator and GUI...")
                 check_live2d_sdk_availability()
                 agent_orchestrator_fb = EnhancedConsciousAgent()
                 if cli_args.load:
                     logger.info("Fallback Mode: Attempting to load agent state...")
                     agent_orchestrator_fb.load_agent()

                 window_fb = EnhancedGameGUI(agent_orchestrator_fb)
                 window_fb.show()
                 logger.info("Fallback Mode: Initialization complete. Starting standard Qt event loop.")
                 final_exit_code = app_fb.exec_()
                 logger.info(f"Fallback Mode: Qt event loop finished with code: {final_exit_code}")

                 if cli_args.save_on_exit and agent_orchestrator_fb:
                     logger.info("Fallback Mode: Saving agent state on exit...")
                     try: agent_orchestrator_fb.save_agent()
                     except Exception as fb_save_err: logger.error(f"Fallback Mode: Failed to save: {fb_save_err}", exc_info=True)

             except Exception as fb_init_err:
                 logger.critical(f"CRITICAL ERROR during Fallback Init/Execution: {fb_init_err}", exc_info=True)
                 try: QMessageBox.critical(None, "Fatal Fallback Error", f"Failed fallback init/run:\n{fb_init_err}")
                 except Exception: pass
                 final_exit_code = 1
             finally:
                 logger.info("Fallback Mode: Initiating cleanup...")
                 # Ensure cleanup happens in fallback
                 if agent_orchestrator_fb:
                     logger.debug("Fallback Mode: Calling orchestrator cleanup...")
                     try:
                         # Close window first if it exists, then cleanup orchestrator
                         if window_fb and window_fb.isVisible(): window_fb.close()
                         agent_orchestrator_fb.cleanup()
                     except Exception as fb_clean_err:
                         logger.error(f"Fallback Mode: Error during cleanup: {fb_clean_err}", exc_info=True)
                 logger.info("Fallback Mode: Cleanup finished.")
         else:
             logger.critical(f"CRITICAL ERROR: Unhandled RuntimeError: {e}", exc_info=True)
             final_exit_code = 1
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught at top level (fallback). Exiting.")
        final_exit_code = 0
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Unhandled Exception at top level: {e}", exc_info=True)
        final_exit_code = 1
    finally:
        logger.info(f"--- Application Exiting (Final Code: {final_exit_code}) ---")
        logging.shutdown()

    sys.exit(final_exit_code)

# --- END OF FILE main.py ---
