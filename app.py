# Root-level entry point for Streamlit Community Cloud deployment.
# Delegates to the actual app inside streamlit_app/.
import runpy, os, sys

# Ensure the project root is on sys.path so all imports work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app", "app.py"),
    run_name="__main__",
)
