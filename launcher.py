from pathlib import Path
import sys

from streamlit.web import cli as stcli


def _runtime_base_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent)).resolve()


def _app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).parent.resolve()


def _resolve_app_script() -> Path:
    candidates = [
        _runtime_base_dir() / "streamlit_app_full.py",
        _app_base_dir() / "streamlit_app_full.py",
        Path.cwd() / "streamlit_app_full.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate streamlit_app_full.py")


def main() -> int:
    app_script = _resolve_app_script()
    sys.argv = [
        "streamlit",
        "run",
        str(app_script),
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--browser.serverAddress=127.0.0.1",
        "--browser.serverPort=8501",
        "--global.developmentMode=false",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
