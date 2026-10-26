"""Explicit loopback-only launch; never starts on workspace open."""

import sys
from pathlib import Path


def main():
    from streamlit.web.cli import main as streamlit_main

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).with_name("ui.py")),
        "--theme.base=" + str(Path(__file__).with_name("theme.toml")),
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
        "--server.runOnSave=false",
        "--server.maxUploadSize=8",
        "--server.enableStaticServing=false",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
        "--client.showErrorDetails=false",
        "--client.toolbarMode=minimal",
    ]
    streamlit_main()


if __name__ == "__main__":
    main()
