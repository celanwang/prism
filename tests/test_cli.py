import subprocess
import sys

from prism import __version__


def test_module_entrypoint_prints_scaffolding_banner() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "prism"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in result.stdout
    assert "P0" in result.stdout
