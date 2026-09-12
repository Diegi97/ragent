import asyncio
import os
import signal
from pathlib import Path

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.config import (
    PiThinkingLevel,
)

PI_CODING_AGENT_DIRECTORY_ENV = "PI_CODING_AGENT_DIR"


PI_PHOENIX_EXTENSION = Path("npm/node_modules/pi-phoenix/index.ts")


PI_TIMEOUT_SECONDS = 45 * 60


async def run_pi(
    *,
    prompt: str,
    model: str,
    thinking: PiThinkingLevel | None,
    working_directory: Path,
    session_directory: Path,
    session_name: str,
    system_instructions: str,
    environment: dict[str, str],
) -> None:
    working_directory = working_directory.resolve()
    if not working_directory.is_dir():
        raise ValueError(f"PI working directory does not exist: {working_directory}")
    session_directory = session_directory.resolve()
    if not session_directory.is_dir():
        raise ValueError(f"PI session directory does not exist: {session_directory}")
    command = [
        "pi",
        "--print",
        "--model",
        model,
        "--session-dir",
        str(session_directory),
        "--name",
        session_name,
        "--no-extensions",
        "--extension",
        str(_pi_phoenix_extension()),
    ]
    if thinking is not None:
        command.extend(["--thinking", thinking.value])
    command.extend(
        [
            "--system-prompt",
            system_instructions,
            prompt,
        ]
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=working_directory,
        env={**os.environ, **environment},
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        # Timeout/cancellation must also stop any tools spawned by Pi.
        start_new_session=True,
    )
    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=PI_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        await _stop_process(process)
        raise TimeoutError(
            f"PI exceeded its {PI_TIMEOUT_SECONDS:g}-second timeout in {working_directory}"
        ) from exc
    except asyncio.CancelledError:
        await _stop_process(process)
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"PI failed in {working_directory} with exit code {process.returncode}:\n"
            f"{stderr.decode(errors='replace')}"
        )


def _pi_phoenix_extension() -> Path:
    configured_directory = os.getenv(PI_CODING_AGENT_DIRECTORY_ENV, "").strip()
    agent_directory = Path(configured_directory or "~/.pi/agent").expanduser()
    extension = (agent_directory / PI_PHOENIX_EXTENSION).resolve()
    if not extension.is_file():
        raise FileNotFoundError(
            f"pi-phoenix extension not found at {extension}; install it with "
            "'pi install npm:pi-phoenix' or set PI_CODING_AGENT_DIR"
        )
    return extension


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
