"""Watchdog supervisor for the Arena Hero tactic.

Runs the tactic as a child process and restarts it whenever it exits. Creates
pid, stop, and log files in the same directory. Stop cleanly by creating the
stop file, sending SIGTERM to the supervisor, or running `python supervisor.py
stop` from this directory. The stop file is a one-shot request: it is cleared
at startup so a stopped supervisor always starts again.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Arena Hero production is POSIX.
    fcntl = None

BASE = Path(__file__).resolve().parent
PYTHON = BASE / ".venv" / "bin" / "python"
TACTIC = BASE / "tactic.py"
ENV_FILE = BASE / ".env"
STATE_FILE = BASE / "scout_state.json"
LOG_FILE = BASE / "supervisor.log"
PID_FILE = BASE / "supervisor.pid"
STOP_FILE = BASE / "supervisor.stop"
LOCK_FILE = BASE / "supervisor.lock"
CHILD_LOG = BASE / "tactic.log"

BACKOFF_MIN = 5
BACKOFF_MAX = 60
UPTIME_RESET = 60
SHUTDOWN_GRACE = 10
SUPERVISOR_LOG_MAX_BYTES = 1 * 1024 * 1024
SUPERVISOR_LOG_BACKUPS = 3

log = logging.getLogger("supervisor")

_child: subprocess.Popen | None = None
_stopping = False
_lock_handle = None


def _read_pid() -> int | None:
    try:
        value = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    if value <= 0:
        return None
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return value
    return value


def _write_pid(pid: int) -> None:
    PID_FILE.write_text(f"{pid}\n")


def _acquire_lock() -> bool:
    """Hold an advisory process lock for the supervisor lifetime."""

    global _lock_handle
    if fcntl is None:
        return _read_pid() is None
    handle = LOCK_FILE.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return False
    _lock_handle = handle
    return True


def _release_lock() -> None:
    global _lock_handle
    if _lock_handle is None:
        return
    if fcntl is not None:
        fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_UN)
    _lock_handle.close()
    _lock_handle = None


def _lock_is_held() -> bool:
    """Return whether another process currently owns the supervisor lock."""

    if fcntl is None:
        return _read_pid() is not None
    handle = LOCK_FILE.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _pid_is_our_supervisor(pid: int) -> bool:
    """Verify a PID belongs to this workspace's running supervisor script.

    Older deployed supervisors predate ``LOCK_FILE``.  Checking their command
    line lets the stop command still reach them without trusting a stale PID
    file whose number may have been reused by an unrelated process.
    """

    process_dir = Path("/proc") / str(pid)
    try:
        raw_args = process_dir.joinpath("cmdline").read_bytes().split(b"\0")
        cwd = Path(os.readlink(process_dir / "cwd"))
    except (OSError, ValueError):
        return False
    args = [os.fsdecode(arg) for arg in raw_args if arg]
    return _command_runs_our_supervisor(args, cwd)


def _command_runs_our_supervisor(args: list[str], cwd: Path) -> bool:
    """Match a process command against this workspace's supervisor entrypoint."""

    expected = (BASE / Path(__file__).name).resolve()
    for index, arg in enumerate(args):
        if Path(arg).name != expected.name:
            continue
        candidate = Path(arg)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            matches = candidate.resolve() == expected
        except OSError:
            matches = False
        if matches and "stop" not in args[index + 1 :]:
            return True
    return False


def _remove_pid() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def _request_stop() -> None:
    STOP_FILE.touch()


def _clear_stop() -> None:
    """Consume a pending stop request so the next run is not a silent no-op."""

    try:
        STOP_FILE.unlink()
    except FileNotFoundError:
        pass


def _should_run() -> bool:
    return not _stopping and not STOP_FILE.exists()


def _terminate(signum, frame) -> None:
    """Ask the loop to stop and take the child down with us.

    Raising SystemExit here would unwind the supervisor while leaving the
    tactic running as an orphan, so the child is signalled explicitly instead.
    """

    global _stopping
    _stopping = True
    log.info("received signal %s, stopping", signum)
    _request_stop()
    if _child is not None and _child.poll() is None:
        _child.terminate()


def _sleep(seconds: float) -> None:
    """Sleep in short slices so a stop request is noticed promptly."""

    deadline = time.monotonic() + seconds
    while _should_run():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))


def _stop_running_supervisor() -> int:
    """Implement the `stop` subcommand."""

    _request_stop()
    lock_held = _lock_is_held()
    pid = _read_pid()
    if pid is None or not _pid_is_our_supervisor(pid):
        print("no running supervisor found; stop file created")
        return 0
    if not lock_held:
        print(f"stopping legacy supervisor {pid} without a process lock")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"could not signal supervisor {pid}: {exc}")
        return 1
    for _ in range(SHUTDOWN_GRACE * 10):
        if _read_pid() is None:
            print(f"supervisor {pid} stopped")
            return 0
        time.sleep(0.1)
    print(f"supervisor {pid} did not exit within {SHUTDOWN_GRACE}s")
    return 1


def _stop_child(child: subprocess.Popen) -> int:
    """Terminate one child with a bounded wait and forced-kill fallback."""

    if child.poll() is not None:
        return child.wait()
    child.terminate()
    try:
        return child.wait(timeout=SHUTDOWN_GRACE)
    except subprocess.TimeoutExpired:
        log.warning("tactic did not exit in time; killing pid %s", child.pid)
        child.kill()
        return child.wait()


def _wait_for_child(child: subprocess.Popen) -> int:
    """Wait while polling the stop file instead of blocking indefinitely."""

    while True:
        try:
            return child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            if not _should_run():
                return _stop_child(child)


def run() -> None:
    global _child

    if not _acquire_lock():
        existing = _read_pid()
        log.error("supervisor already running with pid %s", existing)
        sys.exit(1)
    try:
        if os.name == "posix":
            signal.signal(signal.SIGTERM, _terminate)
            signal.signal(signal.SIGINT, _terminate)

        _clear_stop()
        _write_pid(os.getpid())
        log.info("supervisor started (pid %s), child: %s", os.getpid(), TACTIC)

        backoff = BACKOFF_MIN
        while _should_run():
            # The tactic owns this file through RotatingFileHandler, so it can
            # rotate while a child stays alive for days.  Keep the supervisor's
            # stdout/stderr detached to avoid a second unbounded append stream.
            _child = subprocess.Popen(
                [
                    str(PYTHON),
                    "-u",
                    str(TACTIC),
                    "--env",
                    str(ENV_FILE),
                    "--state",
                    str(STATE_FILE),
                    "--log-file",
                    str(CHILD_LOG),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=BASE,
            )
            started_at = time.monotonic()
            log.info("tactic started (pid %s)", _child.pid)
            code = _wait_for_child(_child)
            uptime = time.monotonic() - started_at
            if not _should_run():
                log.info("stop requested; tactic exited with code %s", code)
                break
            log.warning(
                "tactic exited with code %s after %.1fs; restarting", code, uptime
            )
            if uptime >= UPTIME_RESET:
                backoff = BACKOFF_MIN
            _sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
    finally:
        if _child is not None and _child.poll() is None:
            _stop_child(_child)
        _remove_pid()
        _release_lock()
        log.info("supervisor stopped")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            RotatingFileHandler(
                LOG_FILE,
                maxBytes=SUPERVISOR_LOG_MAX_BYTES,
                backupCount=SUPERVISOR_LOG_BACKUPS,
                encoding="utf-8",
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if len(sys.argv) > 1:
        if sys.argv[1] == "stop":
            sys.exit(_stop_running_supervisor())
        print(f"usage: {Path(sys.argv[0]).name} [stop]", file=sys.stderr)
        sys.exit(2)
    run()


if __name__ == "__main__":
    main()
