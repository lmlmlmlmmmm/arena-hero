"""Offline tests for the watchdog's one-shot stop and restart guarantees."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import supervisor


def test_stop_file_is_consumed_once(tmp_path, monkeypatch):
    stop_file = tmp_path / "supervisor.stop"
    monkeypatch.setattr(supervisor, "STOP_FILE", stop_file)
    stop_file.touch()
    supervisor._clear_stop()
    assert not stop_file.exists()
    supervisor._clear_stop()
    assert not stop_file.exists()


def test_terminate_marks_stop_and_signals_the_child(tmp_path, monkeypatch):
    class Child:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    child = Child()
    monkeypatch.setattr(supervisor, "STOP_FILE", tmp_path / "supervisor.stop")
    monkeypatch.setattr(supervisor, "_stopping", False)
    monkeypatch.setattr(supervisor, "_child", child)
    supervisor._terminate(signal.SIGTERM, None)
    assert supervisor.STOP_FILE.exists()
    assert supervisor._stopping is True
    assert child.terminated is True


def test_run_restarts_after_a_crash_with_backoff(tmp_path, monkeypatch):
    class Child:
        def __init__(self, pid, code):
            self.pid = pid
            self.code = code

        def wait(self, timeout=None):
            return self.code

        def poll(self):
            return self.code

    children = [Child(101, 1), Child(102, 0)]
    launches = []

    def fake_popen(command, **kwargs):
        launches.append((command, kwargs))
        return children[len(launches) - 1]

    should_run = iter((True, True, True, False))
    monotonic = iter((0.0, 1.0, 2.0, 3.0))
    sleeps = []
    monkeypatch.setattr(supervisor, "BASE", tmp_path)
    monkeypatch.setattr(supervisor, "PID_FILE", tmp_path / "supervisor.pid")
    monkeypatch.setattr(supervisor, "STOP_FILE", tmp_path / "supervisor.stop")
    monkeypatch.setattr(supervisor, "LOCK_FILE", tmp_path / "supervisor.lock")
    monkeypatch.setattr(supervisor, "_child", None)
    monkeypatch.setattr(supervisor, "_stopping", False)
    monkeypatch.setattr(supervisor, "_lock_handle", None)
    monkeypatch.setattr(supervisor, "_read_pid", lambda: None)
    monkeypatch.setattr(supervisor, "_should_run", lambda: next(should_run))
    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(supervisor.signal, "signal", lambda *args: None)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(supervisor, "_sleep", sleeps.append)
    supervisor.run()

    assert len(launches) == 2
    assert sleeps == [supervisor.BACKOFF_MIN]
    assert "--log-file" in launches[0][0]
    assert not supervisor.PID_FILE.exists()


def test_wait_for_child_observes_a_raw_stop_file(tmp_path, monkeypatch):
    class Child:
        pid = 101

        def __init__(self):
            self.terminated = False
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                supervisor.STOP_FILE.touch()
                raise subprocess.TimeoutExpired("tactic", timeout)
            return -signal.SIGTERM

        def poll(self):
            return -signal.SIGTERM if self.terminated else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            raise AssertionError("graceful termination should not need kill")

    child = Child()
    monkeypatch.setattr(supervisor, "STOP_FILE", tmp_path / "supervisor.stop")
    monkeypatch.setattr(supervisor, "_stopping", False)
    assert supervisor._wait_for_child(child) == -signal.SIGTERM
    assert child.terminated is True


def test_restart_backoff_observes_a_raw_stop_file(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "STOP_FILE", tmp_path / "supervisor.stop")
    monkeypatch.setattr(supervisor, "_stopping", False)
    monkeypatch.setattr(
        supervisor.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(
            AssertionError("a pending stop must skip the backoff sleep")
        ),
    )
    supervisor.STOP_FILE.touch()
    supervisor._sleep(supervisor.BACKOFF_MAX)


def test_process_identity_matches_only_this_supervisor(tmp_path, monkeypatch):
    workspace = tmp_path / "arena-hero"
    workspace.mkdir()
    entrypoint = workspace / "supervisor.py"
    entrypoint.touch()
    monkeypatch.setattr(supervisor, "BASE", workspace)

    assert supervisor._command_runs_our_supervisor(
        ["python", "-u", "supervisor.py"], workspace
    )
    assert supervisor._command_runs_our_supervisor(
        ["python", str(entrypoint)], Path("/")
    )
    assert not supervisor._command_runs_our_supervisor(
        ["python", "supervisor.py"], tmp_path
    )
    assert not supervisor._command_runs_our_supervisor(
        ["python", "supervisor.py", "stop"], workspace
    )


def test_stop_reaches_a_verified_legacy_supervisor(tmp_path, monkeypatch):
    pids = iter((321, None))
    signals = []
    monkeypatch.setattr(supervisor, "STOP_FILE", tmp_path / "supervisor.stop")
    monkeypatch.setattr(supervisor, "_lock_is_held", lambda: False)
    monkeypatch.setattr(supervisor, "_read_pid", lambda: next(pids))
    monkeypatch.setattr(supervisor, "_pid_is_our_supervisor", lambda pid: pid == 321)
    monkeypatch.setattr(supervisor.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    assert supervisor._stop_running_supervisor() == 0
    assert signals == [(321, signal.SIGTERM)]
    assert supervisor.STOP_FILE.exists()


def test_stop_never_signals_a_reused_stale_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "STOP_FILE", tmp_path / "supervisor.stop")
    monkeypatch.setattr(supervisor, "_lock_is_held", lambda: False)
    monkeypatch.setattr(supervisor, "_read_pid", lambda: 321)
    monkeypatch.setattr(supervisor, "_pid_is_our_supervisor", lambda pid: False)
    monkeypatch.setattr(
        supervisor.os,
        "kill",
        lambda pid, sig: (_ for _ in ()).throw(
            AssertionError("an unverified PID must never be signalled")
        ),
    )

    assert supervisor._stop_running_supervisor() == 0
    assert supervisor.STOP_FILE.exists()


def test_supervisor_lock_prevents_a_second_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "LOCK_FILE", tmp_path / "supervisor.lock")
    monkeypatch.setattr(supervisor, "_lock_handle", None)
    assert supervisor._acquire_lock() is True
    try:
        assert supervisor._lock_is_held() is True
    finally:
        supervisor._release_lock()
    assert supervisor._lock_is_held() is False
