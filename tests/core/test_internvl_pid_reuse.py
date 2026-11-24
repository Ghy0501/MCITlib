# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Protect long-lived InternVL supervisors from unsafe PID reuse assumptions."""
import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from mcitlib import internvl_process_guard as guard


def row(pid, ticks, *, ppid=100, session=100):
    return {'pid': pid, 'ticks': ticks, 'ppid': ppid, 'session': session, 'pgrp': session, 'state': 'S'}


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19])


def item(fd, identity):
    return {'pidfd': fd, 'ticks': identity['ticks'], 'first': identity, 'last': identity, 'signals': []}


def tracker():
    result = guard.OwnedProcesses.__new__(guard.OwnedProcesses)
    result.parent = 100
    result.child = type('Child', (), {'pid': 123456})()
    result.leader_ticks = 1
    result.known = {}
    result.retired = []
    return result


class InternvlPidReuseTests(unittest.TestCase):
    def test_reused_pid_requires_real_exited_pidfd_before_repin(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.15)'])
        fd = guard.pidfd_open(child.pid)
        old = row(child.pid, ticks(child.pid))
        child.wait(timeout=5)
        t = tracker()
        t.known[child.pid] = item(fd, old)
        replacement = row(child.pid, old['ticks'] + 100)
        new_fd = guard.pidfd_open(os.getpid())
        try:
            with mock.patch.object(guard, 'pidfd_open', return_value=new_fd), \
                 mock.patch.object(guard, 'process_table', return_value={child.pid: replacement}), \
                 mock.patch.object(guard, 'pidfd_signal') as send:
                t._register(replacement)
            self.assertEqual(t.known[child.pid]['ticks'], replacement['ticks'])
            self.assertEqual(t.retired[0]['start_ticks'], old['ticks'])
            self.assertTrue(t.retired[0]['pidfd_exit_mask'])
            send.assert_not_called()
            with self.assertRaises(OSError):
                os.fstat(fd)
        finally:
            os.close(new_fd)

    def test_live_pinned_identity_is_neither_retired_nor_signalled(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.3)'])
        fd = guard.pidfd_open(child.pid)
        old = row(child.pid, ticks(child.pid))
        t = tracker()
        t.known[child.pid] = item(fd, old)
        try:
            with mock.patch.object(guard, 'pidfd_open') as opener, mock.patch.object(guard, 'pidfd_signal') as send:
                with self.assertRaisesRegex(RuntimeError, 'not proven exited'):
                    t._register(row(child.pid, old['ticks'] + 1))
            self.assertEqual(t.known[child.pid]['pidfd'], fd)
            self.assertFalse(t.retired)
            opener.assert_not_called()
            send.assert_not_called()
        finally:
            child.wait(timeout=5)
            os.close(fd)

    def test_foreign_reused_pid_is_not_registered_or_signalled(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.15)'])
        old_fd = guard.pidfd_open(child.pid)
        old = row(child.pid, ticks(child.pid))
        child.wait(timeout=5)
        leader_fd = guard.pidfd_open(os.getpid())
        leader = row(123456, 10, session=123456)
        foreign = row(child.pid, old['ticks'] + 1, ppid=999, session=999)
        t = tracker()
        t.known = {leader['pid']: item(leader_fd, leader), old['pid']: item(old_fd, old)}
        try:
            with mock.patch.object(guard, 'process_table', return_value={leader['pid']: leader, foreign['pid']: foreign}), \
                 mock.patch.object(guard, 'pidfd_open') as opener, mock.patch.object(guard, 'pidfd_signal') as send:
                t.signal_owned(signal.SIGTERM)
            self.assertNotIn(old['pid'], t.known)
            self.assertEqual(t.retired[0]['replacement_start_ticks'], foreign['ticks'])
            opener.assert_not_called()
            send.assert_called_once_with(leader_fd, signal.SIGTERM)
        finally:
            os.close(leader_fd)

    def test_exited_absent_identity_releases_pidfd(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.15)'])
        fd = guard.pidfd_open(child.pid)
        old = row(child.pid, ticks(child.pid))
        child.wait(timeout=5)
        t = tracker()
        t.known[old['pid']] = item(fd, old)
        with mock.patch.object(guard, 'process_table', return_value={}), \
             mock.patch.object(guard, 'pidfd_signal') as send:
            self.assertEqual(t.refresh(), {})
        self.assertFalse(t.known)
        self.assertIsNone(t.retired[0]['replacement_start_ticks'])
        with self.assertRaises(OSError):
            os.fstat(fd)
        send.assert_not_called()

    def test_invalid_fd_is_not_exit_proof(self):
        fd = guard.pidfd_open(os.getpid())
        os.close(fd)
        with self.assertRaisesRegex(RuntimeError, 'invalid'):
            guard.pidfd_exit_mask(fd)


if __name__ == '__main__':
    unittest.main()
