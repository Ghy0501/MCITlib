# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Track and clean only this controller's subprocesses, including orphans."""
from __future__ import annotations

import ctypes
import errno
import os
import platform
import select
import signal
import sys
import time
from pathlib import Path

PROC_ROOT = Path('/proc')


def _syscall(number, *args):
    if sys.platform != 'linux' or platform.machine() not in {'x86_64', 'aarch64'}:
        raise RuntimeError('pidfd protection requires Linux on x86_64 or aarch64')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    value = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(arg) for arg in args))
    if value == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(value)


def pidfd_open(pid):
    return os.pidfd_open(pid) if hasattr(os, 'pidfd_open') else _syscall(434, pid, 0)


def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(fd, sig)
    else:
        _syscall(424, fd, sig, 0, 0)


def enable_subreaper():
    if sys.platform != 'linux':
        raise RuntimeError('child subreaper requires Linux')
    libc = ctypes.CDLL(None, use_errno=True)
    # A child which exits after creating an orphan cannot hand it off to PID 1;
    # the orphan is adopted by this dedicated controller, even after setsid().
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    value = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 1:
        raise RuntimeError('cannot confirm PR_GET_CHILD_SUBREAPER')


def process_table():
    """Observe same-owner process identities; denial is not absence."""
    result = {}
    for proc in PROC_ROOT.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            uid_lines = [line for line in (proc / 'status').read_text().splitlines() if line.startswith('Uid:')]
            if len(uid_lines) != 1:
                raise ValueError(f'missing or ambiguous Uid metadata for {proc.name}')
            uids = uid_lines[0].split()[1:]
            if len(uids) != 4:
                raise ValueError(f'invalid Uid metadata for {proc.name}')
            if os.getuid() not in [int(uid) for uid in uids]:
                continue
            line = (proc / 'stat').read_text()
            close = line.rfind(')')
            fields = line[close + 2:].split()
            if close < 0 or len(fields) < 20:
                raise ValueError(f'incomplete process stat for {proc.name}')
            pid = int(proc.name)
            result[pid] = {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                           'pgrp': int(fields[2]), 'session': int(fields[3]), 'ticks': int(fields[19])}
        except (FileNotFoundError, ProcessLookupError):
            # The process exited between directory enumeration and the read.
            continue
        except PermissionError as error:
            raise RuntimeError(f'cannot prove process visibility: {proc}') from error
    return result


def is_live(info):
    return info is not None and info['state'] not in {'Z', 'X', 'x'}


def same_identity(a, b):
    return a is not None and b is not None and all(a[key] == b[key] for key in
                                                    ('pid', 'ticks', 'ppid', 'pgrp', 'session'))


def pidfd_exit_mask(fd):
    """Prove the pinned identity exited; invalid or ambiguous polls are not proof."""
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    events = poller.poll(0)
    if not events:
        return None
    if len(events) != 1 or events[0][0] != fd or events[0][1] & (select.POLLERR | select.POLLNVAL):
        raise RuntimeError('pidfd exit proof is invalid')
    return events[0][1] if events[0][1] & select.POLLIN else None


class OwnedProcesses:
    """One child session at a time, with pidfds for independently proven members."""

    def __init__(self):
        enable_subreaper()
        self.parent = os.getpid()
        self.child = None
        self.leader_ticks = None
        self.known = {}
        self.retired = []
        self.history = []

    def assert_no_existing_children(self):
        table = process_table()
        existing = [row for row in table.values() if row['ppid'] == self.parent and is_live(row)]
        if existing:
            raise RuntimeError(f'controller already has live children: {existing}')

    def _retire(self, pid, old, mask, replacement_ticks):
        self.retired.append({'pid': pid, 'start_ticks': old['ticks'], 'first': old['first'],
                             'last': old['last'], 'signals': old['signals'], 'pidfd_exit_mask': mask,
                             'replacement_start_ticks': replacement_ticks})
        del self.known[pid]
        os.close(old['pidfd'])

    def _register(self, row):
        pid = row['pid']
        old = self.known.get(pid)
        if old is not None:
            if old['ticks'] == row['ticks']:
                return
            mask = pidfd_exit_mask(old['pidfd'])
            if mask is None:
                raise RuntimeError(f'owned PID was reused while the old identity is not proven exited: {pid}')
            # The replacement must still pass this caller's descendant/session
            # attribution and the independent pidfd/start-ticks check below.
            self._retire(pid, old, mask, row['ticks'])
        try:
            fd = pidfd_open(pid)
        except OSError as error:
            if error.errno == errno.ESRCH:
                return
            raise
        try:
            current = process_table().get(pid)
            if not same_identity(current, row):
                # In particular, never bind a recycled PID to the old process.
                os.close(fd)
                return
            self.known[pid] = {'ticks': row['ticks'], 'pidfd': fd, 'first': row, 'last': row, 'signals': []}
        except BaseException:
            os.close(fd)
            raise

    def start(self, child):
        if self.child is not None or self.known:
            raise RuntimeError('previous subprocess session has not been closed')
        self.child = child
        table = process_table()
        leader = table.get(child.pid)
        if (leader is None or leader['ppid'] != self.parent or leader['session'] != child.pid or
                leader['pgrp'] != child.pid):
            raise RuntimeError(f'new process is not the expected private session: {leader}')
        self.leader_ticks = leader['ticks']
        self._register(leader)
        if child.pid not in self.known:
            raise RuntimeError('could not pin new child identity')
        self.refresh()

    def refresh(self):
        if self.child is None:
            return {}
        table = process_table()
        # Retire disappeared, proven-exited identities promptly so a long queue
        # does not retain one pidfd per historical worker. Keep visible zombies
        # pinned until cleanup has an opportunity to reap adopted children.
        for pid, item in list(self.known.items()):
            current = table.get(pid)
            if current is not None and current['ticks'] == item['ticks']:
                continue
            mask = pidfd_exit_mask(item['pidfd'])
            if mask is not None:
                self._retire(pid, item, mask, current['ticks'] if current is not None else None)
            elif current is not None:
                raise RuntimeError(f'owned PID was reused while the old identity is not proven exited: {pid}')
        # Process IDs already tracked are accepted only while start ticks match.
        connected = {pid for pid, item in self.known.items() if pid in table and
                     table[pid]['ticks'] == item['ticks']}
        # All direct children of this dedicated subreaper during this serial
        # phase are its spawned leader or an orphan adopted from that leader.
        attempted = set()
        while True:
            added = {pid for pid, row in table.items() if row['ticks'] >= self.leader_ticks and
                     (row['ppid'] == self.parent or row['ppid'] in connected)} - connected - attempted
            if not added:
                break
            attempted.update(added)
            for pid in added:
                self._register(table[pid])
            connected.update(pid for pid in added if pid in self.known and
                             self.known[pid]['ticks'] == table[pid]['ticks'])
        # Session membership alone is accepted only while another pinned member
        # proves that this is still the original session, avoiding SID reuse.
        anchor = any(is_live(table.get(pid)) and table[pid]['session'] == self.child.pid
                     for pid in connected if pid in table)
        unknown = [row for pid, row in table.items() if row['session'] == self.child.pid and
                   row['ticks'] >= self.leader_ticks and pid not in connected and is_live(row)]
        if unknown and not anchor:
            raise RuntimeError(f'cannot establish ownership of session members: {unknown}')
        for row in unknown:
            self._register(row)
        table = process_table()
        for pid, item in self.known.items():
            current = table.get(pid)
            if current is not None and current['ticks'] == item['ticks']:
                item['last'] = current
        return {pid: item['last'] for pid, item in self.known.items()
                if pid in table and table[pid]['ticks'] == item['ticks'] and is_live(table[pid])}

    def signal_owned(self, sig):
        live = self.refresh()
        for pid, row in live.items():
            item = self.known[pid]
            current = process_table().get(pid)
            if not same_identity(current, row) or current['ticks'] != item['ticks']:
                raise RuntimeError(f'process changed before pidfd signal: {pid}')
            try:
                pidfd_signal(item['pidfd'], sig)
                item['signals'].append(int(sig))
            except OSError as error:
                if error.errno != errno.ESRCH:
                    raise

    def pinned_zombie(self, pid):
        """Accept only an observed zombie whose still-open pidfd pins its ticks."""
        item = self.known.get(pid)
        if item is None:
            return False
        row = process_table().get(pid)
        if (row is None or row['state'] != 'Z' or row['ticks'] != item['ticks'] or
                row['session'] != item['last']['session'] or row['pgrp'] != item['last']['pgrp']):
            return False
        poller = select.poll()
        poller.register(item['pidfd'], select.POLLIN)
        events = poller.poll(0)
        return (len(events) == 1 and events[0][0] == item['pidfd'] and
                bool(events[0][1] & select.POLLIN) and
                not bool(events[0][1] & (select.POLLNVAL | select.POLLERR)))

    def cleanup(self, lease_check):
        """Do not return until all proven descendants are gone and visible."""
        if self.child is None:
            self.assert_no_existing_children()
            return
        started = time.monotonic()
        last_notice = -100.0
        while True:
            try:
                self.child.poll()  # reap the direct child if it already exited
                if self.leader_ticks is None:
                    # If identity setup failed, do not guess whom to signal.
                    # The subreaper keeps any surviving orphan as our child.
                    if self.child.returncode is not None:
                        self.assert_no_existing_children()
                        break
                    raise RuntimeError('leader identity unavailable; waiting without signalling')
                live = self.refresh()
                if not live:
                    # Adopted zombies are dead and cannot own a CUDA context;
                    # reap only the identities proven to be our children.
                    table = process_table()
                    for pid, item in self.known.items():
                        row = table.get(pid)
                        if row and row['ticks'] == item['ticks'] and row['ppid'] == self.parent and row['state'] == 'Z':
                            try:
                                os.waitpid(pid, os.WNOHANG)
                            except ChildProcessError:
                                pass
                    self.assert_no_existing_children()
                    break
                self.signal_owned(signal.SIGTERM if time.monotonic() - started < 5 else signal.SIGKILL)
                problem = f'waiting for owned processes to exit: {list(live)}'
            except Exception as error:
                # Keep the controller and its four-GPU owner lock alive. No
                # unproven PID is ever signalled to make progress.
                problem = f'cleanup proof unavailable: {type(error).__name__}: {error}'
            now = time.monotonic()
            if now - last_notice >= 15:
                print(problem + '; retaining the owner lock', flush=True)
                last_notice = now
            try:
                lease_check()
            except Exception as error:
                print(f'lease status during cleanup: {type(error).__name__}: {error}', flush=True)
            time.sleep(1)
        final_table = process_table()
        self.history.append({'leader_pid': self.child.pid, 'leader_ticks': self.leader_ticks,
                             'tracked': [{'pid': pid, 'start_ticks': item['ticks'],
                                          'first': item['first'], 'last': item['last'],
                                          'signals': item['signals'],
                                          'final': final_table.get(pid) if
                                          final_table.get(pid, {}).get('ticks') == item['ticks'] else None}
                                         for pid, item in self.known.items()],
                             'retired': list(self.retired), 'no_live_owned_children': True})
        for item in self.known.values():
            os.close(item['pidfd'])
        self.known.clear()
        self.retired.clear()
        self.child = None
        self.leader_ticks = None
