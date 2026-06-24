"""Standalone polling process — run separately from the web server.

Auto-started by app.py. Runs an APScheduler cron job (schedule from the
`poll_cron` setting, reloaded on SIGHUP). Each fire runs the poll in a fresh
subprocess so it always picks up the latest code.
"""
import logging
import os
import signal
import subprocess
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import db

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_DEFAULT_CRON = '0 4 * * *'
_cwd = os.path.dirname(os.path.abspath(__file__))
PIDFILE = os.path.join(_cwd, 'poll.pid')

_POLL_CMD = (
    'import logging; '
    'logging.basicConfig(level=logging.INFO, '
    '  format="%(asctime)s %(levelname)-8s %(name)s — %(message)s"); '
    'import db; db.init_db(); '
    'from poller import poll_all_due; poll_all_due()'
)


def _run_poll():
    try:
        subprocess.run([sys.executable, '-c', _POLL_CMD], cwd=_cwd, check=False)
    except Exception as e:
        logger.error('Poll subprocess error: %s', e)


def _load_cron() -> str:
    return db.get_setting('poll_cron', _DEFAULT_CRON).strip() or _DEFAULT_CRON


def _shutdown(signum, frame):
    logger.info('Shutting down poller')
    try:
        os.unlink(PIDFILE)
    except OSError:
        pass
    sys.exit(0)


if __name__ == '__main__':
    # Singleton: exit if already running
    if os.path.exists(PIDFILE):
        try:
            with open(PIDFILE) as f:
                existing_pid = int(f.read().strip())
            os.kill(existing_pid, 0)
            logger.info('Poller already running as PID %d — exiting', existing_pid)
            sys.exit(0)
        except (OSError, ValueError):
            pass  # stale PID file

    with open(PIDFILE, 'w') as f:
        f.write(str(os.getpid()))

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    db.init_db()
    cron = _load_cron()

    scheduler = BlockingScheduler()
    scheduler.add_job(_run_poll, CronTrigger.from_crontab(cron), id='poll')
    logger.info('Poller started — schedule: %s', cron)

    def _reload_schedule(signum, frame):
        new_cron = _load_cron()
        try:
            scheduler.reschedule_job('poll', trigger=CronTrigger.from_crontab(new_cron))
            logger.info('Poll schedule reloaded: %s', new_cron)
        except Exception as e:
            logger.error('Failed to apply poll schedule %r: %s', new_cron, e)

    signal.signal(signal.SIGHUP, _reload_schedule)

    try:
        scheduler.start()
    finally:
        try:
            os.unlink(PIDFILE)
        except OSError:
            pass
