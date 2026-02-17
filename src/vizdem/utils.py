#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vizdem.utils
~~~~~~~~~~~~~~~~~~~~~~~

:copyright: (c) 2016 - 2026 Regents of the University of Colorado
:license: MIT, see LICENSE for more details.
"""

import shutil
import math
import logging
from tqdm import tqdm
from fetchez.utils import int_or

logger = logging.getLogger(__name__)


def yield_srcwin(n_size=(), n_chunk=10, step=None, msg='Chunking srcwin',
                 end_msg='Chunked srcwin', start_at_edge=True, verbose=True):
    """Yield source windows in n_chunks at step"""

    if step is None:
        step = n_chunk

    n_edge = n_chunk if start_at_edge else step
    x_chunk = n_edge

    total_steps = (math.ceil((n_size[0] + (n_chunk - n_edge)) / step) * math.ceil((n_size[1] + (n_chunk - n_edge)) / step))

    with tqdm(total=total_steps,
             desc=f'{msg} @ chunk:{int_or(n_chunk)}/step:{int_or(step)}...',
             leave=verbose) as pbar:
        while True:
            y_chunk = n_edge
            while True:
                this_x_chunk = min(x_chunk, n_size[1])
                this_y_chunk = min(y_chunk, n_size[0])

                this_x_origin = max(0, int(x_chunk - n_chunk))
                this_y_origin = max(0, int(y_chunk - n_chunk))

                this_x_size = int(this_x_chunk - this_x_origin)
                this_y_size = int(this_y_chunk - this_y_origin)

                if this_x_size <= 0 or this_y_size <= 0:
                    break

                srcwin = (this_x_origin, this_y_origin, this_x_size, this_y_size)
                pbar.update()
                yield srcwin

                if y_chunk > (n_size[0] * step):
                    break
                y_chunk += step

            if x_chunk > (n_size[1] * step):
                break
            x_chunk += step


def buffer_srcwin(srcwin=(), n_size=None, buff_size=0, verbose=True):
    """Buffer the srcwin by `buff_size`"""

    if n_size is None:
        n_size = srcwin

    x_origin = max(0, srcwin[0] - buff_size)
    y_origin = max(0, srcwin[1] - buff_size)

    x_buff_size = buff_size * 2 if x_origin != 0 else buff_size
    y_buff_size = buff_size * 2 if y_origin != 0 else buff_size

    x_size = srcwin[3] + x_buff_size
    if (x_origin + x_size) > n_size[1]:
        x_size = n_size[1] - x_origin

    y_size = srcwin[2] + y_buff_size
    if (y_origin + y_size) > n_size[0]:
        y_size = n_size[0] - y_origin

    return int(x_origin), int(y_origin), int(x_size), int(y_size)


# System Command Functions
cmd_exists = lambda x: any(os.access(os.path.join(path, x), os.X_OK)
                           for path in os.environ['PATH'].split(os.pathsep))


def run_cmd(cmd, data_fun=None, cwd='.'):
    """Run a system command while optionally passing data.

    `data_fun` should be a function to write to a file-port:
    >> data_fun = lambda p: datalist_dump(wg, dst_port = p, ...)
    """

    out = None
    cols, _ = shutil.get_terminal_size()
    width = cols - 55

    silent = logger.getEffectiveLevel() > logging.INFO

    with tqdm(desc=f'`{cmd.rstrip()[:width]}...`', leave=False, disable=silent) as pbar:
        pipe_stdin = subprocess.PIPE if data_fun is not None else None

        p = subprocess.Popen(
            cmd, shell=True, stdin=pipe_stdin, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, close_fds=True, cwd=cwd
        )

        if data_fun is not None:
            logger.info('Piping data to cmd subprocess...')
            data_fun(p.stdin)
            p.stdin.close()

        io_reader = io.TextIOWrapper(p.stderr, encoding='utf-8')
        while p.poll() is None:
            err_line = io_reader.readline()
            if not silent and err_line:
                pbar.write(err_line.rstrip())
                sys.stderr.flush()
            pbar.update()

        out = p.stdout.read()
        p.stderr.close()
        p.stdout.close()

        logger.info(f'Ran cmd {cmd.rstrip()} and returned {p.returncode}')

    return out, p.returncode


def yield_cmd(cmd, data_fun=None, cwd='.'):
    """Yield output from a system command.

    `data_fun` should be a function to write to a file-port:
    >> data_fun = lambda p: datalist_dump(wg, dst_port = p, ...)
    """

    logger.info(f'Running cmd {cmd.rstrip()}...')

    pipe_stdin = subprocess.PIPE if data_fun is not None else None

    p = subprocess.Popen(
        cmd, shell=True, stdin=pipe_stdin, stdout=subprocess.PIPE,
        close_fds=True, cwd=cwd
    )

    if data_fun is not None:
        logger.info("Piping data to cmd subprocess...")
        data_fun(p.stdin)
        p.stdin.close()

    while p.poll() is None:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        yield line

    p.stdout.close()
    logger.info(f"Ran cmd {cmd.rstrip()}, returned {p.returncode}.")


def cmd_check(cmd_str, cmd_vers_str):
    """check system for availability of 'cmd_str'"""

    if cmd_exists(cmd_str):
        cmd_vers, status = run_cmd(f"{cmd_vers_str}")
        return cmd_vers.rstrip()
    return b"0"
