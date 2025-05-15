#!/usr/bin/env python
# encoding: utf-8

"""
    GPU Task Scheduler
    (1) Runs tasks on multiple GPUs in parallel.
    (2) Ensures FIFO order of task dispatching.
    (3) Uses threading and locking for safety.
"""

import time, random
import torch.multiprocessing as mp

from typing import Callable, Dict, Any, List, Union, Tuple


class Task:
    def __init__(self, task_fn: Callable[..., Any], kwargs: Dict, name: str = None):
        self.task_fn = task_fn
        self.kwargs = kwargs.copy()  # the 'device' key will be added or updated later
        self.name = name

    def __str__(self):
        return f"Task(name={self.name}, device={self.kwargs.get('device', '?')})"


def worker(gpu_id: int, task_queue: mp.Queue):
    """
        The worker function that runs on each GPU.
        :param gpu_id: The ID of the GPU to use.
        :param task_queue: The queue of tasks to run.
    """
    device = f"cuda:{gpu_id}"
    while True:
        try:
            task_fn, kwargs, name = task_queue.get(timeout=2)
        except Exception as e:
            break

        print(f"[START] {name} assigned to {device}")
        try:
            kwargs['device'] = device
            kwargs['log_file'] = kwargs['log_file'].format(device=device.replace(':', '_'))
            task_fn(**kwargs)
            print(f"[DONE]  {name} finished on {device}")
        except Exception as e:
            print(f"[ERROR] {name} failed on {device}: {e}")


class GPUScheduler:
    """
        Process level GPU Scheduler.
    """

    def __init__(self, gpu_ids: Union[List[int], Tuple[int]]):

        self.gpu_ids = gpu_ids

        self.ctx = mp.get_context("spawn")  # Consistent context
        self.task_queue = self.ctx.Queue()
        self.processes: List[mp.Process] = []

    def run(self, tasks: List[Task]):
        for task in tasks:
            self.task_queue.put((task.task_fn, task.kwargs, task.name))

        for gpu_id in self.gpu_ids:
            p = self.ctx.Process(target=worker, args=(gpu_id, self.task_queue))
            p.start()
            self.processes.append(p)

        for p in self.processes:
            p.join()


def dummy_gpu_task(name: str, device: str = 'cuda:0', duration: int = 2):
    print(f"[{device}] --> Running {name} for {duration}s")
    time.sleep(duration)
    print(f"[{device}] <-- Finished {name}")


def test():
    scheduler = GPUScheduler(gpu_ids=[0, 1])

    tasks = []
    for i in range(10):
        t = Task(
            task_fn=dummy_gpu_task,
            kwargs={'name': f"Task-{i}", 'duration': random.randint(2, 5)},
            name=f"Task-{i}"
        )
        tasks.append(t)

    scheduler.run(tasks)
