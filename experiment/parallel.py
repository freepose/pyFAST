#!/usr/bin/env python
# encoding: utf-8

"""

    Run experiments on several GPUs parallel.

"""

import os
import sys
import logging
import subprocess
from datetime import datetime
from multiprocessing import Process, Queue, Value
from multiprocessing import Lock
from typing import List, Dict, Callable, Optional
from tqdm import tqdm  # For progress bar

import torch


class GPUInfo:
    @staticmethod
    def get(gpu_id: int) -> Dict[str, str]:
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=temperature.gpu,power.draw,power.limit,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits", "-i", str(gpu_id)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            temperature, power_draw, power_limit, total_mem, used_mem, utilization = result.stdout.strip().split(", ")
            return {
                "temperature": temperature,
                "power": f"{power_draw} / {power_limit}",
                "memory": f"{used_mem} / {total_mem}",
                "utilization": utilization
            }
        except Exception:
            return {
                "temperature": "N/A",
                "power": "N/A",
                "memory": "N/A",
                "utilization": "N/A"
            }

    @staticmethod
    def print(gpu_ids: Optional[List[int]] = None) -> None:
        if gpu_ids is None:
            num_gpus = torch.cuda.device_count()
            gpu_ids = list(range(num_gpus))
            print(f"当前设备有 {num_gpus} 个 GPU")
        print(
            f"{'GPU':^5} {'Name':^20} {'Temp (°C)':^10} {'Power (W)':^15} {'Memory Usage (MB)':^20} {'Utilization (%)':^15}")
        for gpu_id in gpu_ids:
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                gpu_info = GPUInfo.get(gpu_id)
                print(
                    f"{gpu_id:^5} {props.name:^20} {gpu_info['temperature']:^10} {gpu_info['power']:^15} {gpu_info['memory']:^20} {gpu_info['utilization']:^15}")
            except RuntimeError:
                print(f"GPU {gpu_id} 不存在或不可用")


class ParallelExecutor:
    def __init__(self, gpu_ids, username_postfix='test'):
        self.gpu_ids = gpu_ids
        if not self.gpu_ids:
            print("No GPUs found.")
            sys.exit(1)
        print(f"Available GPUs: {self.gpu_ids}")

        self.parallel_queue = None
        self.username_postfix = username_postfix
        base_path = os.path.expanduser('~/') if os.name == 'posix' else 'D:/'
        self.result_path = os.path.join(base_path, f"benchmark-{self.username_postfix}")
        os.makedirs(self.result_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.overall_log_file = os.path.join(self.result_path, f"log_{timestamp}.txt")

    def populate_queue(self, settings_list):
        self.parallel_queue = Queue()  # Reset the parallel queue
        for setting in settings_list:
            self.parallel_queue.put(setting)

    def execute(self, processor):
        """
        Start worker processes and manage execution.
        :param processor: Function to process individual settings.
        """
        if not self.gpu_ids:
            print("No GPUs available. Exiting.")
            return

        # Shared counter for progress tracking
        total_tasks = self.parallel_queue.qsize()
        completed_tasks = Value('i', 0)  # Shared integer
        lock = Lock()  # To synchronize updates to the counter

        def _worker_logic(gpu_id, parallel_queue, completed_tasks, lock):
            """
            Logic for processing settings on a specific GPU.
            :param gpu_id: ID of the GPU assigned to this worker.
            """
            while not parallel_queue.empty():
                try:
                    setting = parallel_queue.get_nowait()
                    device = f'cuda:{gpu_id}'
                    setting['device'] = device
                    dataset_name = setting['dataset_name']
                    task = setting['task']
                    input_window_size = setting['input_window_size']
                    output_window_size = setting['output_window_size']
                    horizon = setting['horizon']
                    stride = setting['stride']
                    model_name = setting['model_name']
                    lr = setting['lr']

                    result_path = os.path.join(self.result_path, dataset_name, task,
                                               f"L{input_window_size}_H{output_window_size}_h{horizon}_s{stride}")
                    os.makedirs(result_path, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_file_path = os.path.join(result_path, f"{model_name}_{lr}_{timestamp}.txt")

                    # Set up logger
                    logger = logging.getLogger(f"worker_GPU{gpu_id}_{model_name}")
                    logger.setLevel(logging.INFO)
                    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
                    file_handler.setLevel(logging.INFO)
                    logger.addHandler(file_handler)

                    # Redirect stdout and stderr to log file
                    original_stdout, original_stderr = sys.stdout, sys.stderr
                    sys.stdout = open(log_file_path, "a", encoding="utf-8")
                    sys.stderr = open(log_file_path, "a", encoding="utf-8")

                    try:
                        # Call the processor function
                        gpu_info = GPUInfo.get(gpu_id)
                        gpu_status = (
                            f"GPU {gpu_id} | Temp: {gpu_info['temperature']}°C | Power: {gpu_info['power']} | "
                            f"Memory: {gpu_info['memory']} | Utilization: {gpu_info['utilization']}"
                        )
                        # print(
                        #     f"[{device}] Running {model_name} (L{input_window_size}_H{output_window_size}_h{horizon}_s{stride}), "
                        #     f"Log: {log_file_path}\n{gpu_status}"
                        # )
                        with open(self.overall_log_file, "a", encoding="utf-8") as overall_log:
                            message = (
                                f"[{device}] Running {model_name} (L{input_window_size}_H{output_window_size}_h{horizon}_s{stride}), "
                                f"Log: {log_file_path}\n{gpu_status}"
                            )
                            print(message)
                            overall_log.write(message + "\n")
                        processor(**setting)
                    except Exception as e:
                        logger.error("An error occurred during execution", exc_info=True)
                    finally:
                        # Restore stdout and stderr
                        sys.stdout.close()
                        sys.stderr.close()
                        sys.stdout, sys.stderr = original_stdout, original_stderr

                    # Remove the file handler to prevent duplicate logs
                    logger.removeHandler(file_handler)
                    file_handler.close()

                    # Update progress
                    with lock:
                        completed_tasks.value += 1

                except Exception as e:
                    logging.error("An error occurred in worker logic", exc_info=True)

        # Start worker processes
        workers = []
        for gpu_id in self.gpu_ids:
            process = Process(target=_worker_logic, args=(gpu_id, self.parallel_queue, completed_tasks, lock))
            process.start()
            workers.append(process)

        # Display progress in the main process
        with tqdm(total=total_tasks, desc="Progress") as pbar:
            while any(worker.is_alive() for worker in workers):
                with lock:
                    pbar.n = completed_tasks.value
                    pbar.refresh()

        for process in workers:
            process.join()

        print("All executions have been completed.")


if __name__ == '__main__':
    from example.paper.benchmark import ts
    from example.paper.benchmark import get_settings

    max_epochs = 20
    settings_list = [
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=48, output_window_size=24,
                     model_name='ar', lr=0.0001, max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=48, output_window_size=24,
                     model_name='ar', lr=0.0002, max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=96, output_window_size=48,
                     model_name='ar', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=192, output_window_size=96,
                     model_name='ar', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=672, output_window_size=288,
                     model_name='ar', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="multivariate", input_window_size=48, output_window_size=24,
                     model_name='ar', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="multivariate", input_window_size=96, output_window_size=48,
                     model_name='ar', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="multivariate", input_window_size=192, output_window_size=96,
                     model_name='ar', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="multivariate", input_window_size=672, output_window_size=288,
                     model_name='ar', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=48, output_window_size=24,
                     model_name='dlinear', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=96, output_window_size=48,
                     model_name='dlinear', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=192, output_window_size=96,
                     model_name='dlinear', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="univariate", input_window_size=672, output_window_size=288,
                     model_name='dlinear', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="multivariate", input_window_size=48, output_window_size=24,
                     model_name='dlinear', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="multivariate", input_window_size=96, output_window_size=48,
                     model_name='dlinear', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="multivariate", input_window_size=192, output_window_size=96,
                     model_name='dlinear', max_epochs=max_epochs),
        get_settings(dataset_name="ETTh2", task="multivariate", input_window_size=672, output_window_size=288,
                     model_name='dlinear', max_epochs=max_epochs),
    ]

    gpu_ids = [0, 1]

    executor = ParallelExecutor(gpu_ids, 'hy')

    executor.populate_queue(settings_list)

    executor.execute(ts)
