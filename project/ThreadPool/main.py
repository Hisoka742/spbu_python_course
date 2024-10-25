import threading
from typing import Callable, Any, Tuple


class ThreadPool:
    """
    ThreadPool class that manages a fixed number of worker threads.
    Each thread waits for tasks and executes them.
    """

    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.tasks: list[Tuple[Callable[[], Any], threading.Event]] = []    # List to hold tasks with a lock for thread-safe operations
        self.lock = threading.Lock()  # Lock for synchronizing access to tasks
        self.task_available = (
            threading.Event()
        )  # Event to signal the availability of tasks
        self.shutdown_flag = threading.Event()  # Event flag to signal shutdown
        self.threads = []  # List to keep track of worker threads

        for _ in range(num_threads):
            thread = threading.Thread(target=self.worker)
            thread.start()
            self.threads.append(thread)

    def worker(self):
        while not self.shutdown_flag.is_set():
            self.task_available.wait()  # Wait until a task is available or shutdown is signaled
            with self.lock:
                if self.tasks:
                    task, done_event = self.tasks.pop(0)  # Get a task
                    result = task()  # Execute the task
                    done_event.set()  # Mark task as done
                else:
                    self.task_available.clear()  # Reset the event if no tasks are available

    def enqueue(self, task: Callable[[], Any]):
        if not self.shutdown_flag.is_set():
            done_event = threading.Event()
            with self.lock:
                self.tasks.append((task, done_event))
                self.task_available.set()  # Signal that a new task is available
            return done_event
        return None

    def dispose(self):
        self.shutdown_flag.set()  # Signal all threads to stop
        self.task_available.set()  # Wake up any waiting threads
        for thread in self.threads:
            thread.join()  # Wait for all threads to complete
