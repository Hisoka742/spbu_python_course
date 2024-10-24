import threading
from queue import Queue
from typing import Callable, Any

class ThreadPool:
    """
    ThreadPool class that manages a fixed number of worker threads.
    Each thread waits for tasks from a queue and executes them.
    """

    def __init__(self, num_threads: int):
        """
        Initializes the thread pool with the specified number of threads.

        Args:
            num_threads (int): Number of worker threads in the pool.
        """
        self.num_threads = num_threads  # Number of threads in the pool
        self.tasks = Queue()  # Queue to store tasks to be processed by threads
        self.threads = []  # List to keep track of worker threads
        self.shutdown_flag = threading.Event()  # Event flag to signal shutdown

        # Create and start the specified number of worker threads
        for _ in range(num_threads):
            thread = threading.Thread(target=self.worker)
            thread.start()
            self.threads.append(thread)

    def worker(self):
        """
        Worker thread function that continuously picks tasks from the queue
        and executes them unless the shutdown flag is set.
        """
        while not self.shutdown_flag.is_set():
            try:
                # Get a task from the queue and execute it
                task, done_event = self.tasks.get(timeout=1)
                result = task()  # Execute the task (a function)
                done_event.set()  # Mark the task as completed
            except Queue.Empty:
                # If no task is available, continue waiting
                continue

    def enqueue(self, task: Callable[[], Any]):
        """
        Enqueues a new task to be executed by a worker thread.

        Args:
            task (Callable): A function that performs a task and returns a result.

        Returns:
            done_event (threading.Event): Event that can be used to wait for the task completion.
                                          Returns None if the pool is shut down.
        """
        if not self.shutdown_flag.is_set():
            done_event = threading.Event()  # Event to track task completion
            self.tasks.put((task, done_event))  # Add the task to the queue
            return done_event  # Return the event to wait for task completion
        return None  # If the pool is shut down, return None

    def dispose(self):
        """
        Shuts down the thread pool by setting the shutdown flag and
        waiting for all worker threads to complete.
        """
        self.shutdown_flag.set()  # Signal all threads to stop
        for thread in self.threads:
            thread.join()  # Wait for each thread to finish
