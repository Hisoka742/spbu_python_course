import pytest
import time
from project.ThreadPool.main import ThreadPool  # Assuming the ThreadPool class is saved in thread_pool.py

def test_basic_threadpool_execution():
    """
    Test basic functionality of the thread pool by enqueuing several tasks
    and ensuring that they are executed and completed.
    """
    def simple_task():
        # Task simulates a small delay to mimic work
        time.sleep(0.1)
        return "Task completed"

    pool = ThreadPool(3)  # Create a pool with 3 threads

    # Enqueue 5 tasks and store the event objects to wait on
    done_events = [pool.enqueue(simple_task) for _ in range(5)]

    # Wait for all tasks to complete by waiting for the events
    for event in done_events:
        event.wait()

    pool.dispose()  # Dispose the pool to stop threads
    assert all(event.is_set() for event in done_events)  # Ensure all tasks are completed

def test_threadpool_size():
    """
    Test that the thread pool correctly creates the specified number of threads.
    """
    pool = ThreadPool(4)  # Create a pool with 4 threads
    assert len(pool.threads) == 4  # Ensure 4 threads were created
    pool.dispose()  # Dispose the pool

def test_task_execution_and_reuse():
    """
    Test that tasks are executed correctly and that threads are reused
    for multiple tasks.
    """
    results = []

    def add_task(x):
        # Task that squares the input number and appends it to results
        time.sleep(0.1)
        results.append(x * x)
        return x * x

    pool = ThreadPool(2)  # Create a pool with 2 threads

    # Enqueue 6 tasks, each calculating the square of a number
    done_events = [pool.enqueue(lambda x=i: add_task(x)) for i in range(6)]

    # Wait for all tasks to complete
    for event in done_events:
        event.wait()

    pool.dispose()  # Dispose the pool
    assert sorted(results) == [i * i for i in range(6)]  # Check if all results are correct

def test_dispose_stops_new_tasks():
    """
    Test that after calling dispose, no new tasks can be added to the pool.
    """
    def task():
        return "Should not be executed"

    pool = ThreadPool(2)  # Create a pool with 2 threads
    pool.dispose()  # Dispose the pool

    # Try to enqueue a task after dispose, should return None
    assert pool.enqueue(task) is None
