# coding=utf-8
from multiprocessing import Process


def assign_task(tasks, num_work):
    num_work = min(len(tasks), num_work)

    tasks_ = []
    for i in range(num_work):
        sub_tasks = []
        j = i
        while j < len(tasks):
            sub_tasks.append(tasks[j])
            j += num_work
        tasks_.append(sub_tasks)
    return tasks_


def start_task(target_func, args, num_work, tasks):
    assert isinstance(args, list)
    processes = []
    for i in range(num_work):
        args_ = args.copy()
        args_.append(tasks[i])
        process = Process(target=target_func, args=tuple(args_))
        processes.append(process)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
