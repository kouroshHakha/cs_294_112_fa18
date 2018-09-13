from multiprocessing import Process
import os
import random

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print(name)
    for i in range(100000000):
        k = random.random()

if __name__ == '__main__':
    info('main line')
    processes = []
    p = Process(target=f, args=('bob',))
    p.start()
    processes.append(p)
    print(213)
    k = Process(target=f, args=('kourosh',))
    # k.run()
    processes.append(k)

    l = Process(target=f, args=('l',))
    # l.start()
    processes.append(l)
    p.join()
    # processes[0].join()
    # for p in processes:
    #     print(p)
    #     p.join()
    f("end")