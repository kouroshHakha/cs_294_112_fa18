import os
import sys
from multiprocessing import Process

if __name__ == '__main__':

    bs = [10000, 30000, 50000]
    rs = [0.005, 0.01, 0.02]

    processes = []
    for b in bs:
        for r in rs:
            command = 'python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9' \
                      ' -n 100 -e 3 -l 2 -s 32 -b %d -lr %f --exp_name hc_b%d_r%f' %(b,r,b,r)

            p = Process(target=os.system, args=(command,))
            p.start()
            processes.append(p)


    for p in processes:
        p.join()
