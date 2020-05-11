#
#
# import numpy as np
#
# a = np.array([[1, 2, 3],[4,5,6]])
# b = []
# b.append(a[0,1])
# print(b)
# print(a)


# import multiprocessing
# import time
# import random

# def func(msg):
#   print(msg)
#   sleep_time = random.randint(1, 10)
#   time.sleep(sleep_time)
#   return "done " + msg
#
# if __name__ == "__main__":
#   pool = multiprocessing.Pool(processes=4)
#   result = []
#   for i in range(20):
#     msg = "hello %d" %(i)
#     pool.apply_async(func, (msg, ))
#   pool.close()
#   pool.join()
#   # for res in result:
#   #   print(res.get())
#
import multiprocessing
import random
import time


def func(msg):
  print(multiprocessing.current_process().name + '-' + msg)
  sleep_time = random.randint(1, 20)
  time.sleep(sleep_time)
  return msg


if __name__ == "__main__":
  pool = multiprocessing.Pool(processes=4)  # 创建4个进程
  result = []
  for i in range(10):
    msg = "hello %d" % (i)
    result.append(pool.apply_async(func, (msg,)))
  pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
  pool.join()  # 等待进程池中的所有进程执行完毕，必须在close()之后调用
  for res in result:
    print(res.get())
# step = 3200//1000
# print(step)
# import sys
# print(sys.path)