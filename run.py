import threading
from time import sleep,ctime
import demo
import os
import sys

cnt_cuda = 4
def run_motpy(cudaid,input):
    print('CUDA_VISIBLE_DEVICES='+str(cudaid)+' python3 '+'mot/yolov3_deepsort++.py '+input)
    f=os.system(r'CUDA_VISIBLE_DEVICES='+str(cudaid)+' python3 '+'mot/yolov3_deepsort++.py '+input)
    # f=os.popen(r'CUDA_VISIBLE_DEVICES='+str(cudaid)+' python3 '+'mot/yolov3_deepsort++.py '+input)
    # print('跟踪', input, '结束于：', ctime())
    # tracks=f.readlines()
    # print(tracks)
    # print(type(tracks))
    # return tracks
    # 利用os.system运行文件,后面的r为将引号中的内容当成raw string不解析,此处不写没影响
def run_demopy(cudaid,input):
    # print('开始分割', input, 'at:', ctime())
    print('CUDA_VISIBLE_DEVICES='+str(cudaid)+' python3 '+'demo.py '+'--video-input '+input)
    os.system(r'CUDA_VISIBLE_DEVICES='+str(cudaid)+' python3 '+'demo.py '+'--video-input '+input)
    # print('分割', input, '结束于：', ctime())
    # 利用os.system运行文件,后面的r为将引号中的内容当成raw string不解析,此处不写没影响

def main1(cudaid,vdo_files):

    print('跟踪程序开始于：', ctime())
    threads = []
    cnt_files = range(len(vdo_files))

    for i in cnt_files:
        cudaid += 1
        t = threading.Thread(target=run_motpy, args=(cudaid%cnt_cuda, vdo_files[i]))  # 循环 实例化2个Thread类，传递函数及其参数，并将线程对象放入一个列表中
        threads.append(t)
    for i in cnt_files:
        print('开始跟踪', vdo_files[i], 'at:', ctime())
        threads[i].start()  # 循环 开始线程
    for i in cnt_files:
        threads[i].join()  # 循环 join()方法可以让主线程等待所有的线程都执行完毕。
        print('跟踪', vdo_files[i], '结束于：', ctime())

    print('跟踪任务完成于：', ctime())

def main2(cudaid,vdo_files):
    print('分割程序开始于：', ctime())
    threads = []
    cnt_files = range(len(vdo_files))

    for i in cnt_files:
        cudaid += 1
        t = threading.Thread(target=run_demopy, args=(cudaid%cnt_cuda, vdo_files[i]))  # 循环 实例化2个Thread类，传递函数及其参数，并将线程对象放入一个列表中
        threads.append(t)
    for i in cnt_files:
        print('开始分割', vdo_files[i], 'at:', ctime())
        threads[i].start()  # 循环 开始线程
    for i in cnt_files:
        threads[i].join()  # 循环 join()方法可以让主线程等待所有的线程都执行完毕。
        print('分割', vdo_files[i], '结束于：', ctime())

    print('分割任务完成于：', ctime())


if __name__ == '__main__':
    vdo_files = []
    cudaid=int(sys.argv[1])
    input_path=sys.argv[2]
    print(cudaid,input_path)
    for root, dirs, files in os.walk(input_path):  # 这里就填文件夹目录就可以了
        for file in files:
            if ('.mp4' in file):
                path = os.path.join(root, file)
                vdo_files.append(path)
    main1(cudaid,vdo_files)
    main2(cudaid,vdo_files)
    print('全部任务完成于：', ctime())

