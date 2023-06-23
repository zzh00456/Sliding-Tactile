# import xlrd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import numpy
import csv
import nidaqmx
from nidaqmx.constants import TerminalConfiguration


def read_csv(name, size, half_size, judge):
    data = np.genfromtxt(name, delimiter=",")
    # data = np.delete(data, 0, axis = 0)
    # data = np.delete(data, 0, axis = 1)
#     data = np.flipud(data)
    data_min = np.min(data, axis = 0)# axis=0;  
    data_max = np.max(data, axis = 0)# axis=0; 
    print(data_min)
    print(data_max)
    threshold = data_min + (data_max - data_min)*0.1
#     print(threshold[0])
    x, y = np.shape(data)
    print("x: %s" %x)
    print(y)
#     size = 4000
#     half_size = 2000
    i = 0
    judge = 0
    data_cut = np.zeros((1,size,4))
    while i+ size < x:
        if ((data[i][judge] < threshold[judge]) and (data[i+size][judge] < threshold[judge]) and (data[i+half_size][judge]> threshold[judge])):
            i = i + 250
            data_cut = np.append(data_cut, [data[i:i+size]],axis = 0)
            i = i + size
        else:
            i = i+1
    return data_cut


def zero(data_np):
    for i in range(4):
        data_np[:,i] = data_np[:,i] - min(data_np[:,i])
    return data_np


def sensitive(data_np):
    m = np.max(data_np, 0) #axis=0 
    mm = max(m)
    data_pp = data_np
    for k in range(4):
        data_pp[:,k] = data_np[:,k] * (mm/m[k])
        #print("The sensitivity parameter is:", mm/m[k])
    return data_pp

def plot_curve(data_np):
    l1 = plt.plot(data_np[:,0],label='sensor0') ##取第1列 X[:, m:n]，即取所有数据的第m到n-1列数据，含左不含右
#     print(data_np[:,0])
    l2 = plt.plot(data_np[:,1],label='sensor1')
    l3 = plt.plot(data_np[:,2],label='sensor2')
    l4 = plt.plot(data_np[:,3],label='sensor3')

    l5 = plt.plot(data_np[:,4],label='sensor4')
    l6 = plt.plot(data_np[:,5],label='sensor5')
    l7 = plt.plot(data_np[:,6],label='sensor6')
    l8 = plt.plot(data_np[:,7],label='sensor7')
    plt.legend()
    plt.show()


def plot_gray(data_pp):
    plt.axis('off')
#     plt.imshow(data_pp, cmap=plt.cm.gray, aspect=0.0042*1.3)
#     plt.imshow(data_pp[:,0:1], cmap=plt.cm.gist_gray, aspect=0.001,vmax=1.5)
    plt.imshow(data_pp, cmap=plt.cm.gist_gray, aspect=0.0042*0.3)
    
    plt.show()
    plt.pause(5)
    plt.close()

 

def plot_compress(data_np, rate):
    x,y = np.shape(data_np)
    data_pp = np.zeros((1,4))
    data_pp = [np.mean(data_np[0:rate,:], axis=0)] #Returns 1*n matrix with mean values for each column
    for i in range(rate, (x-rate), rate):
        data_pp = np.append(data_pp, [np.mean(data_np[i:i+rate,:], axis=0)], axis=0)
    return data_pp

def plot_save(data_pp,name,index):
    plt.figure(num=None, figsize=(10,5), facecolor=None, edgecolor=None, clear=False, )
    plt.suptitle('Recognition Result', fontsize=22, x=0.43, fontstyle='oblique', fontweight='bold')
    plt.subplot(1, 2, 1)
    l1 = plt.plot(data_pp[:,0],label='sensor0') 
    l2 = plt.plot(data_pp[:,1],label='sensor1')
    l3 = plt.plot(data_pp[:,2],label='sensor2')
    l4 = plt.plot(data_pp[:,3],label='sensor3')
#     l5 = plt.plot(data_np[:,4],label='sensor4')
#     l6 = plt.plot(data_np[:,5],label='sensor5')
#     l7 = plt.plot(data_np[:,6],label='sensor6')
#     l8 = plt.plot(data_np[:,7],label='sensor7')
    plt.legend()
    # plt.show()

    plt.subplot(1, 2, 2)
    plt.axis('off')
    # plt.imshow(data_pp, cmap=plt.cm.gist_gray, aspect=0.001,vmin=0,vmax=1.5)#
    plt.imshow(data_pp, cmap='rainbow', aspect=0.0042*0.5)#
    plt.savefig('%s_%s.png' %(name,index),bbox_inches='tight',pad_inches=0.0)
    plt.show()
    # plt.pause(5)
    # plt.close()

def plot_save_csv(data_pp,name):
    with open('Processing-%s.csv' %name,'w',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerows(data_pp)

def main_test(name, size, half_size, judge):
    test1 = read_csv(name, size, half_size, judge)
    index, x, y = np.shape(test1)
    k = 0
    row=0
    print(index)
    print(x)
    print(y)
    # name1 = name.split('.csv')[0].split('CSV数据/')[1]
    # print(name1)
    data_intercept = np.zeros((x-1,(index-1)*2))
    for i in range(1,index):
        plt.figure(i)
        data1 = plot_compress(test1[i], 1) 
        data_np = zero(data1)
        #plot_curve(data_np)
        data_pp = sensitive(data_np)
        # plot_curve(data_pp)
        
#         plot_curve(data_intercept)
        plot_save(data_pp,name,k)
        plot_save_csv(data_pp,name)
        # plot_gray(data_pp)
        # plot_gray(data_pp)
        k=k+1



count = 0
i = 0
while count<30:
    i =0
    price_str = input("1")
    # if price_str == 1:
    #     count += 1
    count += 1    
    with nidaqmx.Task() as task:
        with open("%s.csv"%count,'w') as file:
            task.ai_channels.add_ai_voltage_chan(
            "Dev1/ai0:3",
            terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            task.timing.cfg_samp_clk_timing(
            rate=1000, sample_mode=nidaqmx.constants.AcquisitionType.FINITE)
        # freq=1k,continuously get data
            while i<7000:
                aiData = task.read()
                a = str(aiData)
                a = a[1:]
                a = a[:-1]
                file.write(a+"\n")
                i = i+1
                print(a)
            print(count)
    main_test("%s.csv"%count , 5000, 2050, 0)
    # main_test("./0.csv" , 5000, 1000, 2)
