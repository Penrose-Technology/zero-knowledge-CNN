import numpy as np
import matplotlib.pyplot as plt

def picture():

    plt.figure(figsize=(16, 10)) 
    x = np.array([32*32, 64*64, 128*128, 256*256])
    y1 = np.array([0.005483447, 0.007822628, 0.017533129, 0.064691039])
    y2 = np.array([0.789332039, 0.940464879, 2.567314022, 9.59606365])
    y3 = np.array([0.130637499, 0.12069016, 0.192489441, 0.507538899])
    l = np.array([3*3, 5*5, 7*7, 11*11])
    z1 = np.array([0.066983604, 0.064691039, 0.074866659, 0.073855763])
    z2 = np.array([9.573812415, 9.59606365, 10.70343352, 13.788594169])
    z3 = np.array([0.504311071, 0.507538899, 0.602098304, 0.787465155])

    plt.subplot(1, 2, 1)
    plt.plot(x, y1, label='setup', marker='o', color='r')  
    plt.plot(x, y2, label='proving', marker='s', color='g')  
    plt.plot(x, y3, label='verification', marker='d', color='b')  
    plt.title('Runtime (kernel size is 5*5)')
    plt.xlabel('data size')
    plt.ylabel('secs')
    x_labels = ['32*32', '64*64', '128*128', '256*256']
    plt.xticks(x, x_labels, rotation=45)
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(l, z1, label='setup', marker='o', color='r')  
    plt.plot(l, z2, label='proving', marker='s', color='g')  
    plt.plot(l, z3, label='verification', marker='d', color='b')  
    plt.title('Runtime (data size is 256*256)')
    plt.xlabel('kernel size')
    plt.ylabel('secs')
    l_labels = ['3*3', '5*5', '7*7', '11*11']
    plt.xticks(l, l_labels, rotation=45)
    plt.legend(loc="best")

    plt.savefig('./performance/runtime.png')
    


if __name__ == "__main__":
    picture()