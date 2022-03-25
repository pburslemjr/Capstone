from time import sleep
import matplotlib.pyplot as plt
import math


def readFile(fileName):
    fileObj = open(fileName, "r") #opens the file in read mode
    lines = fileObj.read().splitlines() #puts the file into an array
    fileObj.close()

    #Convert string element to floats
    data = []
    for i in range(len(lines)):
        line = lines[i].split()
        for j in range(len(line)):
            data.append(float(line[j]))

    return data

def getColumnAtIndex(index, arr):
    res = []
    numOfDataPointsPerLine = 3
    for i in range(len(arr)):
        if i % numOfDataPointsPerLine == index:
            res.append(arr[i])

    return res


def plot(data_y, y_type):
    data_x = []
    for i in range(len(data_y)):
        data_x.append(i)

    #create average of data to plot
    avg_data_y = []
    avg_data_x = []
    for i in range(math.floor((len(data_y) - 5) / 10)):
        last_few_avg = data_y[i*10 + 0] + data_y[i*10 + 1] + data_y[i*10 + 2]
        last_few_avg += data_y[i*10 + 3] + data_y[i*10 + 4] + data_y[i*10 + 5]
        last_few_avg += data_y[i*10 + 6] + data_y[i*10 + 7] + data_y[i*10 + 8]
        last_few_avg += data_y[i*10 + 9]

        last_few_avg = last_few_avg / 10

        avg_data_y.append(last_few_avg)
        avg_data_x.append(i*10 + 5)

    x = data_x
    y = data_y
    plt.plot(x,y)
    plt.plot(avg_data_x,avg_data_y)
    plt.xlabel('Episode')

    if (y_type == "true"):
        plt.ylabel('Shaped + Unshaped Reward')
    elif (y_type == "shaped"):
        plt.ylabel('Shaped Reward')
    elif (y_type == "unshaped"):
        plt.ylabel('Unshaped Reward')
    else:
        plt.ylabel('Unknown Y-value Type')

    plt.title("Reward vs. Episode")
    plt.show()


all_data = readFile('rewards.txt')

true_rews = getColumnAtIndex(0, all_data)
shaped_rews = getColumnAtIndex(1, all_data)
unshaped_rews = getColumnAtIndex(2, all_data)

plot(true_rews, "true")
plot(shaped_rews, "shaped")
plot(unshaped_rews, "unshaped")