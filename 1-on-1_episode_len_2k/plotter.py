import matplotlib.pyplot as plt
import math

def readFile(fileName):
    fileObj = open(fileName, "r") #opens the file in read mode
    data = fileObj.read().splitlines() #puts the file into an array
    fileObj.close()

    #Convert string element to floats
    for i in range(len(data)):
        data[i] = float(data[i])

    return data

data_y = readFile('rewards.txt')
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
plt.xlabel('Episode (2000 steps/episode)')
plt.ylabel('Reward')
plt.title("Reward vs. Episode")
plt.show()