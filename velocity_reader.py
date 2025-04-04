
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from matplotlib.animation import FuncAnimation


# def reject_outliers(data, m=2):
#     return data[abs(data - np.median(data)) < m * np.std(data)]
def show_data(data, window_size):
    timetamps = []
    encoder_velocities = []
    previous_time = 0
    previous_angle = 0  
    threshold = 180
    for item in data:
        current_time = item[0]
        current_angle = (item[1]/262144) * 360
        time_diff = current_time - previous_time
        angle_diff = current_angle - previous_angle
        if angle_diff > threshold:
            angle_diff -= 360
        elif angle_diff <= -threshold:
            angle_diff += 360
        current_velocity = (angle_diff / time_diff) / 360 * 60
        timetamps.append(current_time)
        encoder_velocities.append(current_velocity)
        previous_time = current_time
        previous_angle = current_angle
    smoothed_data = []
    for i in range(len(encoder_velocities)):
        if i < window_size:
            smoothed_data.append(sum(encoder_velocities[:i+1]) / (i+1))
        else:
            window = encoder_velocities[i-window_size:i]
            smoothed_data.append(sum(window) / window_size)
    #fig, ax = plt.subplots(figsize=(10, 6))
    plt.figure(figsize=(10,6))
    plt.plot(timetamps, smoothed_data, label = "Velocity Measure")
    plt.xlabel("time s")
    plt.ylabel("Velocity Measure rpm")
    plt.title("Velocity vs. time")
    plt.legend()
    plt.grid(True)
    #fig.canvas.mpl_connect('close_event', on_close)
    plt.show()
    return smoothed_data


# if len(sys.argv) < 2:
#     print("Please provide the filename as a command-line argument")
#     sys.exit(1)

filename0 = r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\Instrument_Calibration\2024-03-20_mixing_protcol_timing\A02_external_data_20240320_160451.csv"
 #r"C:\Users\imrul\Downloads\A10_external_data_20240221_134102.csv"
filename1 = r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\Instrument_Calibration\2024-03-20_mixing_protcol_timing\A08_external_data_20240320_161308.csv"
#r"C:\Users\imrul\Downloads\A02_external_data_20240221_140535.csv"
filename2= r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\Instrument_Calibration\2024-03-20_mixing_protcol_timing\A10_external_data_20240320_154657.csv"
#r"C:\Users\imrul\Downloads\A08_external_data_20240221_142339.csv"

df0 = pd.read_csv(filename0)
df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)


encoderData0 = df0.values
encoderData1 = df1.values
encoderData2 = df2.values


outputData0= show_data(encoderData0, 200)
outputData1= show_data(encoderData1, 200)
outputData2= show_data(encoderData2, 200)


#%%

dTime = np.mean(np.diff(df0['Time']))
x0 = np.arange(len(outputData0))- 7.942e4  #21687
x1 = np.arange(len(outputData1))-1.6453e5   #83270
x2 = np.arange(len(outputData2))- 119817   #158146

# x0 = np.arange(len(outputData0))-21687
# x1 = np.arange(len(outputData1))-83270
# x2 = np.arange(len(outputData2))-158146


t0=x0*dTime
t1=x1*dTime
t2=x2*dTime


plt.figure()
plt.plot(t0, outputData0, label = 'A02')
plt.plot(t1, outputData1, label = 'A08')
plt.plot(t2, outputData2, label = 'A10')
plt.legend()
plt.grid(True)
plt.ylabel('Vel (rpm)')
plt.xlabel('Time (s)')

# acc0=np.diff(outputData0)/np.diff(t0)
# plt.figure()
# plt.plot(t0[:-1],acc0)
#%%

yy = np.array(outputData2)- 300

idx = np.where(abs(yy)< 5)[0] # find zero crossing points


idxFilter = np.diff(idx) # to filter out too close by points
idxClean = []


for i in range(len(idxFilter)):
    if abs(idxFilter[i]) > 100:
        idxClean.append(idx[i])
        
    

# plt.figure()
# plt.plot(yy)
# plt.plot(idxClean,yy[idxClean],'o')


idxCleanFilter = np.diff(np.array(idxClean))

startIndex = np.where(idxCleanFilter>50000)[0]

try:
    timePeriods = idxCleanFilter[startIndex[0]+2::2]
except:
    timePeriods = idxCleanFilter[0::2]

if len(timePeriods) >= 20:
    timePeriods = timePeriods[:20]
else:
    print('Not all 20 cycle periods are found')


meanPeriod = np.mean(timePeriods)*dTime
stdPeriod = np.std(timePeriods)*dTime / meanPeriod*100


print(f'mean time for each mixing cycle {meanPeriod} sec with % STD of {stdPeriod} %')

plt.figure(200)
plt.plot(timePeriods*dTime,'o')
plt.xlabel('Mixing Cycle #')
plt.ylabel('Time (s)')
plt.grid(True)
plt.ylim(0.5,2)