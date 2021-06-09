import matplotlib.pyplot as plt
import numpy as np

with open("5.15/data_subscription_data_in_main.txt", "r") as f:
    data = f.readlines()
important_timestamps = []
cur_millisecond = 0
cur_microsecond = 0
v_x = ''
v_y = ''
v_z = ''
max_v = 0
valid_timestamp_counter = 0
valid_v_counter = 0

total_v = np.zeros(612)
cur_v = 0
last_v = 0
valid_timestamp_start = []
valid_timestamp_stop = []

for i in range(0, len(data), 4):
    ## timestamp
    raw_timestamps = data[i].split()
    cur_millisecond = raw_timestamps[2]
    cur_microsecond = raw_timestamps[4].strip('.')
    if int(cur_millisecond) < 1009336 or int(cur_millisecond) > 1621876:
        continue
    # velocity
    raw_v = data[i + 2].split()
    v_x = raw_v[2]
    v_y = raw_v[4]
    v_z = raw_v[6].strip(',')
    if (float(v_x)) ** 2 + (float(v_y)) ** 2 > 20:
        valid_v_counter += 1
    total_v[valid_timestamp_counter] = ((float(v_x)) ** 2 + (float(v_y)) ** 2) ** 0.5
    cur_v = ((float(v_x)) ** 2 + (float(v_y)) ** 2) ** 0.5
    if last_v < 4 and cur_v >= 4:
        valid_timestamp_start.append((cur_millisecond, cur_microsecond))

    if last_v > 4 and cur_v < 4:
        valid_timestamp_stop.append((cur_millisecond, cur_microsecond))
    
    last_v = cur_v

    valid_timestamp_counter += 1

print("total number of valid points =", valid_timestamp_counter)
print("total number of valid velocity =", valid_v_counter)

print(valid_timestamp_start)
print(valid_timestamp_stop)

plt.figure()
plt.plot(range(612), total_v)
plt.savefig("velocity_time.png")
