import numpy as np
import os
import matplotlib.pyplot as plt


combine_inside_8 = np.array([0.901,0.958,0.976,0.093,0.388,0.042,0.165,15.312,0.09])
combine_outside_8 = np.array([0.826,0.923,0.96,0.136,0.487,0.062,0.212,18.868,0.124])
combine_inside_6 = np.array([0.899,0.957,0.974,0.094,0.375,0.042,0.161,14.570,0.097])
combine_outside_6 = np.array([0.786,0.915,0.960,0.162,0.496,0.071,0.232,20.248,0.129])
combine_inside_4 = np.array([0.9,0.96,0.979,0.093,0.284,0.04,0.138,11.738,0.083])
combine_outside_4 = np.array([0.727,0.9,0.955,0.189,0.537,0.083,0.258,22.185,0.149])
combine_inside_2 = np.array([0.89,0.958,0.979,0.101,0.215,0.041,0.121,8.328,0.09])
combine_outside_2 = np.array([0.676,0.887,0.95,0.219,0.551,0.091,0.274,23.021,0.17])

combine_inside = [combine_inside_2,combine_inside_4, combine_inside_6, combine_inside_8]
combine_outside = [combine_outside_2,combine_outside_4, combine_outside_6, combine_outside_8]

baseline_inside_8 = np.array([0.881,0.952,0.973,0.118,0.42,0.054,0.188,16.108,0.106])
baseline_outside_8 = np.array([0.799,0.914,0.957,0.157,0.516,0.071,0.234,20.039,0.145])
baseline_inside_6 = np.array([0.881,0.95,0.972,0.121,0.408,0.052,0.18,15.343,0.131])
baseline_outside_6 = np.array([0.755,0.906,0.956,0.189,0.525,0.079,0.25,21.598,0.169])
baseline_inside_4 = np.array([0.871,0.943,0.971,0.138,0.358,0.053,0.169,13.162,0.169])
baseline_outside_4 = np.array([0.722,0.888,0.945,0.228,0.582,0.086,0.271,22.859,0.256])
baseline_inside_2 = np.array([0.807,0.895,0.937,0.239,0.466,0.076,0.213,12.538,0.417])
baseline_outside_2 = np.array([0.565,0.788,0.889,0.407,0.789,0.127,0.363,25.461,0.652])

baseline_inside = [baseline_inside_2, baseline_inside_4, baseline_inside_6, baseline_inside_8]
baseline_outside = [baseline_outside_2,baseline_outside_4, baseline_outside_6, baseline_outside_8]

case = 0

# plt.plot([i*2 for i in [4, 3,2,1]],[combine_outside[i][case] for i in range(4)],'r')
# plt.plot([i*2 for i in [4, 3,2,1]],[baseline_outside[i][case] for i in range(4)],'b')
# plt.show()

#plot performance boost of combine and baseline
plt.plot([i*2 for i in range(1,5)],[combine_outside[i][case]-baseline_outside[i][case] for i in range(4)],'r',label='combine')
plt.plot([i*2 for i in range(1,5)],[combine_inside[i][case]-baseline_inside[i][case] for i in range(4)],'b',label='baseline')
plt.title(case)
plt.show()

