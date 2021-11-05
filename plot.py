import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from scipy.ndimage import gaussian_filter1d
# pth = 'train_logtt.txt'
# X1=[]
# Y1=[]
# X2=[]
# Y2=[]
# X3=[]
# Y3=[]
# X4=[]
# Y4=[]
# X5=[]
# Y5=[]
# sum_b=0
# sum_agent2 = 0
# with open(pth,'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         val = [float(s) for s in line.split()] # 10 5 state, 5 action         
#         Y1.append(val[0])
#         Y2.append(val[1])
#         Y3.append(val[2])
#         Y4.append(val[3])
#         Y5.append(val[4])
#         X1 = range(60)
# # model = make_interp_spline(X1, Y1)
# Y1_s = gaussian_filter1d(Y1, sigma=0.5)
# Y2_s = gaussian_filter1d(Y2, sigma=0.5)
# Y3_s = gaussian_filter1d(Y3, sigma=0.5)
# Y4_s = gaussian_filter1d(Y4, sigma=0.5)
# Y5_s = gaussian_filter1d(Y5, sigma=0.5)

# plt.plot(X1, Y1_s,alpha=0.6,label='agent1')
# plt.plot(X1, Y2_s,alpha=0.6,label='agent2')
# plt.plot(X1, Y3_s,alpha=0.6,label='agent3')
# plt.plot(X1, Y4_s,alpha=0.6,label='agent4')
# plt.plot(X1, Y5_s,alpha=0.6,label='agent5')


# plt.xlabel('Iteration Step')
# plt.ylabel('Bus Voltage')
# plt.legend()
# plt.show()

# control-action vs bus votage
pth = 'v-u.txt'
X1=[]
Y1=[]
X2=[]
Y2=[]
X3=[]
Y3=[]
X4=[]
Y4=[]
X5=[]
Y5=[]
sum_b=0
sum_agent2 = 0
with open(pth,'r') as f:
    lines = f.readlines()
    for line in lines:
        val = [float(s) for s in line.split()] # 10 5 state, 5 action         
        Y1.append(val[0+1])
        Y2.append(val[1+1])
        Y3.append(val[2+1])
        Y4.append(val[3+1])
        Y5.append(val[4+1])
        X1.append(val[0])

plt.plot(X1, Y1,alpha=0.6,label='agent1')
plt.plot(X1, Y2,alpha=0.6,label='agent2')
plt.plot(X1, Y3,alpha=0.6,label='agent3')
plt.plot(X1, Y4,alpha=0.6,label='agent4')
plt.plot(X1, Y5,alpha=0.6,label='agent5')


plt.xlabel('Bus Voltage')
plt.ylabel('Control action')
plt.legend()
plt.show()