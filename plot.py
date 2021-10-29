import matplotlib.pyplot as plt
pth = '50cnn_log.txt'
X1=[]
Y1=[]
sum_b=0
sum_agent2 = 0
with open(pth,'r') as f:
    lines = f.readlines()
    for line in lines:
        val = [float(s) for s in line.split()] # 10 5 state, 5 action         
        Y1.append(val[0])
        # Y1.append(val[5])
        X1 = range(60)
plt.scatter(X1,Y1,alpha=0.6,label='agent1')

plt.xlabel('Iteration Step')
plt.ylabel('Bus Voltage')
plt.legend()
plt.show()