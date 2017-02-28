import numpy as np
import matplotlib.pyplot as plt
import math

noise = np.array([59,71])
Ns = np.mean(noise)

I = np.array([570,620,603])
I0 = np.mean(I)
err_I0 = math.sqrt(np.std(I)**2/2 + Ns**2)

d_Al = np.array([0,2,4,6,8,10])*0.1332
d_Cu = np.array([0,2,4,6,8,10])*0.0682
d_Pb = np.array([0,1,2,3,4])*0.205

I_al = np.array([[I0,I0], [570,533], [504,533], [481,529], [551,490], [493,508]]) - Ns
err_al = np.asarray(list(map(lambda x: math.sqrt(np.std(I_al[x])**2 + Ns**2 + np.std(noise)**2), range(len(I_al)))))
I_cu = np.array([[I0,I0], [549,560], [464,468], [446,484], [431,430], [468,415]]) - Ns
err_cu = np.asarray(list(map(lambda x: math.sqrt(np.std(I_cu[x])**2 + Ns**2 + np.std(noise)**2), range(len(I_cu)))))
I_pb = np.array([[I0,I0], [513,501], [407,447], [334,334], [274,285]]) - Ns
err_pb = np.asarray(list(map(lambda x: math.sqrt(np.std(I_pb[x])**2 + Ns**2 + np.std(noise)**2), range(len(I_pb)))))
I0 -= Ns
# --------------------------------------------------------------------------------------------------
err_d = np.array([0,2,4,6,8,10])*0.0016
err_al = np.asarray(list(map(lambda x: math.sqrt(err_I0**2*np.mean(I_al[x])**2/I0**2+err_al[x]**2)/np.mean(I_al[x]), range(len(I_al)))))	
I_al = np.log(list(map(lambda x: np.mean(I_al[x]), range(len(I_al))))/I0)
A = np.vstack([d_Al, np.ones(len(d_Al))]).T
m,c = np.linalg.lstsq(A, I_al)[0]
reducials = np.linalg.lstsq(A, I_al)[1]

#print("error : ", math.sqrt(reducials/len(x)/(len(x)-1)))
print("log linear coef: ", m)

fig = plt.figure()
x = np.linspace(-0.25*np.max(d_Al),1.5*np.max(d_Al),2)
# plt.ylim(-0.25,0.05)
plt.xlabel('plate thickness, cm')
plt.ylabel('lg(I/I0)')
plt.errorbar(d_Al, I_al, yerr=err_al, xerr=err_d, fmt='o')
#plt.plot(d1, I_al, 'o', label='Original data', markersize=10)
plt.plot(x, m*x + c, 'r', label='Fitted line')
plt.legend()
title = "Fit: y = %.4f * x + %.4f" % (m, c)
plt.title(title)
plt.show()
fig.savefig('Al_errors.eps')
