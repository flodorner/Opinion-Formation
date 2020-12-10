from os import listdir
import re, pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
lsdr=listdir("results_max_s")
n=len(lsdr)
i=0
y=np.zeros(n)
sd=np.zeros(n)
x=np.zeros(n)

for subf in lsdr:
    phi=re.search("phi([0-9.]+)",subf)[1]
    x[i]=phi
    with open("results_max_s/"+subf, "rb") as f:
        subresult=pickle.load(f)
        
    y[i]=np.mean(subresult["max_connected_components"])
    sd[i]=np.std(subresult["max_connected_components"])
    i=i+1
results={"x":x,"y":y,"sd":sd}
with open("maxSphi0-458.pickle", "wb") as f:
    pickle.dump(results, f)


mpl.rcParams['errorbar.capsize'] = 3

fig=plt.figure(figsize=(5,3.4))
ax = fig.add_axes([0.12, 0.15, 0.87, 0.74])
plt.errorbar(x,y,yerr=sd,fmt="o")
sorti=np.argsort(x)

print(x,y,sd)

plt.ylabel("S")
plt.xlabel("ϕ")
plt.title("Max community size phase transition")
plt.plot([0.38,0.5],[48, 48], linewidth=4)
plt.errorbar(x[sorti],y[sorti])
plt.savefig("maxS")

fig=plt.figure(figsize=(4,3))
ax = fig.add_axes([0.15, 0.15, 0.78, 0.7])
plt.errorbar(x,y,yerr=sd,fmt="o")
plt.xlim((0.38,0.5))
plt.ylabel("S")
plt.xlabel("ϕ")

plt.plot([0.39,0.495],[48, 48], linewidth=4)
plt.title("phase transition detail")
plt.savefig("maxSdetail")
    