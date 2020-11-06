from os import listdir
import re, pickle
import numpy as np
from matplotlib import pyplot as plt
lsdr=listdir("sub2")
n=len(lsdr)
i=0
y=np.zeros(n)
sd=np.zeros(n)
x=np.zeros(n)

for subf in lsdr:
    phi=re.search("phi([0-9.]+)",subf)[1]
    x[i]=phi
    with open("sub2/"+subf, "rb") as f:
        subresult=pickle.load(f)
        
    y[i]=np.mean(subresult["max_connected_components"])
    sd[i]=np.std(subresult["max_connected_components"])
    i=i+1
results={"x":x,"y":y,"sd":sd}
with open("maxSphi0-458.pickle", "wb") as f:
    pickle.dump(results, f)
plt.errorbar(x,y,yerr=sd,fmt="o")
print(x,y,sd)
plt.savefig("maxS")
plt.xlim((0.38,0.5))
 
plt.savefig("maxSdetail")
    