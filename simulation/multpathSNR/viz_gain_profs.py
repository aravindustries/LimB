import torch
import utils
import matplotlib.pyplot as plt

angs = torch.linspace(-torch.pi/4, torch.pi/4, 8)

plt.figure(figsize=(8, 6))

handles = []
labels = []

for alpha in angs:
    X = utils.create_tone(alpha.item(), 0) 
    y = utils.thetaScan(X, 64)
    line, = plt.plot(y, label=f"{alpha.item():.2f}") 
    handles.append(line)
    labels.append(f"{torch.rad2deg(alpha).item():.1f}°")  

plt.legend(handles, labels, loc="best", fontsize='small', ncol=2)
plt.xlabel("Index")
plt.ylabel("Gain")
plt.title("Gain Profile")
plt.savefig('gain profile.png')