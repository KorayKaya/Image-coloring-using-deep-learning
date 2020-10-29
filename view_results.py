import numpy as np

with open("results.txt", "r") as f:
    fileinfo =  f.read().split('\n')

data=[]
for line in fileinfo:
    post        =   []
    data_line   =   line.split(',')
    for data_post in data_line:
        number  =   float(data_post)
        post.append(number)
    data.append(post)

data    =   np.array(data)
epochs  =   np.array(list(range(len(data))))
print(data.shape)

import matplotlib.pyplot as plt

fig, (ax1, ax2) =   plt.subplots(ncols=2)
ax1.plot(epochs, data[:,0], label="Training loss")
ax1.plot(epochs, data[:,2], label="Validation loss")
ax2.plot(epochs, data[:,1], label="Training accuracy")
ax2.plot(epochs, data[:,3], label="Validation accuracy")
ax1.set_xlabel("Epochs")
ax2.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")
ax1.legend()
ax2.legend()
ax1.title.set_text("Loss")
ax2.title.set_text("Accuracy")
plt.show()
