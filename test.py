import torch
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

x = torch.rand(size=(2, 2), requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=1, epochs=100
)
lr_list = []
lr_scheduler = []
for epoch in range(100):
    lr_list.append(optimizer.param_groups[0]["lr"])
    optimizer.step()
    lr_scheduler.append(scheduler.get_last_lr()[0])
    scheduler.step()
plt.plot(lr_list, color="red")
plt.plot(lr_scheduler, color="blue")
plt.show()
