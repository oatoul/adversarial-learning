import torch
import numpy as np
from classifier import Net as Model
import foolbox as fb #Load foolbox

device = 'cpu' #CHANGE THIS LINE TO 'cpu' IF NO GPU TO USE
model = Model().to(device)
model.load_state_dict(torch.load('../model/classifier.pt',
                                 map_location = torch.device(device)))
model.eval()

#Dummy Input and Targets
x = torch.normal(0, 1, (1, 156), device = device, requires_grad = True)
targets = torch.randint(0, 10, (1,), device = device)

#Compute loss and backpropagate
pred = model(x)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(pred, targets)
loss.backward()

#Printing gradient of loss function at x
print(x.grad)

#fmodel = fb.models.PyTorchModel(model, bounds=(0, 1), num_classes=10) #An example how to load foolbox pytorch model
