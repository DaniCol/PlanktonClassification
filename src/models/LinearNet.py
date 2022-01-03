import torch
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

x = torch.randn(2, 1, 300, 300)
model = LinearNet(300*300,10)
output = model(x)

# Should return torch.Size([2, 10])
print(output.shape)