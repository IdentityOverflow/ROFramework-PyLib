import torch
import matplotlib.pyplot as plt

##x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
x = torch.linspace(-3, 3, 3000).unsqueeze(1)
y = x**2

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = torch.nn.Linear(1, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 1) 
        
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x
    
model = SimpleNet()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3000):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

test = torch.tensor([[-3.0], [-1.5], [0.5], [2.5]])
with torch.no_grad():
    print(model(test))

# with torch.no_grad():
#     preds = model(x)

# plt.scatter(x.numpy(), y.numpy(), label="true")
# plt.plot(x.numpy(), preds.numpy(), label="model")
# plt.legend()
# plt.show()