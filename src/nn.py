class SigmaNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SigmaNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        y = self.output(x)
        return y

def compute_expectation(Sigma, num_samples=10000):
    expectation = 0
    for i in range(num_samples):
        H_bar = torch.ones((2, 3))
        H_tilda = torch.randn((2, 3))
        H = (1 / torch.sqrt(torch.tensor(2.0))) * (H_bar + H_tilda)
        det_term = torch.det(torch.eye(2) + (5/3)*torch.matmul(H, torch.matmul(Sigma, H.T)))
        expectation += torch.log(det_term)
        # print(H)
        # print(torch.log(det_term))
        # print('\n')
    return expectation / num_samples

def loss_function(y, p, num_samples=1000):
    Sigma = torch.matmul(y, y.T)
    neg_expectation = -compute_expectation(Sigma, num_samples)
    # trace_penalty = torch.maximum(torch.tensor(0.0), torch.trace(Sigma) - p)
    trace_penalty = (torch.trace(Sigma) - p)**2
    loss = neg_expectation + 10000*trace_penalty
    return loss

# Training setup
input_size = 6  # B_R (4x2), B_T (4x2), A1, rho2 (the amplitude of the second diagonal entry in A)
hidden_size = 90  # Number of hidden units
output_size = 9  # The output size is 2 for the real and imaginary parts of Sigma
learning_rate = 0.001

# Initialize the neural network model
model = SigmaNetwork(input_size, hidden_size, output_size)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Example data (input values for B_R, B_T, A1, rho2, and initial Sigma)
# y = torch.eye(3)
y = y_init
inp = torch.ones((2, 3))

num_epochs = 10000
p = 3.0 

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    inp = inp.flatten()
    
    y = model(inp)
    y = y.reshape(3, 3)

    # Compute the loss
    loss = loss_function(y, p)
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update the model parameters
    optimizer.step()
    
    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')