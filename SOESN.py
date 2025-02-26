import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class EchoStateNetwork(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=1.5, leaky_rate=0.5):
        super(EchoStateNetwork, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leaky_rate = leaky_rate

        # Initialize the reservoir (random but sparse)
        self.W_reservoir = torch.randn(reservoir_size, reservoir_size) * 0.2
        self.W_reservoir = self.W_reservoir - torch.diag(torch.diag(self.W_reservoir))  # Set diagonal to 0
        self.W_reservoir = self.spectral_normalization(self.W_reservoir, spectral_radius)

        # Initialize input weights (but we'll use feedback instead of real input)
        self.W_input = torch.randn(reservoir_size, input_size) * 0.2 if input_size > 0 else torch.zeros(reservoir_size,
                                                                                                        input_size)

        # Initialize output weights (random, to be trained)
        self.W_output = torch.randn(output_size, reservoir_size) * 0.2

        # States
        self.reservoir_state = torch.zeros(reservoir_size)

    def spectral_normalization(self, W, radius):
        """Normalize the reservoir matrix to ensure spectral radius is <= radius"""
        eigvals, _ = torch.linalg.eig(W)  # Compute eigenvalues using torch.linalg.eig
        eigvals_real = eigvals.real  # Extract real part of eigenvalues
        spectral_radius = torch.max(torch.abs(eigvals_real))  # Find the maximum absolute eigenvalue
        return W * (radius / spectral_radius)

    def forward(self, feedback):
        # Update the reservoir state with feedback loop (leaky update)
        self.reservoir_state = (1 - self.leaky_rate) * self.reservoir_state + self.leaky_rate * torch.tanh(
            torch.matmul(self.W_reservoir, self.reservoir_state) + feedback)
        # Compute the output
        output = torch.matmul(self.W_output, self.reservoir_state)
        return output


def train_esn_v2(esn, num_steps, learning_rate=0.01, feedback_strength=0.5):
    """
    This method includes a feedback mechanism in training.
    The feedback_strength controls how much the output affects the next input.
    """
    # Collect reservoir states and desired outputs
    states = []
    targets = []

    # Adding feedback loop to generate self-oscillatory behavior
    feedback = torch.randn(esn.reservoir_size) * 0.1  # Initialize feedback with the same size as reservoir_state
    print("feedback:", feedback.shape)#[300]
    print("feedback:", feedback)

    for i in range(num_steps):
        print(i)
        # No external input, only feedback
        output = esn.forward(feedback)
        states.append(esn.reservoir_state.detach().numpy())
        targets.append(output.detach().numpy())

        # Update feedback by adding the output to the feedback signal
        feedback = output * feedback_strength  # Use output as feedback to next iteration
        print("feedback:", feedback.shape)
        print("feedback:", feedback)

    # Convert to numpy arrays
    states = np.array(states)
    targets = np.array(targets)

    # Train output weights using linear regression
    X = np.linalg.pinv(states.T @ states) @ states.T @ targets
    print("X:", X.shape)#(300, 1)
    esn.W_output = torch.tensor(X.T, dtype=torch.float32)


def plot_oscillatory_behavior_v2(esn, num_steps=500):
    states = []
    for _ in range(num_steps):
        output = esn.forward(torch.zeros(esn.input_size))  # No external input
        print("output:", output.shape)
        states.append(esn.reservoir_state.detach().numpy())

    states = np.array(states)

    # Ensure states has the correct shape before plotting
    if states.ndim == 2:
        plt.figure(figsize=(10, 6))
        for i in range(min(states.shape[1], 30)):  # Show up to 30 neurons to get a clearer view
            plt.plot(states[:, i], label=f"Neuron {i + 1}")
        plt.title("Self-Oscillatory Behavior of the Echo State Network")
        plt.xlabel("Time Step")
        plt.ylabel("State Value")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
    else:
        print("Error: The state data is not in the correct format. States should be a 2D array.")


def plot_oscillatory_behavior_separate(esn, num_steps=500):
    states = []
    for _ in range(num_steps):
        output = esn.forward(torch.zeros(esn.input_size))  # No external input
        states.append(esn.reservoir_state.detach().numpy())

    states = np.array(states)

    # Number of neurons to visualize (limit to the first 30 neurons for better visualization)
    num_neurons = min(states.shape[1], 5)

    # Create subplots for each neuron
    fig, axes = plt.subplots(num_neurons, 1, figsize=(10, 2 * num_neurons), sharex=True)
    if num_neurons == 1:
        axes = [axes]  # Make sure axes is iterable even when there's only one plot

    for i in range(num_neurons):
        axes[i].plot(states[:, i], label=f"Neuron {i + 1}")
        axes[i].set_title(f"Neuron {i + 1}")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("State Value")
        axes[i].legend()

    plt.tight_layout()
    plt.show()


# Show Self-Oscillatory
def animate_oscillation(esn, num_steps=500, interval=10):
    states = []
    feedback = torch.randn(esn.reservoir_size) * 0.1

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    lines = [axes.plot([], [], label=f"Neuron {i + 1}")[0] for i in range(min(30, esn.reservoir_size))]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        nonlocal feedback

        output = esn.forward(feedback)
        states.append(esn.reservoir_state.detach().numpy())
        feedback = output * 0.8  # update feedback

        states_array = np.array(states)
        for i, line in enumerate(lines):
            line.set_data(np.arange(len(states_array)), states_array[:, i])

        axes.set_xlim(0, len(states_array))
        axes.set_ylim(min(states_array.min(), -1.0), max(states_array.max(), 1.0))
        # print("update")

        if frame >= num_steps - 1:
            ani.event_source.stop()
        return lines

    ani = animation.FuncAnimation(fig, update, frames=num_steps, init_func=init, interval=interval)

    plt.title("Real-Time Self-Oscillatory Behavior of the Echo State Network")
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.show()

    return esn.W_reservoir


def save_reservoir_parameters(esn, filepath="esn_reservoir.pth"):
    torch.save(esn.W_reservoir, filepath)
    print(f"Reservoir parameters saved to {filepath}")

def load_reservoir_parameters(esn, filepath="esn_reservoir.pth"):
    esn.W_reservoir = torch.load(filepath)
    print(f"Reservoir parameters loaded from {filepath}")

esn_v2 = EchoStateNetwork(input_size=0, reservoir_size=300, output_size=1, spectral_radius=1.5)

load_reservoir_parameters(esn_v2)

updated_reservoir = animate_oscillation(esn_v2, num_steps=300, interval=50)

save_input = input("If you are happy with the oscillation behavior, type 'save' to save the reservoir parameters: ")
if save_input.lower() == 'save':
    save_reservoir_parameters(esn_v2)

load_input = input("To load the saved reservoir parameters next time, type 'load': ")
if load_input.lower() == 'load':
    load_reservoir_parameters(esn_v2)


# train_esn_v2(esn_v2, num_steps=3000, learning_rate=0.01, feedback_strength=0.8)

# plot_oscillatory_behavior_separate(esn_v2)
