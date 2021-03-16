from IPython import display
import matplotlib.pyplot as plt
import numpy as np


plt.ion()

def visualize_Q_values(activations):
    mean_activations = activations.detach().cpu().numpy().mean(0)
    display.clear_output(wait=True)
    fig = plt.gcf()
    plt.title('Q-Values')
    plt.xlabel('Nodes')
    plt.ylabel('Activations')
    plt.bar(np.arange(mean_activations.shape[-1]), mean_activations)
    plt.ylim(ymin=0)
    plt.show(block=False)
    display.display(fig)
    return fig