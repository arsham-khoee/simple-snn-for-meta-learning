import argparse
import os
from time import time as t
from eagerpy import ones
import logging
import json

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_conv2d_weights,
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre, WeightDependentPostPre, MSTDP, MSTDPET, Rmax
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import  DiehlAndCookNodes, Input, LIFNodes, AdaptiveLIFNodes, MaxPool2dLIFNodes
from bindsnet.network.topology import Connection, Conv2dConnection, MaxPool2dConnection, GlobalMaxPoolConnection, SparseConnection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax

print()

formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
torch.set_printoptions(threshold=torch.nan)

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--kernel_size", type=int, default=8)
parser.add_argument("--stride", type=int, default=2)
parser.add_argument("--dilation", type=int, default=1)
parser.add_argument("--n_filters", type=int, default=30)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
batch_size = args.batch_size
kernel_size = args.kernel_size
stride = args.stride
dilation=args.dilation
n_filters = args.n_filters
padding = args.padding
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if not train:
    update_interval = n_test

conv_size = int((28 - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

c_num = list(range(0, 2)) # kelasaye morede niaz

# Build network.
network = Network()

input_layer = Input(n=784, shape=(1, 28, 28), traces=True)

conv_layer = LIFNodes(
    n=n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    dilation=dilation,
    #update_rule=PostPre,
    norm=0.4 * kernel_size**2,
    nu=[1e-4, 1e-2],
    wmax=1.0,
    wmin=0,
)

conv_conn.w = torch.load('weights1.pt') # load pretrained conv2d weights using STDP

out_layer = MaxPool2dLIFNodes(
    n=n_filters,
    shape=(n_filters, 1, 1),
)

out_conn = GlobalMaxPoolConnection(
    source=conv_layer,
    target=out_layer,
)

w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[fltr1, i, j, fltr2, i, j] = -100.0

w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)
recurrent_conn = Connection(conv_layer, conv_layer, w=w)


pred_layer = LIFNodes(
    n=10*len(c_num),
    #shape=(1, 1, n_filters),
    shape=(10*len(c_num), 1),
    traces=True,
    thresh= -60.0,
)

conv_pred_conn = Connection(
    conv_layer,
    pred_layer, 
    w = torch.multiply((0.05 + torch.randn(conv_layer.n, pred_layer.n)), torch.empty(conv_layer.n, pred_layer.n).random_(2)),
    wmin=0,  # minimum weight value
    wmax=1,  # maximum weight value
    update_rule=MSTDPET,  # learning rule
    nu=1e-1,  # learning rate
    norm= pred_layer.n, # * 0.5,  # normalization
)

# lateral connection
w = torch.kron(torch.eye(len(c_num)),torch.ones([10, 10]))
w[w == 0] = -3
w[w == 1] = 0 # in khate mitune nabashe
pred_pred_conn = Connection(pred_layer, pred_layer, w=w) # w = -1 * (torch.ones(pred_layer.n, pred_layer.n).fill_diagonal_(-1))


network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
#network.add_layer(out_layer, name="O")
network.add_layer(pred_layer, name="P")

network.add_connection(conv_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")
#network.add_connection(out_conn, source="Y", target="O")
#network.add_connection(o_pred_conn, source="O", target="P")
network.add_connection(conv_pred_conn, source="Y", target="P")
network.add_connection(pred_pred_conn, source="P", target="P")


# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network.to("cuda")

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights1_im = None
voltage_ims = None
voltage_axes = None


json_string = """
[]
"""
json.dump( json.loads(json_string), open( "spiking_mnist.json", 'w' ) )


total_reward = 0
rewards = torch.zeros(time)
reward = 0

wn_mean=0
wn_std=3

for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=gpu,
    )

    conv_pred_conn.learnable = False

    for step, batch in enumerate(tqdm(train_dataloader)):

        # Get next input sample.  
        if step > n_train:
            break
        inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, 28, 28)}
        if gpu: 
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        if (label.item() in c_num):

            # inaro dadam jolo bara if
            if reward > 0 and wn_std > (0.01 / len(c_num)):
                wn_std -= (0.01 / len(c_num)) # 
                #pred_pred_conn.w[pred_pred_conn.w < 0] += 0.005 # 0.001 

            #reward = 0 ###

            #print("Input shape: " + str(inputs['X'].shape))

            print("Initial reward: " + str(reward))

            print("White noise std: " + str(wn_std))
            wn = torch.normal(mean=wn_mean, std=wn_std, size=(time , 1, pred_layer.n, 1))
            # Run the network on the input.
            network.run(inputs=inputs, time=time, input_time_dim=1, reward=reward, injects_v={"P": wn.to(device)}) 
            #input_layer.reset_state_variables()
            #conv_layer.reset_state_variables()
            #out_layer.reset_state_variables()

            #logger_o = setup_logger('out_layer', 'out_layer.log')
            #logger_o.info(out_layer.s.float().squeeze())

            #logger_p = setup_logger('pred_layer', 'pred_layer.log')
            #logger_p.info("step: " + str(step) + " " + str(voltages["P"].get("v").float().squeeze()))

            pred_firing_rate = spikes["P"].get("s").float().squeeze().T.sum(axis=1)
            print("pred_layer firing rate: " + str(pred_firing_rate))

            p = torch.where(pred_firing_rate == torch.amax(pred_firing_rate))[0]
            #print("Index of spiking neurons: " + str(p))

            #if len(p) > 1 and torch.all(pred_firing_rate == 0):
            #    reward = 0
            #elif len(p) > 1 and not(torch.all(pred_firing_rate == 0)): 
            #    reward = -1
            #elif p.item() == label.item():
            #    reward = 1
            #elif p.item() != label.item():
            #    reward = -1
        
            # rewarding policy for 10 neuron per class in pred_layer
            m = torch.zeros(len(c_num), dtype=torch.bool)

            for i in range(len(p)):
                m[int(p[i] / 10)] = True

            c = torch.where(m == True)[0]

            print("Predicted classes: " + str(c))
        
            print("Label: " + str(label.item())) # print(label.data[0])

            if len(c) > 1 and torch.all(pred_firing_rate == 0):
                reward = 0
            elif len(c) > 1 and not(torch.all(pred_firing_rate == 0)): 
                reward = -1
            elif c.item() == label.item():
                reward = 1
            elif c.item() != label.item():
                reward = -1

            total_reward += reward
            #print("reward = " + str(reward) + ", total_reward = " + str(total_reward))

            # Update network with cumulative reward
            if network.reward_fn is not None:
                network.reward_fn.update(accumulated_reward=total_reward)

            # reset the network before updating the reward (test)
            #network.reset_state_variables() ###

            print("Updated reward: " + str(reward))
            # Update network with new reward
            #network.run(inputs=inputs, time=time, input_time_dim=1, reward=reward) ###

            # Optionally plot various simulation information.
            if plot and batch_size == 1:
                image = batch["image"].view(28, 28)

                inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
                weights1 = conv_conn.w
                _spikes = {
                    "X": spikes["X"].get("s").view(time, -1),
                    "Y": spikes["Y"].get("s").view(time, -1),
                    "P": spikes["P"].get("s").view(time, -1),                
                }
                _voltages = {"Y": voltages["Y"].get("v").view(time, -1)}

                inpt_axes, inpt_ims = plot_input(
                    image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
                )
                spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
                weights1_im = plot_conv2d_weights(weights1, im=weights1_im)
            
                voltage_ims, voltage_axes = plot_voltages(
                    _voltages, ims=voltage_ims, axes=voltage_axes
                )

                plt.pause(1)

                #print("step: " + str(step))
                #if step < 60000:
                    #with open('spiking_mnist.json', "r+") as file:
                        #data = json.load(file)
                        #data.append({"data": spikes["O"].get("s").float().squeeze().tolist(), "label": label.item()})
                        #file.seek(0)
                        #json.dump(data, file)
                    # torch.save(weights1, 'weights1.pt')
                    # print("weights1 saved to file successfully!")

            network.reset_state_variables()  # Reset state variables.
        
print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")

