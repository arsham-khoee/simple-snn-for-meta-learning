import argparse
import os
from time import time as t
from eagerpy import ones
import logging
import json
import sys
import torch
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.mnist_sampler import DatasetLoader, CategoriesSampler
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
from bindsnet.network.nodes import  DiehlAndCookNodes, Input, LIFNodes, AdaptiveLIFNodes
from bindsnet.network.topology import Connection, Conv2dConnection, MaxPool2dConnection, SparseConnection
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
parser.add_argument("--n_way", type=int, default=2)
parser.add_argument("--k_shot", type=int, default=5)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--n_tasks", type=int, default=100)
parser.add_argument("--n_test", type=int, default=20)
parser.add_argument("--n_train", type=int, default=80)
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
n_tasks = args.n_tasks
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
n_way = args.n_way
k_shot = args.k_shot

# Load MNIST data.

train_data = DatasetLoader(
    path = './data/mnist-meta', 
    time = time, 
    dt = dt,
    intensity = intensity)

train_sampler = CategoriesSampler(
    train_data.label, 
    n_tasks, 
    n_way, 
    k_shot)

task_loader = DataLoader(
    dataset=train_data, 
    batch_sampler=train_sampler, 
    num_workers=0, 
    pin_memory=gpu)

#train_loader = task_loader[:n_train]
#test_loader = task_loader[n_train:(n_train + n_test)]


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

w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[fltr1, i, j, fltr2, i, j] = -100.0

w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

lif_layer = LIFNodes(
    n=300, 
    #shape=(1, 1, n_filters),
    shape=(300, 1),
    traces=True,
  #  thresh=-52.5,
    refrac=0,
)

conv_lif_conn = Connection(
    conv_layer,
    lif_layer,
    w=torch.multiply((0.05 + torch.randn(conv_layer.n, lif_layer.n)), torch.empty(conv_layer.n, lif_layer.n).random_(2)),
    wmin=0,  # minimum weight value
    wmax=1,  # maximum weight value
    update_rule=MSTDPET,  # learning rule
    nu=1e-1,  # learning rate
    norm= lif_layer.n * 0.5 ,  # normalization
)

w_lif_lif = -1 * torch.ones(lif_layer.n, lif_layer.n)
lif_lif_conn = Connection(
    lif_layer,
    lif_layer, 
    w = w_lif_lif, #-0.15 * torch.ones(lif_layer.n, lif_layer.n),
    #w=-1*torch.multiply((0.05 + torch.randn(lif_layer_c1.n, lif_layer_c1.n)), torch.empty(lif_layer_c1.n, lif_layer_c1.n).random_(2)),
    #wmin=-1,  # minimum weight value
    #wmax=0,  # maximum weight value
    #update_rule=MSTDPET,  # learning rule
    #nu=1e-1,  # learning rate
    #norm= lif_layer_c1.n * 0.5 ,  # normalization
)

pred_layer = LIFNodes(
    n=10*n_way,
    #shape=(1, 1, n_filters),
    shape=(10*n_way, 1),
    traces=True,
    thresh= -62.0,
)

lif_pred_conn = Connection(
    lif_layer,
    pred_layer, 
    w = torch.multiply((0.05 + torch.randn(lif_layer.n, pred_layer.n)), torch.empty(lif_layer.n, pred_layer.n).random_(2)),
    wmin=0,  # minimum weight value
    wmax=1,  # maximum weight value
    update_rule=MSTDPET,  # learning rule
    nu=1e-1,  # learning rate
    norm= pred_layer.n * 2 # * 0.5,  # normalization
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
w = torch.kron(torch.eye(n_way),torch.ones([10, 10]))
w[w == 0] = -3
w[w == 1] = 0 # in khate mitune nabashe
pred_pred_conn = Connection(pred_layer, pred_layer, w=w) # w = -1 * (torch.ones(pred_layer.n, pred_layer.n).fill_diagonal_(-1))


network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
network.add_layer(lif_layer, name="E")
network.add_layer(pred_layer, name="P")

network.add_connection(conv_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")
network.add_connection(conv_lif_conn, source="Y", target="E")
network.add_connection(lif_lif_conn, source="E", target="E")
network.add_connection(lif_pred_conn, source="E", target="P")
network.add_connection(pred_pred_conn, source="P", target="P")


# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network.to("cuda")

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


rewards = torch.zeros(time)
reward = 0

wn_mean=0
wn_std=3

for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()
        
    total_reward = 0

    total_corrects_count = 0
    total_attempts_count = 0

    for step, task in enumerate(tqdm(task_loader)):
        labels = torch.sort(torch.unique(task['label']))[0]
        task = [{"image": task['image'][idx], "encoded_image":task['encoded_image'][idx], "label":task['label'][idx]} for idx in range(task['image'].shape[0])]
        # Get next input sample.  inaro dadam jolo bara if
        if step > n_train:  
            break
        print("EPISODE: " + str(step))
        conv_pred_conn.learnable = False
        for batch_idx, batch in enumerate(task):
            inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, 28, 28)}
            if gpu: 
                inputs = {k: v.cuda() for k, v in inputs.items()}
            label = batch["label"]
            block_n = int(lif_layer.n / 10)
            lif_lif_conn.w[lif_lif_conn.w == 0] = -1
            lif_lif_conn.w[label.item()*block_n:(label.item()+1)*block_n, label.item()*block_n:(label.item()+1)*block_n] = 0

            total_attempts_count += 1

            reward = 0 ###

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
            lif_firing_rate = spikes["E"].get("s").float().squeeze().T.sum(axis=1)
            #lif_firing_rate[lif_firing_rate > 0] = 1 
            #lif_activity_pct = (torch.sum(lif_firing_rate).item() / lif_layer.n)*100
            lif_activity_pct = (torch.sum(lif_firing_rate).item() / (lif_layer.n*time))*100
            #print(lif_activity_pct)

            pred_firing_rate = spikes["P"].get("s").float().squeeze().T.sum(axis=1)
            #print("pred_layer firing rate: " + str(pred_firing_rate))

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
            m = torch.zeros(n_way, dtype=torch.bool)

            for i in range(len(p)):
                m[int(p[i] / 10)] = True

            c = torch.where(m == True)[0]

            if len(c) == 1 and c.item() == (labels == label.item()).nonzero(as_tuple=True)[0].item():
                total_corrects_count += 1

            print("Predicted classes: " + str(c) + ", Label: " + str((labels == label.item()).nonzero(as_tuple=True)[0].item())) # print(label.data[0])
                    
            if batch_idx in range(0, n_way*k_shot):
                print("INNER LOOP ADAPTATION: " + str(batch_idx))
                if lif_activity_pct >= 2.5 and lif_activity_pct < 7.5:
                    reward = 1
                else:
                    reward = -1

            if batch_idx in range(n_way*k_shot,(n_way*k_shot)+n_way):
                print("OUTER LOOP ADAPTATION: " + str(batch_idx))
                conv_pred_conn.learnable = True

                if len(c) > 1 and torch.all(pred_firing_rate == 0):
                    reward = 0
                elif len(c) > 1 and not(torch.all(pred_firing_rate == 0)): 
                    reward = -1
                elif c.item() == (labels == label.item()).nonzero(as_tuple=True)[0].item():
                    reward = 1
                    if wn_std > (0.01 / n_way):
                        wn_std -= (0.01 / n_way) 
                elif c.item() != (labels == label.item()).nonzero(as_tuple=True)[0].item():
                    reward = -1

            total_reward += reward
            #print("reward = " + str(reward) + ", total_reward = " + str(total_reward))

            # Update network with cumulative reward
            if network.reward_fn is not None:
                network.reward_fn.update(accumulated_reward=total_reward)

            # reset the network before updating the reward (test)
            network.reset_state_variables() ###

            print("Updated reward: " + str(reward))
            # Update network with new reward
            network.run(inputs=inputs, time=time, input_time_dim=1, reward=reward) ###

            # Optionally plot various simulation information.
            if plot and batch_size == 1:
                image = batch["image"].view(28, 28)

                inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
                weights1 = conv_conn.w
                _spikes = {
                    "X": spikes["X"].get("s").view(time, -1),
                    "Y": spikes["Y"].get("s").view(time, -1),
                    "E": spikes["E"].get("s").view(time, -1),
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
            
    print("Training Accuracy: " + str(total_corrects_count/total_attempts_count))
    df = pd.DataFrame(columns=['epoch', 'accuracy'])
    new_row = {
        'epoch': epoch,
        'accuracy': total_corrects_count/total_attempts_count
    }
    df = df.append(new_row, ignore_index=True)
    df.to_csv('result.csv', mode='a', index=False, header=False)

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")
