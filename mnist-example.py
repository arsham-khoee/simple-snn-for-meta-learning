import argparse
import os, sys
from time import time as t

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_conv2d_weights,
    plot_input,
    plot_spikes,
    plot_voltages,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre, WeightDependentPostPre, MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes, AdaptiveLIFNodes
from bindsnet.network.topology import Connection, Conv2dConnection, LocalConnection, GlobalMaxPoolConnection, SparseConnection
from bindsnet.pipeline.action import select_softmax
from bindsnet.pipeline import EnvironmentPipeline

print()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--n_filters", type=int, default=25)
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

# Build network.
network = Network()
input_layer = Input(n=784, shape=(1, 28, 28), traces=True)

# layers
lif_layer = LIFNodes(
    n=n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)
#adaptive_layer = DiehlAndCookNodes(
#    n=n_filters * conv_size * conv_size,
#    shape=(n_filters, conv_size, conv_size),
#    tc_decay=10.0,
#    tc_theta_decay=1000.0,
#    traces=True,
#)

adaptive_layer = AdaptiveLIFNodes(
    n=n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

inhibit_layer = LIFNodes(
    n = n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces = True,
)

lif_out = LIFNodes(
    n=4,
   #shape=(n_filters, conv_size, conv_size),
    traces=True,
)

adaptive_out_conn = GlobalMaxPoolConnection(
    source=adaptive_layer,
    target=lif_out,
    wmin=0,
    wmax=1,
   # update_rule=MSTDPET,
    norm=0.5 * lif_out.n,
    kernel_size=kernel_size,
)

lif_out_conn = GlobalMaxPoolConnection(
    source=lif_layer,
    target=lif_out,
    wmin=0,
    wmax=1,
   # update_rule=MSTDPET,
    norm=0.5 * lif_out.n,
    kernel_size=kernel_size,
)

inhibit_out_conn = GlobalMaxPoolConnection(
    source=inhibit_layer,
    target=lif_out,
    wmin=0,
    wmax=1,
   # update_rule=MSTDPET,
    norm=0.5 * lif_out.n,
    kernel_size=kernel_size,
)

# Conntections
input_adaptive_conn = Conv2dConnection(
    input_layer,
    adaptive_layer,
    kernel_size=kernel_size,
    stride=stride,
   # update_rule=WeightDependentPostPre,
    norm=0.4 * kernel_size ** 2,
    nu=[1e-3, 1e-2],
    wmax=1.0,
    wmin=0,
  #  w = (0.05 + 0.1 * torch.randn(input_layer.n, adaptive_layer.n)) * torch.empty(input_layer.n, adaptive_layer.n).random_(2),
)
input_lif_conn = Conv2dConnection(
    input_layer,
    lif_layer,
    kernel_size=kernel_size,
    stride=stride,
   # update_rule=WeightDependentPostPre,
    norm=0.4 * kernel_size ** 2,
    nu=[1e-3, 1e-2],
    wmax=1.0,
    wmin=0,
  #  w = (0.05 + 0.1 * torch.randn(input_layer.n, lif_layer.n)) * torch.empty(input_layer.n, lif_layer.n).random_(2),
)
input_inhibit_conn = Conv2dConnection(
    input_layer,
    inhibit_layer,
    kernel_size=kernel_size,
    stride=stride,
   # update_rule=WeightDependentPostPre,
    norm=0.4 * kernel_size ** 2,
    nu=[1e-3, 1e-2],
    wmax=1.0,
    wmin=0,
   # w = (0.05 + 0.1 * torch.randn(input_layer.n, inhibit_layer.n)) * torch.empty(input_layer.n, inhibit_layer.n).random_(2),
)
adaptive_recurrent_conn = Connection(
    adaptive_layer, 
    adaptive_layer, 
  #  w = (0.05 + 0.1 * torch.randn(adaptive_layer.n, adaptive_layer.n)) * torch.empty(adaptive_layer.n, adaptive_layer.n).random_(2),
    # w=w
)
lif_recurrent_conn = Connection(
    lif_layer, 
    lif_layer, 
 #   w = (0.05 + 0.1 * torch.randn(lif_layer.n, lif_layer.n)) * torch.empty(lif_layer.n, lif_layer.n).random_(2),
    # w=w,
)
lif_adaptive_conn = Connection(
    lif_layer, 
    adaptive_layer,
   # w = (0.05 + 0.1 * torch.randn(lif_layer.n, adaptive_layer.n)) * torch.empty(lif_layer.n, adaptive_layer.n).random_(2),

)
adaptive_lif_conn = Connection(
    adaptive_layer,
    lif_layer,
  #  w = (0.05 + 0.1 * torch.randn(adaptive_layer.n, lif_layer.n)) * torch.empty(adaptive_layer.n, lif_layer.n).random_(2),
)

inhibit_lif_conn = Connection(
    inhibit_layer,
    lif_layer,
    wmin= -1.0,
    wmax= -0.1,
  #  w = (0.05 + 0.1 * torch.randn(inhibit_layer.n, lif_layer.n)) * torch.empty(inhibit_layer.n, lif_layer.n).random_(2),
)
lif_inhibit_conn = Connection(
    lif_layer,
    inhibit_layer,
  #  w = (0.05 + 0.1 * torch.randn(lif_layer.n, inhibit_layer.n)) * torch.empty(lif_layer.n, inhibit_layer.n).random_(2),
)
inhibit_adaptive_conn = Connection(
    inhibit_layer,
    adaptive_layer,
    wmin = -1.0,
    wmax = -0.1,
  #  w = (0.05 + 0.1 * torch.randn(inhibit_layer.n, adaptive_layer.n)) * torch.empty(inhibit_layer.n, adaptive_layer.n).random_(2),
)
adaptive_inhibit_conn = Connection(
    adaptive_layer,
    inhibit_layer,
   # w = (0.05 + 0.1 * torch.randn(adaptive_layer.n, inhibit_layer.n)) * torch.empty(adaptive_layer.n, inhibit_layer.n).random_(2),
)

adaptive_adaptive_conn = Connection(
    adaptive_layer,
    adaptive_layer,
  #  w = (0.05 + 0.1 * torch.randn(adaptive_layer.n, adaptive_layer.n)) * torch.empty(adaptive_layer.n, adaptive_layer.n).random_(2),
)

lif_lif_conn = Connection(
    lif_layer,
    lif_layer,
   # w = (0.05 + 0.1 * torch.randn(lif_layer.n, lif_layer.n)) * torch.empty(lif_layer.n, lif_layer.n).random_(2),
)

inhibit_inhibit_conn = Connection(
    inhibit_layer,
    inhibit_layer,
  #  w = (0.05 + 0.1 * torch.randn(inhibit_layer.n, inhibit_layer.n)) * torch.empty(inhibit_layer.n, inhibit_layer.n).random_(2),
)
# w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
# for fltr1 in range(n_filters):
#     for fltr2 in range(n_filters):
#         if fltr1 != fltr2:
#             for i in range(conv_size):
#                 for j in range(conv_size):
#                     w[fltr1, i, j, fltr2, i, j] = -100.0

# w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)


network.add_layer( 
    input_layer, 
    name="input",
)
network.add_layer(
    inhibit_layer,
    name="inhibit",
)
network.add_layer(
    adaptive_layer, 
    name="adaptive",
)
network.add_layer(
    lif_layer, 
    name="lif",
)

network.add_layer(
    lif_out, 
    name="output",
)
# connection
network.add_connection(
    input_adaptive_conn,
    source="input", 
    target="adaptive",
)
network.add_connection(
    input_lif_conn, 
    source="input", 
    target="lif",
)
network.add_connection(
    input_inhibit_conn,
    source="input",
    target="inhibit"
)
network.add_connection(
    adaptive_recurrent_conn, 
    source="adaptive", 
    target="adaptive",
)
network.add_connection(
    lif_recurrent_conn,
    source="lif",
    target="lif",
)
network.add_connection(
    lif_adaptive_conn,
    source = "lif",
    target = "adaptive",
)
network.add_connection(
    adaptive_lif_conn,
    source = "adaptive",
    target = "lif",
)
network.add_connection(
    inhibit_lif_conn,
    source="inhibit",
    target="lif"
)
network.add_connection(
    lif_inhibit_conn,
    source="lif",
    target="inhibit",
)
network.add_connection(
    inhibit_adaptive_conn,
    source="inhibit",
    target="adaptive",
)
network.add_connection(
    adaptive_inhibit_conn,
    source="adaptive",
    target="inhibit",
)


network.add_connection(
    inhibit_inhibit_conn,
    source="inhibit",
    target="inhibit",
)

network.add_connection(
    adaptive_adaptive_conn,
    source="adaptive",
    target="adaptive",
)

network.add_connection(
    lif_lif_conn,
    source="lif",
    target="lif",
)

network.add_connection(
    lif_lif_conn,
    source="adaptive",
    target="output",
)


network.add_connection(
    lif_lif_conn,
    source="inhibit",
    target="output",
)

network.add_connection(
    lif_lif_conn,
    source="lif",
    target="output",
)
# network.add_connection()

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(
    network.layers["adaptive"], 
    ["v"], 
    time=time
)
network.add_monitor(
    voltage_monitor, 
    name="output_voltage"
)

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

# print(spikes)
# sys.exit()

voltages = {}
for layer in set(network.layers) - {"input"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
input_adaptive_w_im = None
input_adaptive_w_ax = None
input_lif_w_im = None
input_inhibit_w_im = None
weights2_im = None
voltage_ims = None
voltage_axes = None

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

    for step, batch in enumerate(tqdm(train_dataloader)):
        # Get next input sample.
        if step > n_train:
            break
        inputs = {"input": batch["encoded_image"].view(time, batch_size, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Optionally plot various simulation information.
        if plot and batch_size == 1:
            image = batch["image"].view(28, 28)

            inpt = inputs["input"].view(time, 784).sum(0).view(28, 28)

            _spikes = {
                "input": spikes["input"].get("s").view(time, -1),
                "lif": spikes["lif"].get("s").view(time, -1),
                "inhibit": spikes["inhibit"].get("s").view(time, -1),
                "adaptive": spikes["adaptive"].get("s").view(time, -1),
            }
            _voltages = {
                "lif": voltages["lif"].get("v").view(time, -1),
                "inhibit": voltages["inhibit"].get("v").view(time, -1),
                "adaptive": voltages["adaptive"].get("v").view(time, -1)
            }

            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
            )
            input_adaptive_w = input_adaptive_conn.w
            input_lif_w = input_lif_conn.w
            input_inhibit_w = input_inhibit_conn.w

            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            input_adaptive_w_im = plot_conv2d_weights(
                input_adaptive_w, 
                im=input_adaptive_w_im,
            )
            input_lif_w_im = plot_conv2d_weights(
                input_lif_w, 
                im=input_lif_w_im
            )
            input_inhibit_w_im = plot_conv2d_weights(
                input_inhibit_w, 
                im=input_inhibit_w_im
            )
            voltage_ims, voltage_axes = plot_voltages(
                _voltages, ims=voltage_ims, axes=voltage_axes
            )

            plt.pause(1)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")
