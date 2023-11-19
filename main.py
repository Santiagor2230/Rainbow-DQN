from trainings import DeepQLearning
import torch
from pytorch_lightning import Trainer
from display import display_video

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

model = DeepQLearning(
    'QbertNoFrameskip-v4',
    lr=0.0001,
    sigma = 0.5,
    hidden_size = 512,
    a_last_episode = 4_000,
    b_last_episode=4_000,
    n_steps=5,
    samples_per_epoch = 1_000
)

trainer = Trainer(
    gpus=num_gpus,
    max_epochs = 10_000,
    log_every_n_steps = 1
)

trainer.fit(model)