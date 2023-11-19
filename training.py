import copy
import torch

from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from environment import create_environment
from experience_replay_buffer import ReplayBuffer
from noisy_dqn_model import DQN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class RLDataset(IterableDataset):

  def __init__(self, buffer, sample_size=400):
    self.buffer = buffer
    self.sample_size = sample_size

  def __iter__(self):
    for experience in self.buffer.sample(self.sample_size):
      yield experience

def greedy(state, net, support):
  state = torch.tensor([state]).to(device)
  q_value_probs = net(state) #(1,a,N)

  #Q(s,a) = support * probabilities of q_values
  q_values = (support * q_value_probs).sum(dim=-1) #(1,A)

  #picks the action with the highest q value
  action = torch.argmax(q_values, dim=-1) # (1,1)
  action = int(action.item())# item gives the action
  return action

class DeepQLearning(LightningModule):
  def __init__(self, env_name, policy=greedy, capacity=100_000,
               batch_size=256, lr=1e-3, hidden_size=128, gamma= 0.99,
               loss_fn = F.smooth_l1_loss, optim=AdamW,
               samples_per_epoch = 1_000,sync_rate=10, a_start=0.5,
               a_end = 0.0, a_last_episode=100,
               b_start=0.4, b_end=1.0, b_last_episode=100, sigma=0.5,
               n_steps=3, v_min=-10.0, v_max=10.0, atoms=51):

    super().__init__()

    self.support = torch.linspace(v_min, v_max, atoms, device=device) #(N) a vector [-10,....,10], atoms number of bins
    self.delta = (v_max - v_min) / (atoms-1)



    self.env = create_environment(env_name)

    obs_size = self.env.observation_space.shape
    n_actions = self.env.action_space.n

    self.q_net = DQN(hidden_size, obs_size, n_actions, sigma=sigma, atoms=atoms) # q network

    self.target_q_net = copy.deepcopy(self.q_net) #target q network

    self.policy = policy
    self.buffer = ReplayBuffer(capacity=capacity)

    self.save_hyperparameters() #saves hyperparameters

    while len(self.buffer) < self.hparams.samples_per_epoch:
      print(f"{len(self.buffer)} samples in experience buffer. Filling...")
      self.play_episode()

  @torch.no_grad()
  def play_episode(self, policy=None):
    state = self.env.reset()
    done = False
    transitions = []

    while not done:
      if policy:
        action = policy(state, self.q_net, self.support)
      else:
        action = self.env.action_space.sample()
      next_state, reward, done, info = self.env.step(action)
      exp = (state, action, reward, done, next_state)
      transitions.append(exp)
      state = next_state

    for i, (s,a,r,d,ns) in enumerate(transitions):
        batch = transitions[i: i + self.hparams.n_steps]
        # r + gamma * r2 + gamma^2 * r3
        #t[2] = transitions[2] + n_steps of reward trajectory = reward
        # j=0, j=1, j=2 because it is the index
        ret = sum([t[2] * self.hparams.gamma**j for j, t in enumerate(batch)])
        #last done and last state
        _,_,_, ld, ls = batch[-1]
        #state, action, return, last done and last state in the buffer
        self.buffer.append((s,a,ret,ld,ls))

      #forward
  def forward(self, x):
    return self.q_net(x)

  # configure optimizers
  def configure_optimizers(self):
    q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
    return [q_net_optimizer]

  #create dataloader
  def train_dataloader(self):
    dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
    dataloader = DataLoader(
        dataset= dataset,
        batch_size= self.hparams.batch_size
    )
    return dataloader

   #training step
  def training_step(self, batch, batch_idx):
    indices, weights, states, actions, returns, dones, next_states = batch
    returns = returns.unsqueeze(1)
    dones = dones.unsqueeze(1)
    batch_size = len(indices)

    q_value_probs = self.q_net(states) # (B,A,N) = Batch row, action column and atoms depth
    #action_value_probs = (B,N)
    action_value_probs = q_value_probs[range(batch_size), actions, :] #each action in each row and the elements related to it
    log_action_value_probs = torch.log(action_value_probs + 1e-6)#(B,N)


    with torch.no_grad():
      next_q_value_probs = self.q_net(next_states) # (B,A,N)
      next_q_values = (next_q_value_probs * self.support).sum(dim=-1) #(Batch, Actions)
      next_actions = next_q_values.argmax(dim=-1) #(B,)

      next_q_value_probs = self.target_q_net(next_states) # (B,A,N)
      next_action_value_probs = next_q_value_probs[range(batch_size), next_actions, :] #(B,N)


    m = torch.zeros(batch_size * self.hparams.atoms, device=device, dtype = torch.float64) #(B*N)

    '''tz <- [Rt + Y^n * zj]'''
    Tz = returns + ~dones * self.hparams.gamma **self.hparams.n_steps * self.support.unsqueeze(0) # (B,1)*(1, N)-> (B,N)
    '''tz clamp(vmax, vmin)'''
    Tz.clamp_(min=self.hparams.v_min, max=self.hparams.v_max) #(B,N)
    #now they are in between [-10:10]

    '''bj<- (Tzj - Vmin)/ Zdelta'''
    b = (Tz-self.hparams.v_min)/ self.delta #(B,N)

    '''low <-[bj], up<-[bj]  ex: if bj = 2.5, low = 2, up = 3'''
    l,u = b.floor().long(), b.ceil().long() #(B,N)

    '''tensor of many rows as batch_elements and each row goes up from 0,1,2,...'''
    offset = torch.arange(batch_size, device=device).view(-1,1) * self.hparams.atoms #(B,1)

    '''the elements will be 0+l, 1N+l, 2N + l, 3N +l'''
    l_idx = (l + offset).flatten() # (B * N)
    u_idx = (u + offset).flatten() # (B * N)

    #mi <- mi +pj(xt+1, a)(u-bj)
    upper_probs = (next_action_value_probs * (u-b)).flatten() #(B*N)

    #mu <- mu +pj(xt+1, a)(bj-l)
    lower_probs = (next_action_value_probs * (b-l)).flatten() #(B*N)

    #mi <- mi +pj(xt+1, a)(u-bj)
    m.index_add_(dim=0, index=l_idx, source=upper_probs)

    #mu <- mu +pj(xt+1, a)(bj-l)
    m.index_add_(dim=0, index=u_idx, source=lower_probs)

    m = m.reshape(batch_size, self.hparams.atoms) #(B, N)

    '''-E(mi * log pi(St,At))'''
    cross_entropies = -(m * log_action_value_probs).sum(dim=-1) # (B,N) -> (B,)

    for idx, e in zip(indices, cross_entropies):
      self.buffer.update(idx, e.detach().item())

    loss = (weights * cross_entropies).mean()

    self.log("episode/Q-Error", loss)
    return loss

  #training epoch end
  def training_epoch_end(self, training_step_outputs):

    alpha = max(
        self.hparams.a_end,
        self.hparams.a_start - self.current_epoch / self.hparams.a_last_episode
    )
    beta = min(
        self.hparams.b_end,
        self.hparams.b_start + self.current_epoch / self.hparams.b_last_episode
    )

    self.buffer.alpha = alpha
    self.buffer.beta = beta

    self.play_episode(policy=self.policy)
    self.log("episode/Return", self.env.return_queue[-1]) #last episode play by agent

    if self.current_epoch % self.hparams.sync_rate == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())