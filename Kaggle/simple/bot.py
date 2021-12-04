import random
import torch
import sys
from lux.game import Game
from functools import partial
from collections import deque
from lux.constants import Constants
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from model import DDQN
import numpy as np
from torch.utils.tensorboard import SummaryWriter

DIRECTIONS = Constants.DIRECTIONS
class Bot:
    def __init__(self, save_dir):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        self.action_space = 6
        self.model = DDQN(self.action_space).to(self.device)
        self.memory = deque(maxlen=100000)
        
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e4
        self.batch_size = 16
        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.burnin = 1000 #1e4  # min. experiences before training
        self.learn_every = 2  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        
        self.log_every = 30
        
        self.training = True

        # if self.use_cuda:
        #     self.net = self.net.to(device="cuda")

    def play(self, state, game_objects):
        """Play 1 step with epsilon-greedy policy and return actions"""
        actions = []
        cities_actions = []
        player, city_tiles = game_objects
        global_state_tensor = state
        
        explore = np.random.rand() < self.exploration_rate and self.training

        def get_random_actions(unit):
            actions_list = [unit.move(DIRECTIONS.NORTH), unit.move(DIRECTIONS.SOUTH), unit.move(DIRECTIONS.EAST), 
                            unit.move(DIRECTIONS.WEST), unit.build_city(), unit.pillage()]
            i = np.random.randint(6, size=1)[0]
            
            return actions_list[i], i
        
        def get_city_actions(city):
            if np.random.rand() < 0.5:
                return city.research()
            else:
                return city.build_worker()


        num_citytiles = len([city for city in city_tiles if city.team == player.team])
        num_units = len(player.units)

        masks = torch.zeros(4, 32, 32).to(self.device)
        # 0 - 1 : actions values
        # 2 - 3 : binary mask
                
        # print(player.team)
        if (explore): #!!!!!!!!!!!
        #EXPLORE
            for unit in player.units:
                if unit.is_worker() and unit.can_act():
                    x = unit.pos.x
                    y = unit.pos.y
                    action, idx = get_random_actions(unit)
                    actions.append(action)
                    masks[0, x, y] = idx
                    masks[2, x, y] = 1

            for city in city_tiles:
                if city.team == player.team and city.can_act():
                    x = city.pos.x
                    y = city.pos.y
                    masks[3, x, y] = 1
                    if num_units < num_citytiles and np.random.rand() < 0.55:
                        num_units +=1
                        cities_actions.append(city.build_worker())
                        masks[1, x, y] = 0
                    else:
                        cities_actions.append(city.research())
                        masks[1, x, y] = 1
        else:
        # EXPLOIT
                q_units, q_city = self.model(global_state_tensor.unsqueeze(0), mode="online")
                # select idx with higher Q prediction for each unit
                
                # [channel, height, width]
                action_idx_units = torch.argmax(q_units, axis=1)
                action_idx_city = torch.argmax(q_city, axis=1)

                for unit in player.units:
                    x = unit.pos.x
                    y = unit.pos.y
                    if unit.can_act():
                        actions_list_unit = [unit.move(DIRECTIONS.NORTH), unit.move(DIRECTIONS.SOUTH), unit.move(DIRECTIONS.EAST), 
                                        unit.move(DIRECTIONS.WEST), unit.build_city(), unit.pillage()]
                        action = actions_list_unit[action_idx_units[0, x, y].item()]
                        masks[0, x, y] = action_idx_units[0, x, y]
                        masks[2, x, y] = 1
                        actions.append(action)
                for city in city_tiles:
                    if city.team == player.team and city.can_act():
                        x = city.pos.x
                        y = city.pos.y
                        actions_list_city = [city.build_worker(), city.research()]
                        action = actions_list_city[action_idx_city[0, x, y].item()]
                        masks[1, x, y] = action_idx_city[0, x, y]
                        masks[3, x, y] = 1
                        cities_actions.append(action)

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return actions, cities_actions, masks

    def cache(self, state, next_state, action, reward, done, info, masks):
        global_state = state[0]
        units_state = state[1]
                
        # units_actions = torch.tensor(action[0]).to(self.device)
        # city_actions = torch.tensor(action[1]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).float().to(self.device)

        self.memory.append((state, next_state, masks.long(), reward, done,))
    
    
    def get_batch(self):
        """Sample a batch from the cache"""            
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, reward, done, masks = map(torch.stack, zip(*batch))
        return state, next_state, reward.squeeze(), done.squeeze(), masks
    
    def save(self):
        save_path = (
            self.save_dir + f"LuxRN_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.model.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"LuxRN saved to {save_path} at step {self.curr_step}")

    def load(self,filename):
        saved = torch.load(filename)
        self.model.load_state_dict(saved["model"])
        self.exploration_rate = (saved["exploration_rate"])

    def learn(self):

        def td_estimate(state, masks):
            """Estimate for a sequence of units"""
            # action.register_hook(lambda grad : print(grad))
            Q_units, Q_cities = self.model(state, "online")
            
            td_units = torch.gather(Q_units, 1, masks[:,0].unsqueeze(1)).squeeze(1)
            td_cities = torch.gather(Q_cities, 1, masks[:,1].unsqueeze(1)).squeeze(1)
            # need to retrieve the active units and cities from the 2d grid
            td_units *= masks[:,2]
            td_cities *= masks[:,3]
            
            return [td_units, td_cities]

        @torch.no_grad()
        def td_target(reward, next_state, done, masks):
            """Target for a sequence of units"""
            next_state_Q_units, next_state_Q_cities = self.model(next_state, "online")
            # need to mask se sequence here?

            best_actions_units = torch.argmax(next_state_Q_units, axis=1)
            best_actions_cities = torch.argmax(next_state_Q_cities, axis=1)

            next_Q_units, next_Q_cities = self.model(next_state, "target")
            
            next_best_Q_units = torch.gather(next_Q_units, 1, best_actions_units.unsqueeze(1)).squeeze(1)
            next_best_Q_cities = torch.gather(next_Q_cities, 1, best_actions_cities.unsqueeze(1)).squeeze(1)
            
            reward = reward[:, None, None].expand([16, 32, 32])
            done = done[:, None].expand([16, 32, 32])
            units_target = (reward + (1 - done) * self.gamma * next_best_Q_units).float()
            cities_target = (reward + (1 - done) * self.gamma * next_best_Q_cities).float()
            units_target*= masks[:,2]
            cities_target*= masks[:,3]
            
            return [units_target, cities_target]
        
        @torch.enable_grad()
        def update_Q_online(td_estimate, td_target):
            losses = torch.stack([self.loss_fn(td_e, td_t) for (td_e, td_t) in zip(td_estimate, td_target)])
            # losses.register_hook(lambda grad : print(grad))
            loss = torch.sum(losses)*10
            # loss.register_hook(lambda grad : print(grad))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print (self.model.online.rnn.weight_ih_l0.grad)
            return loss.item()

        def sync_Q_target():
            self.model.target.load_state_dict(self.model.online.state_dict())
            
            
        if self.curr_step % self.sync_every == 0:
            sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step == self.burnin:
            print("Learning started")
        

        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # Sample from memory
        state, next_state, masks, reward, done = self.get_batch()
        # action = [city_actions, units_actions]
        # Get TD Estimate
        td_est = td_estimate(state, masks)

        # Get TD Target
        td_tgt = td_target(reward, next_state, done, masks)
        # print(td_tgt[0].shape)
        # print(td_est[0].shape)

        # Backpropagate loss through Q_online
        loss = update_Q_online(td_est, td_tgt)
        

        # return (td_est.mean().item(), loss)
        return loss, None

    def simple_logging(self, to_log):
        # if self.curr_step % self.log_every == 0:
        for (k,v) in to_log.items():
            print(k, v)
            sys.stdout.flush()
    
    def log_tensorboard(self, writer):
        if self.curr_step > self.burnin:
            
            writer.add_histogram("online_conv1", self.model.online.conv1.weight , self.curr_step)
            writer.add_histogram("online_conv2", self.model.online.conv2.weight, self.curr_step)
            writer.add_histogram("online_conv3", self.model.online.conv3.weight , self.curr_step)
            writer.add_histogram("online_out_unit", self.model.online.out_unit.weight , self.curr_step)
            writer.add_histogram("online_out_city", self.model.online.out_city.weight, self.curr_step)
            
            writer.add_scalar("Exploration_rate", self.exploration_rate, self.curr_step)
            
            #Grads
            writer.add_histogram("online_conv1_grad", self.model.online.conv1.weight.grad , self.curr_step)
            writer.add_histogram("online_conv2_grad", self.model.online.conv2.weight.grad, self.curr_step)
            writer.add_histogram("online_conv3_grad", self.model.online.conv3.weight.grad , self.curr_step)
            writer.add_histogram("online_out_unit_grad", self.model.online.out_unit.weight.grad , self.curr_step)
            writer.add_histogram("online_out_city_grad", self.model.online.out_city.weight.grad, self.curr_step)