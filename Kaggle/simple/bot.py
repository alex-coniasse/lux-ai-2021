import random
import sys
from lux.game import Game
from functools import partial
from collections import deque
from lux.constants import Constants
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
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
        self.save_every = 1e5
        self.batch_size = 16
        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.burnin = 10000 #1e4  # min. experiences before training
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
        global_state_tensor = state[0].unsqueeze(0)
        units_state_tensor = state[1].unsqueeze(0)
        explore = np.random.rand() < self.exploration_rate and self.training

        def get_random_actions(unit):
            actions_list = [unit.move(DIRECTIONS.NORTH), unit.move(DIRECTIONS.SOUTH), unit.move(DIRECTIONS.EAST), 
                            unit.move(DIRECTIONS.WEST), unit.build_city(), unit.pillage()]
            i = np.random.randint(6, size=1)[0]
            return actions_list[i]

        num_citytiles = len([city for city in city_tiles if city.team == player.team])
        num_units = len(player.units)
        for city in city_tiles:
            # TODO: hardcoded city policy here
            if city.team == player.team and city.can_act():
                if num_units < num_citytiles:
                    num_units +=1
                    cities_actions.append(city.build_worker())
                else:
                    cities_actions.append(city.research())
                    

        # print(player.team)
        if (explore):
        #EXPLORE
            for unit in player.units:
                if unit.is_worker() and unit.can_act():
                    actions.append(get_random_actions(unit))
        else:
        # EXPLOIT
            if len(units_state_tensor) > 1:
                action_qvalues = self.model([global_state_tensor, units_state_tensor], mode="online")
                # print(action_qvalues)
                # select idx with higher Q prediction for each unit
                action_idx = [torch.argmax(Q, axis=1).item() for Q in action_qvalues]

                idx=0
                for unit in player.units:
                    if unit.can_act():
                        actions_list = [unit.move(DIRECTIONS.NORTH), unit.move(DIRECTIONS.SOUTH), unit.move(DIRECTIONS.EAST), 
                                unit.move(DIRECTIONS.WEST), unit.build_city(), unit.pillage()]
                        action = actions_list[action_idx[idx]]
                        actions.append(action)
                        idx+=1

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return actions, cities_actions

    def cache(self, state, next_state, action, reward, done, info):
        global_state = state[0]
        units_state = state[1]
        
        next_global_state = next_state[0]
        next_unit_state = next_state[1]
        
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        done = torch.tensor([done]).to(self.device)

        self.memory.append((global_state, units_state, next_global_state, next_unit_state, action, reward, done,))
    
    
    def get_batch(self):
        """Sample a batch from the cache"""
                       
        batch = random.sample(self.memory, self.batch_size)
        batch_constant = [(state[0], state[2], state[5], state[6]) for state in batch] # constant shape: global_state, next_global_state, reward, done
        
        batch_sequences = [(state[1], state[3], state[4]) for state in batch] # variable shape: units_state, next_units_state, action
        seq_lenghts = [(seq[0].shape[0], seq[1].shape[0], seq[2].shape[0]) for seq in batch_sequences]
        
        global_state, next_global_state, reward, done = map(torch.stack, zip(*batch_constant))
        unit_state, next_unit_state, action = map(lambda x: pad_sequence(x, batch_first=True), zip(*batch_sequences))
        
        return global_state, unit_state, next_global_state, next_unit_state, action, reward.squeeze(), done.squeeze(), seq_lenghts
    
    def save(self):
        save_path = (
            self.save_dir + f"LuxRN_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.model.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"LuxRN saved to {save_path} at step {self.curr_step}")

    def learn(self):

        def td_estimate(state, actions, seq_lens):
            """Estimate for a sequence of units"""
            # action.register_hook(lambda grad : print(grad))
            Q_sequence = self.model(state, "online", seq_lens)
            # Q_sequence[0].register_hook(lambda grad : print("gradient:", grad))
            max_len = actions.shape[1]
            actions = torch.chunk(action, max_len, dim=1)
            current_Q_sequence = [torch.gather(current_Q, 1, action) for (action, current_Q) in zip(actions, Q_sequence)]
            return current_Q_sequence

        @torch.no_grad()
        def td_target(reward, next_state, done, seq_lens):
            """Target for a sequence of units"""
            next_state_Q_sequence = self.model(next_state, "online", seq_lens)
            # need to mask se sequence here?

            
            best_actions = [torch.argmax(next_state_Q, axis=1).unsqueeze(1) for next_state_Q in next_state_Q_sequence]
            next_Q_sequence = self.model(next_state, "target", seq_lens)
            next_best_Q_sequence = [torch.gather(current_Q, 1, action) for (action, current_Q) in zip(best_actions, next_Q_sequence)]

            return [(reward.unsqueeze(1) + (1 - done.unsqueeze(1).float()) * self.gamma * next_Q).float() for next_Q in next_best_Q_sequence]
        
        @torch.enable_grad()
        def update_Q_online(td_estimate, td_target):
            losses = torch.stack([self.loss_fn(td_e, td_t) for (td_e, td_t) in zip(td_estimate, td_target)])
            # losses.register_hook(lambda grad : print(grad))
            loss = torch.mean(losses)
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
        global_state, units_state, next_global_state, next_unit_state, action, reward, done, seq_lenghts = self.get_batch()
        state = [global_state, units_state]
        next_state = [next_global_state, next_unit_state]
        
        state_seq_lenghts = [sl[0] for sl in seq_lenghts]
        next_state_seq_lenghts = [sl[1] for sl in seq_lenghts]
        
        # Get TD Estimate
        td_est = td_estimate(state, action, state_seq_lenghts)

        # Get TD Target
        td_tgt = td_target(reward, next_state, done, next_state_seq_lenghts)
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
            
            writer.add_histogram("online_GRU_0_ih", self.model.online.rnn.weight_ih_l0 , self.curr_step)
            writer.add_histogram("online_GRU_0_ih_bias", self.model.online.rnn.bias_ih_l0 , self.curr_step)
            writer.add_histogram("online_GRU_0_hh", self.model.online.rnn.weight_hh_l0 , self.curr_step)
            writer.add_histogram("online_GRU_0_hh_bias", self.model.online.rnn.bias_hh_l0 , self.curr_step)
            
            # writer.add_histogram("online_GRU_1_ih", self.model.online.rnn.weight_ih_l1 , self.curr_step)
            # writer.add_histogram("online_GRU_1_ih_bias", self.model.online.rnn.bias_ih_l1 , self.curr_step)
            # writer.add_histogram("online_GRU_1_hh", self.model.online.rnn.weight_hh_l1, self.curr_step)
            # writer.add_histogram("online_GRU_1_hh_bias", self.model.online.rnn.bias_hh_l1 , self.curr_step)
            
            writer.add_histogram("online_map_conv1", self.model.online.mapEncoder.conv1.weight, self.curr_step)
            writer.add_histogram("online_map_conv2", self.model.online.mapEncoder.conv2.weight, self.curr_step)
            writer.add_histogram("online_map_conv1_bias", self.model.online.mapEncoder.conv1.bias, self.curr_step)
            writer.add_histogram("online_map_conv2_bias", self.model.online.mapEncoder.conv2.bias, self.curr_step)
            
            writer.add_histogram("online_map_ffn", self.model.online.mapEncoder.ffn_map.weight, self.curr_step)
            writer.add_histogram("online_map_ffn_bias", self.model.online.mapEncoder.ffn_map.bias, self.curr_step)
            
            writer.add_histogram("online_joiner", self.model.online.joiner.weight, self.curr_step)
            writer.add_histogram("online_joiner_bias", self.model.online.joiner.bias, self.curr_step)
            
            writer.add_histogram("online_output", self.model.online.output.weight, self.curr_step)
            writer.add_histogram("online_output_bias", self.model.online.output.bias, self.curr_step)
            
            writer.add_scalar("Exploration_rate", self.exploration_rate, self.curr_step)
            
            #Grads
            
            writer.add_histogram("online_GRU_0_ih_grad", self.model.online.rnn.weight_ih_l0.grad , self.curr_step)
            writer.add_histogram("online_GRU_0_ih_bias_grad", self.model.online.rnn.bias_ih_l0.grad , self.curr_step)
            writer.add_histogram("online_GRU_0_hh_grad", self.model.online.rnn.weight_hh_l0.grad , self.curr_step)
            writer.add_histogram("online_GRU_0_hh_bias_grad", self.model.online.rnn.bias_hh_l0.grad , self.curr_step)
            
            writer.add_histogram("online_map_conv1_grad", self.model.online.mapEncoder.conv1.weight.grad, self.curr_step)
            writer.add_histogram("online_map_conv2_grad", self.model.online.mapEncoder.conv2.weight.grad, self.curr_step)
            writer.add_histogram("online_map_conv1_bias_grad", self.model.online.mapEncoder.conv1.bias.grad, self.curr_step)
            writer.add_histogram("online_map_conv2_bias_grad", self.model.online.mapEncoder.conv2.bias.grad, self.curr_step)
            
            writer.add_histogram("online_map_ffn_grad", self.model.online.mapEncoder.ffn_map.weight.grad, self.curr_step)
            writer.add_histogram("online_map_ffn_bias_grad", self.model.online.mapEncoder.ffn_map.bias.grad, self.curr_step)
            
            writer.add_histogram("online_joiner_grad", self.model.online.joiner.weight.grad, self.curr_step)
            writer.add_histogram("online_joiner_bias_grad", self.model.online.joiner.bias.grad, self.curr_step)
            
            writer.add_histogram("online_output_grad", self.model.online.output.weight.grad, self.curr_step)
            writer.add_histogram("online_output_bia_grads", self.model.online.output.bias.grad, self.curr_step)