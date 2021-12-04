from kaggle_environments import make
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from bot import Bot
from wrapper import EnvWrapper

from agent import agent
import argparse



torch.set_printoptions(threshold=5000)
episodes = 17000
max_reward = 1000000


if __name__ == "__main__":
    
    device = "cuda" if (torch.cuda.is_available()) else "cpu"

    # arg parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-ckpt',action='store', type=str,
                    help='The latest checkpoint')
    args = parser.parse_args()
    
    print("Using device: ", device)

    # make deterministic runs
    torch.manual_seed(43)
    np.random.seed(43)

    bot = Bot( "./ckpts/")
    if args.load_ckpt:
        print("loading checkpoint:", args.load_ckpt)
        bot.load(args.load_ckpt)

    env = EnvWrapper(agent)
    writer = SummaryWriter()

    def unit_action_string_to_idx(action):
        # print(action)
        offsets = {"n":0, "s":1, "e":2, "w":3, "b":4, "p":5}
        if action[0] == "m":
            return offsets[action[-1]]
        else:
            return offsets[action[0]]
    
    def city_action_string_to_idx(action):
        if action[0] == "b": #bw
            return 0
        else:
            return 1 # research

    best_episodes = []

    for e in range (episodes):
        state = env.reset()
        game_objects = env.get_game_objects()
        episode_reward = 0
        t0 = time.time()
        
        while True:
            # Run agent on the state
            # Cities action are not learnable yet and must be separated
            action, cities_actions, masks = bot.play(state, game_objects)
            
            # Agent performs action
            next_state, reward, done, info = env.step([*action, *cities_actions])
            game_objects = env.get_game_objects()
            writer.add_scalar('reward', reward, bot.curr_step)

            
            # Remember if action if not empty
            if len(action) + len(cities_actions ) > 0:
                action = [unit_action_string_to_idx(a) for a in action]
                cities_actions = [city_action_string_to_idx(a) for a in cities_actions]
                bot.cache(state, next_state, [action, cities_actions], reward, done, info, masks)

            # Learn
            loss, _ = bot.learn()
            
            episode_reward += reward

            # Checkpoint
            if bot.curr_step == 1 or bot.curr_step % bot.save_every == 0:
                bot.save()

            # Update state
            state = next_state
            if done:
                total_time = time.time() - t0
                bot.log_tensorboard(writer)
                writer.add_scalar('Episode_reward', episode_reward, bot.curr_step)
                writer.add_scalar('Episode_lengh', env.obs["step"], bot.curr_step)
                print ("Throughput: ", env.obs["step"]/ total_time)
                bot.simple_logging({"episode":e, "reward": episode_reward})
                # save best episodes for later
                if episode_reward > max_reward:
                    max_reward = episode_reward
                    best_episodes.append(env.env.toJSON())
                    print(len(best_episodes))
                break
        writer.close()


