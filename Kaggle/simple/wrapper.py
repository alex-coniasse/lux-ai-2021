# wrapper for observation preprocessing - features extraction from the game
import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
from kaggle_environments import make

import torch
import numpy as np

class EnvWrapper:
    def __init__(self, opponent):
        self.game_state = Game()
        self.opponent = opponent
        self.env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 0, "annotations": False}, debug=False)
        self.trainer = self.env.train([None,opponent])
        obs = self.trainer.reset()
        self.update_game_state(obs)
        self.device = "cuda" if (torch.cuda.is_available()) else "cpu"
        self.current_reward = 0


    def update_game_state(self, observation):
        """Update the local game state from observation"""

        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation.player
        else:
            self.game_state._update(observation["updates"])
        self.obs = observation
        self.game_map = self.game_state.map
        self.player = self.game_state.players[observation.player]
        self.opponent = self.game_state.players[(observation.player + 1) % 2]
        self.width, self.height = self.game_map.width, self.game_map.height

        self.resource_tiles: list[Cell] = []
        self.city_tiles: list[Cell] = []
        
    
        for y in range(self.height):
            for x in range(self.width):
                cell = self.game_state.map.get_cell(x, y)
                if cell.has_resource():
                    self.resource_tiles.append(cell)
                if cell.citytile is not None:
                    self.city_tiles.append(cell.citytile)   

        return self.game_state
    
    def get_reward(self):
        """Workaround to get a correct reward"""
        player = self.player
        ct_count = sum([len(v.citytiles) for k, v in player.cities.items()])
        unit_count = len(self.game_state.players[player.team].units)
        score = ct_count * 10000 + unit_count*10
        if score == self.current_reward:
            reward = 0 
        if score > self.current_reward:
            reward = 1
        if score < self.current_reward:
            reward = -1 
        self.current_reward = score

        return reward
    

    def get_cell_resources(self, cell: Cell):
        wood = 0
        coal = 0
        uranium = 0
        if cell.has_resource():
            wood = cell.resource.amount if cell.resource.type == "wood" else 0
            coal = cell.resource.amount if cell.resource.type == 'coal' else 0
            uranium = cell.resource.amount if cell.resource.type == 'uranium' else 0
        return wood, coal, uranium
    
    def get_adjacent_resources(self, pos):
        wood = 0
        coal = 0
        uranium = 0
        x, y = (pos.x, pos.y)
        for i in range(-1,1):
            for j in range(-1,1):
                cell = self.game_map.get_cell(x+i,y+j)
                if cell.has_resource():
                    wood += cell.resource.amount if cell.resource.type == "wood" else 0
                    coal += cell.resource.amount if cell.resource.type == 'coal' else 0
                    uranium += cell.resource.amount if cell.resource.type == 'uranium' else 0
        return wood, coal, uranium
    
    def add_units_state(self, state):
        # player units features
        for unit in self.player.units:
            active = 1 if unit.can_act() else 0.1
            cell = self.game_map.get_cell_by_pos(unit.pos)
            worker_type = 1 if unit.is_worker() else 0.5
            cargo = unit.get_cargo_space_left()
            x = unit.pos.x
            y = unit.pos.y
            is_in_city = cell.citytile != None
            adjacent_wood, adjacent_coal, adjacent_uranium = self.get_adjacent_resources(unit.pos)
            state[-8:-1, x, y] = torch.tensor([worker_type, cargo, adjacent_wood, adjacent_coal, adjacent_uranium, is_in_city, active]).to(self.device)
        # opponent units features
        for unit in self.opponent.units:
            x = unit.pos.x
            y = unit.pos.y
            cell_code = 1
            if unit.is_cart:
                cell_code +=0.5
            state[-8, x, y]= cell_code
        return state

    def get_global_state(self):
        state = torch.zeros(16,32,32, dtype=torch.float32).to(self.device)
        for x in range(self.width):
            for y in range(self.height):
                cell = self.game_map.get_cell(x,y)
                w, c, u = self.get_cell_resources(cell)
                cityid = int(cell.citytile.cityid[2:])+1 if cell.citytile else 0
                city = (cell.citytile.team+1) if cell.citytile else 0
                fuel = (self.opponent.cities[cell.citytile.cityid].fuel if city == self.opponent.team+1
                       else self.player.cities[cell.citytile.cityid].fuel if city == self.player.team+1 else 0)
                road = cell.road
                research = self.player.research_points
                state[:,x,y]= torch.tensor([w, c, u, cityid, fuel, city, road, research, 0, 0, 0, 0, 0, 0, 0, 0]).to(self.device)
        state = self.add_units_state(state)
        return state

    def step(self, action):
        obs, _, done, info = self.trainer.step(action)
        game_state = self.update_game_state(obs)
        reward = self.get_reward()
        global_state = self.get_global_state()
        return global_state, reward, done, info

    def reset(self):
        map_seed = np.random.randint(0,1000000000)
        self.env = make("lux_ai_2021", configuration={"seed": map_seed, "loglevel": 0, "annotations": False}, debug=False)
        self.trainer = self.env.train([None,self.opponent])
        obs = self.trainer.reset()
        self.update_game_state(obs)
        global_state = self.get_global_state()
        print("map size: ", self.height)
        return global_state
    
    def get_game_objects(self):
        return [self.player, self.city_tiles]