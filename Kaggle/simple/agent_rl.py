import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from model import LuxrNet
import torch
import numpy as np

DIRECTIONS = Constants.DIRECTIONS
game_state = None


def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    model = LuxrNet()
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    resource_tiles: list[Cell] = []
    city_tiles: list[Cell] = []

    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
            if cell.citytile is not None:
                city_tiles.append(cell)
    
    
    def cell_is_free(cell):
        return cell.citytile is None and not cell.has_resource()
    
    for citytile in city_tiles:
        # city policy here
        pass
    
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            # EXPLORE
            if (np.random.rand() < configuration.exploration_rate) and configuration.mode == "training":
                actions = get_random_actions()
            else:
            # EXPLOIT
                state_tensor = None
                action = model(state_tensor, mode="online")
                actions.append(action)
            
            

    
    return actions

