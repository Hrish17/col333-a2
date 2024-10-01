import time
import math
import random
import numpy as np
import copy
from helper import *

C = 2

class Node:
    def __init__(self, state, playerID, parent=None):
        self.state = np.copy(state)
        self.parent = parent
        self.id = playerID
        self.valid_actions = []
        self.childrens = []
        self.visits = 0
        self.ucb = np.inf
        self.value = 0
        self.action = None
        

class MCTStree:
    def __init__(self, state, playerID, timer):
        self.root = Node(state, playerID)
        self.root.parent = None
        self.UCB=0
        self.player_number=playerID
        self.timer = timer

    def UCB1(self, node):
        # print('printing ucb')
        if node.visits == 0:
            return np.inf
        # elif node.parent is None or node.parent.visits == 0 or len(node.childrens) == 0:
        #     return node.value
        else:
            return (node.value/node.visits) + C * math.sqrt(math.log(node.parent.visits) / node.visits)
        
    def Rollout(self, node):
        # curr_state = copy.deepcopy(node.state)
        # curr_id = node.id
        curr_node = Node(node.state, node.id)
        curr_node.action = node.action
        curr_node.valid_actions = list(get_valid_actions(curr_node.state))

        while True:
            res, type = check_win(curr_node.state, curr_node.action, 3 - curr_node.id, [])
            if res:
                if curr_node.id == self.player_number:
                    return -1
                else:
                    return 1
            if not curr_node.valid_actions:
                return fetch_remaining_time(self.timer, self.player_number)/fetch_remaining_time(self.timer, 3-self.player_number)
            # print('valid actions:')
            rand_action = random.choice(curr_node.valid_actions)
            curr_node.state[rand_action] = curr_node.id
            curr_node.valid_actions.remove(rand_action)
            old_id = curr_node.id
            curr_node.id = 3 - old_id
            curr_node.action = rand_action




    def BackPropagating(self, curr_node, v):
        # print('backprop.')
        while curr_node is not None:
            # print('starting while')
            if curr_node.id == self.player_number:
                curr_node.value -= v
            else:
                curr_node.value += v
            curr_node.visits += 1
            # curr_node.ucb = self.UCB1(curr_node)
            curr_node = curr_node.parent

    def MCTS(self):
        # print('yaha to aaya')
        self.root.valid_actions = list(get_valid_actions(self.root.state))
        for action in self.root.valid_actions:
            new_node = Node(self.root.state, 3 - self.root.id, self.root)
            new_node.state[action] = self.root.id
            new_node.action = action
            new_node.valid_actions = self.root.valid_actions.copy()
            new_node.valid_actions.remove(action)
            self.root.childrens.append(new_node)
            res, type = check_win(new_node.state, action, self.root.id, [])
            if res:
                return action
            
        for action in self.root.valid_actions:
            opp_state = np.copy(self.root.state)
            opp_state[action] = 3 - self.root.id
            res, type = check_win(opp_state, action, 3 - self.root.id, [])
            if res:
                return action

        # print('kya yaha aaya')
            
        start_time = time.time()
        iterations = 0
        while True:
            iterations += 1
            if time.time() - start_time > 10:
                break
            curr_node = self.root
            while curr_node.childrens:
                for child in curr_node.childrens:
                    child.ucb = self.UCB1(child)
                curr_node = max(curr_node.childrens, key=lambda x: x.ucb)
            if curr_node.visits == 0:
                v = self.Rollout(curr_node)
            else:
                # curr_node.valid_actions = copy.deepcopy(curr_node.parent.valid_actions)
                # curr_node.valid_actions.remove(curr_node.action)
                if not curr_node.valid_actions:
                    v = fetch_remaining_time(self.timer, self.player_number)/fetch_remaining_time(self.timer, 3-self.player_number)
                else:
                    for action in curr_node.valid_actions:
                        new_node = Node(curr_node.state, 3 - curr_node.id, curr_node)
                        new_node.state[action] = curr_node.id
                        new_node.action = action
                        new_node.valid_actions = curr_node.valid_actions.copy()
                        new_node.valid_actions.remove(action)
                        curr_node.childrens.append(new_node)
                        # res, type = check_win(new_node.state, action, curr_node.id, [])
                        # if res:
                        #     return action
                        # opp_state = copy.deepcopy(curr_node.state)
                        # opp_state[action] = 3 - curr_node.id
                        # res, type = check_win(opp_state, action, 3 - curr_node.id, [])
                        # if res:
                        #     return action
                    rand_child = random.choice(curr_node.childrens)
                    v = self.Rollout(rand_child)
                    curr_node = rand_child
                    
            # print('yaha bhi')
            self.BackPropagating(curr_node, v)
        print(iterations)
        return max(self.root.childrens, key=lambda x: x.visits).action
    
class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer

    
    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move

        # Parameters
        `state: Tuple[np.array]`
            - a numpy array containing the state of the board using the following encoding:
            - the board maintains its same two dimensions
            - spaces that are unoccupied are marked as 0
            - spaces that are blocked are marked as 3
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them

        # Returns
        Tuple[int, int]: action (coordinates of a board cell)
        """
        mcts = MCTStree(state, self.player_number, self.timer)
        move = mcts.MCTS()
        return move
