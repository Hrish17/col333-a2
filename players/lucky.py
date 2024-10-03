import time
import math
import random
import numpy as np
import copy
from helper import *

C = 1.4
opp_type = "random"

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
        self.filled = 0

    def UCB1(self, node):
        if node.visits == 0:
            return np.inf
        else:
            return (node.value/node.visits) + C * math.sqrt(math.log(node.parent.visits) / node.visits)
        
    def Rollout(self, node):
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
            rand_action = random.choice(curr_node.valid_actions)
            curr_node.state[rand_action] = curr_node.id
            curr_node.valid_actions.remove(rand_action)
            old_id = curr_node.id
            curr_node.id = 3 - old_id
            curr_node.action = rand_action


    def BackPropagating(self, curr_node, v):
        while curr_node is not None:
            if curr_node.id == self.player_number:
                curr_node.value -= v
            else:
                curr_node.value += v
            curr_node.visits += 1
            curr_node = curr_node.parent

    def MCTS(self):
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
            
        start_time = time.time()
        iterations = 0
        while True:
            if opp_type == "random":
                if self.filled < 10:
                    if time.time() - start_time > 10:
                        break
                elif self.filled < 30:
                    if time.time() - start_time > 12:
                        break
                else:
                    if time.time() - start_time > 5:
                        break
            else:
                if self.filled < 10:
                    if time.time() - start_time > 20:
                        break
                elif self.filled < 30:
                    if time.time() - start_time > 22:
                        break
                else:
                    if time.time() - start_time > 10:
                        break
            iterations += 1
            curr_node = self.root
            while curr_node.childrens:
                for child in curr_node.childrens:
                    child.ucb = self.UCB1(child)
                curr_node = max(curr_node.childrens, key=lambda x: x.ucb)
            if curr_node.visits == 0:
                v = self.Rollout(curr_node)
            else:
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
                    rand_child = random.choice(curr_node.childrens)
                    v = self.Rollout(rand_child)
                    curr_node = rand_child
                    
            # print('yaha bhi')
            self.BackPropagating(curr_node, v)
        print(iterations)
        return max(self.root.childrens, key=lambda x: x.visits).action
    
class MCTS6tree:
    def __init__(self, state, playerID, timer):
        self.root = Node(state, playerID)
        self.root.parent = None
        self.UCB=0
        self.player_number=playerID
        self.timer = timer
        self.filled = 0

    def UCB1(self, node):
        if node.visits == 0:
            return np.inf
        else:
            return (node.value/node.visits) + C * math.sqrt(math.log(node.parent.visits) / node.visits)
        
    def IfNeighbour(self, state, action):
        i = action[0]
        j = action[1]
        neighbours = []
        neighbours.append((i-1,j))
        neighbours.append((i+1,j))
        neighbours.append((i,j-1))
        neighbours.append((i,j+1))
        if j == 5:
            neighbours.append((i-1,j-1))
            neighbours.append((i-1,j+1))

            neighbours.append((i-1,j-2))
            neighbours.append((i-1,j+2))
            neighbours.append((i-2,j-1))
            neighbours.append((i-2,j+1))
            neighbours.append((i+1,j-1))
            neighbours.append((i+1,j+1))
        elif j == 4:
            neighbours.append((i-1,j-1))
            neighbours.append((i+1,j+1))

            neighbours.append((i-2,j-1))
            neighbours.append((i-1,j+1))
            neighbours.append((i,j+2))
            neighbours.append((i+2,j+1))
            neighbours.append((i+1,j-1))
            neighbours.append((i-1,j-2))
        elif j == 6:
            neighbours.append((i+1,j-1))
            neighbours.append((i-1,j+1))
            
            neighbours.append((i-2,j+1))
            neighbours.append((i-1,j+1))
            neighbours.append((i+1,j+1))
            neighbours.append((i+2,j-1))
            neighbours.append((i,j-2))
            neighbours.append((i-1,j-1))
        elif j < 4:
            neighbours.append((i-1,j-1))
            neighbours.append((i+1,j+1))

            neighbours.append((i-1,j+1))
            neighbours.append((i+1,j+2))
            neighbours.append((i+2,j+1))
            neighbours.append((i+1,j-1))
            neighbours.append((i-1,j-2))
            neighbours.append((i-2,j-1))
        else:
            neighbours.append((i+1,j-1))
            neighbours.append((i-1,j+1))

            neighbours.append((i-2,j+1))
            neighbours.append((i-1,j+2))
            neighbours.append((i+1,j+1))
            neighbours.append((i+2,j-1))
            neighbours.append((i+1,j-2))
            neighbours.append((i-1,j-1))
                    
        for neighbour in neighbours:            
            ni = neighbour[0]
            nj = neighbour[1]
            if ni >= 0 and  ni < state.shape[0] and nj >= 0 and nj < state.shape[1]:
                if state[neighbour] == 1 or state[neighbour] == 2:
                    return True

        return False
        
    def Rollout(self, node):
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
            rand_action = random.choice(curr_node.valid_actions)
            curr_node.state[rand_action] = curr_node.id
            curr_node.valid_actions.remove(rand_action)
            old_id = curr_node.id
            curr_node.id = 3 - old_id
            curr_node.action = rand_action


    def BackPropagating(self, curr_node, v):
        while curr_node is not None:
            if curr_node.id == self.player_number:
                curr_node.value -= v
            else:
                curr_node.value += v
            curr_node.visits += 1
            curr_node = curr_node.parent
        
    def MCTS6(self):
        self.root.valid_actions = list(get_valid_actions(self.root.state))
        for action in self.root.valid_actions:
            if self.IfNeighbour(self.root.state, action) and self.filled < 40:
                new_node = Node(self.root.state, 3 - self.root.id, self.root)
                new_node.state[action] = self.root.id
                new_node.action = action
                new_node.valid_actions = self.root.valid_actions.copy()
                new_node.valid_actions.remove(action)
                self.root.childrens.append(new_node)
                res, type = check_win(new_node.state, action, self.root.id, [])
                if res:
                    return action
            elif self.filled >= 40:
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
            
        start_time = time.time()
        iterations = 0
        while True:
            if self.filled < 10:
                if time.time() - start_time > 12:
                    break
            elif self.filled < 40:
                if time.time() - start_time > 13:
                    break
            elif self.filled < 70:
                if time.time() - start_time > 12:
                    break
            else:
                if time.time() - start_time > 5:
                    break
            iterations += 1
            curr_node = self.root
            while curr_node.childrens:
                for child in curr_node.childrens:
                    child.ucb = self.UCB1(child)
                curr_node = max(curr_node.childrens, key=lambda x: x.ucb)

            if curr_node.visits == 0:
                v = self.Rollout(curr_node)
            else:
                if not curr_node.valid_actions:
                    v = fetch_remaining_time(self.timer, self.player_number)/fetch_remaining_time(self.timer, 3-self.player_number)
                else:
                    for action in curr_node.valid_actions:
                        if self.IfNeighbour(curr_node.state, action) and self.filled < 40:
                            new_node = Node(curr_node.state, 3 - curr_node.id, curr_node)
                            new_node.state[action] = curr_node.id
                            new_node.action = action
                            new_node.valid_actions = curr_node.valid_actions.copy()
                            new_node.valid_actions.remove(action)
                            curr_node.childrens.append(new_node)
                        elif self.filled >= 40:
                            new_node = Node(curr_node.state, 3 - curr_node.id, curr_node)
                            new_node.state[action] = curr_node.id
                            new_node.action = action
                            new_node.valid_actions = curr_node.valid_actions.copy()
                            new_node.valid_actions.remove(action)
                            curr_node.childrens.append(new_node)
                    rand_child = random.choice(curr_node.childrens)
                    v = self.Rollout(rand_child)
                    curr_node = rand_child

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
        if fetch_remaining_time(self.timer, self.player_number) > 400:
            opp_type = "student"
        elif fetch_remaining_time(self.timer, self.player_number) > 300:
            opp_type = "ta"

    
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
        if state.shape[0] == 11:
            mcts = MCTS6tree(state, self.player_number, self.timer)
            for i in range(11):
                for j in range(11):
                    if state[i][j] == 1 or state[i][j] == 2:
                        mcts.filled += 1
            if mcts.filled == 0:
                return (0,0)
            move = mcts.MCTS6()
            return move
        else:
            mcts = MCTStree(state, self.player_number, self.timer)
            for i in range(7):
                for j in range(7):
                    if state[i][j] == 1 or state[i][j] == 2:
                        mcts.filled += 1
            if mcts.filled == 0:
                return (0,0)
            move = mcts.MCTS()
            return move
        
