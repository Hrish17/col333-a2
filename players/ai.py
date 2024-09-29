import time
import math
import random
import numpy as np
from helper import *

class MCTS_Node:
    def __init__(self, visits, value):
        self.visits = visits
        self.value = value
        self.children = []
        self.parent = None
        self.state = None
        self.action = None
        self.player = None

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
        self.max_time = 12 # seconds
        self.c = 2

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

        # Do the rest of your implementation here
        if len(np.argwhere(state == 1)) == 0 and len(np.argwhere(state == 2)) == 0:
            return (0,0)
        # get dimensions of the board
        if state.shape[0] == 7:
            moves_played = len(np.argwhere(state == 1))
            if moves_played <= 4:
                self.max_time = 18
            elif moves_played <= 8:
                self.max_time = 14
            else:
                self.max_time = 10
        return self.mcts(state)
    
    def ucb1(self, node: MCTS_Node, parent_visits: int) -> float:
        if node.visits == 0:
            return float('inf')
        exploitation = node.value / node.visits
        exploration = math.sqrt(math.log(parent_visits) / node.visits)
        return exploitation + self.c * exploration
    
    def get_next_state(self, state: np.array, action: Tuple[int, int], player_number: int) -> np.array:
        new_state = np.copy(state)
        new_state[action[0], action[1]] = player_number
        return new_state
    
    def get_num_kites(self, board, action):
        x, y = action[0], action[1]
        dims = board.shape[0]
        count = 0
        dirs = [[(-2,-1), (-1,-1), (-1,0)], [(-2, 1), (-1, 0), (-1, 1)], [(-1, 2), (-1, 1), (0, 1)], [(1,1), (0,1), (1, 0)], [(1,-1), (0,-1), (1, 0)], [(-1,-2), (-1,-1), (0,-1)]]
        for dir in dirs:
            all_valid = True
            for d in dir:
                if not is_valid(x + d[0], y + d[1], dims):
                    all_valid = False
                    break
            if not all_valid:
                continue

            if board[x + dir[0][0], y + dir[0][1]] == self.player_number:
                a , b = 0, 0
                if board[x + dir[1][0], y + dir[1][1]] == self.player_number:
                    a += 1
                elif board[x + dir[1][0], y + dir[1][1]] == 3 - self.player_number:
                    b += 1
                if board[x + dir[2][0], y + dir[2][1]] == self.player_number:
                    a += 1
                elif board[x + dir[2][0], y + dir[2][1]] == 3 - self.player_number:
                    b += 1
                if (a >= b):
                    count += 1
        return count

    
    def mcts(self, state: np.array) -> Tuple[int, int]:
        start_time = time.time()
        possible_actions = get_valid_actions(state)
        
        root = MCTS_Node(0, 0)
        root.state = np.copy(state)
        root.player = self.player_number
        for action in possible_actions:
            child = MCTS_Node(0, 0)
            child.state = self.get_next_state(state, action, self.player_number)
            hasWon, _ = check_win(child.state, action, self.player_number)
            if hasWon:
                return action
            opponent_state = self.get_next_state(state, action, 3 - self.player_number)
            has_opponent_won, _ = check_win(opponent_state, action, 3 - self.player_number)
            if has_opponent_won:
                return action
            kites = self.get_num_kites(child.state, action)
            child.value = kites * 0.1
            child.player = 3 - self.player_number
            child.parent = root
            child.action = action
            root.children.append(child)

        # time limit for MCTS in seconds
        while time.time() - start_time < self.max_time:
            # print('time:', time.time() - start_time)
            node = self.traverse(root)
            if node.visits == 0:
                value = self.rollout(node)
            else:
                # expand
                possible_actions = get_valid_actions(node.state)
                if not possible_actions:
                    value = fetch_remaining_time(self.timer, self.player_number)/fetch_remaining_time(self.timer, 3-self.player_number)
                else:
                    for action in possible_actions:
                        child = MCTS_Node(0, 0)
                        child.state = self.get_next_state(node.state, action, self.player_number)
                        child.player = 3 - node.player
                        kites = self.get_num_kites(child.state, action)
                        if child.player == self.player_number:
                            child.value -= kites * 0.1
                        else:
                            child.value += kites * 0.1
                        child.parent = node
                        child.action = action
                        node.children.append(child)
                    value = self.rollout(random.choice(node.children))

            self.backpropagate(node, value)

        best_node = max(root.children, key=lambda x: x.visits)
        # print("flag 2")
        return best_node.action


    def traverse(self, node: MCTS_Node) -> MCTS_Node:
        while node.children:
            node = max(node.children, key=lambda x: self.ucb1(x, node.visits))
        return node
    
    def rollout(self, node: MCTS_Node) -> float:
        current_state = np.copy(node.state)
        current_player = node.player
        current_node = node

        while True:
            hasWon, _ = check_win(current_state, current_node.action, 3-current_player)
            if hasWon:
                return -1 if current_player == self.player_number else 1
            possible_actions = get_valid_actions(current_state)
            if not possible_actions:
                return fetch_remaining_time(self.timer, self.player_number)/fetch_remaining_time(self.timer, 3-self.player_number)
            action = random.choice(possible_actions)
            current_state = self.get_next_state(current_state, action, current_player)
            # create a new node
            new_node = MCTS_Node(0, 0)
            new_node.state = np.copy(current_state)
            new_node.action = action
            current_node = new_node
            current_player = 3 - current_player

    def backpropagate(self, node: MCTS_Node, value: float):
        while node:
            node.visits += 1
            if node.player == self.player_number:
                node.value -= value
            else:
                node.value += value
            node = node.parent