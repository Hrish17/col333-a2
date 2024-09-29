import time
import math
import random
import numpy as np
from helper import *
from typing import Tuple, List, Optional

# np.random.seed(0)
const = None
eps = 1e-12

class Node:
    def __init__(self,state,p_num,movs,move,par=None):
        self.state=state
        self.visits=0
        self.wins=0
        self.par=None
        self.child=[]
        self.player_num=p_num
        self.moves=movs
        self.par=par
        self.last_move = move
        self.is_expanded=False
        self.is_terminal=False
        self.val=0
        if move:
            self.check()
        
    def check(self):
        ch=check_win(self.state,self.last_move,3-self.player_num)
        if ch[0]:
            self.is_terminal=True
            self.val=-1
        if len(self.moves)==0:
            self.val=0
            self.is_terminal=True
    
    def ucb(self):
        if self.visits==0:
            return 1e9
        return (self.wins/self.visits)+const*np.sqrt(np.log(self.par.visits)/(self.visits))
    
    def best_move(self):
        return self.child[np.argmax(self.ucbs)]
        
    def expand(self):
        self.is_expanded=True
        for move in self.moves:
            new_state=self.state.copy()
            new_state[move]=self.player_num
            new_moves=self.moves.copy()
            new_moves.remove(move)
            self.child.append(Node(new_state,3-self.player_num,new_moves,move,self))   
        self.ucbs=[1e9 for _ in range(len(self.moves))]
    
    def get_win_ratio(self):
        return self.visits
    
    def best_child(self):
        ws=[ch.get_win_ratio() for ch in self.child]
        return self.child[np.argmax(ws)]
    
    def upd(self):
        self.ucbs=[ch.ucb() for ch in self.child]
        
    def print(self):
        print(f"Player to move : {self.player_num}")
        print(self.state)
        print(f"wins: {self.wins}")
        print(f"visits: {self.visits}")
        print()
    
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
        self.move=0
        self.prev_move=None

    def select(self,node:Node):
        while node.is_expanded and not node.is_terminal:
            node=node.best_move()
        return node
    
    def rollout(self,node:Node,p_type):
        state=node.state.copy()
        last_move=node.last_move
        p_num=node.player_num
        moves=node.moves.copy()
        while(1):    
            check=check_win(state,last_move,3-p_num)
            if check[0]:
                if p_num==p_type:
                    return -1
                else :
                    return 1
            if len(moves)==0:
                return 0
            num=np.random.choice(range(len(moves)))
            last_move=moves[num]
            moves.pop(num)
            state[last_move]=p_num
            p_num=3-p_num
            
        return 0
        
    def back_prop(self,node:Node,p_num,reward):
        while node:
            if node.player_num==p_num:
                node.wins += -1*reward
            else:
                node.wins += 1*reward
            node.visits+=1
            if node.is_expanded:
                node.upd()
            node = node.par
            
    def mcts(self,state):
        start=time.time()
        moves=get_valid_actions(state)
        root=Node(state,self.player_number,moves,None)
        p_type = root.player_num
        it=0
        while time.time()-start<self.max_t:
            it+=1
            node=self.select(root)
            if node.is_terminal:
                if node.player_num==self.player_number:
                    self.back_prop(node,self.player_number,node.val)
                else:
                    self.back_prop(node,self.player_number,-1*node.val)
            else:
                node.expand()
                child:Node=random.choice(node.child)
                reward=self.rollout(child,p_type)
                self.back_prop(child,p_type,reward)
        print(it)
        return root.best_child().last_move
                
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
        moved=self.move
        print(moved)
        global const
        const=np.sqrt(3.2/(1+np.sqrt(moved)/4.0))
        self.max_t=16
        if moved>2:
            self.max_t=21
            if moved>6:
                self.max_t=19
                if moved>11:
                    self.max_t=10
                    if moved>14:
                        self.max_t=3
        best_move=None
        start_move=(0,0)
        if self.move==0:
            if state[start_move]==0:
                best_move=start_move
            else:
                best_move=(2,1) 
        else:
            found = 0
            if self.move>2:
                next_moves = get_valid_actions(state,3-self.player_number)
                for i in next_moves:
                    state[i] = self.player_number
                    check=check_win(state,i,self.player_number)
                    if check[0]:
                        best_move = i
                        found  = 1
                        break
                    state[i] = 0
                if not found:
                    for i in next_moves:
                        state[i] = 3 - self.player_number
                        check=check_win(state,i,3-self.player_number)
                        if check[0]:
                            best_move = i
                            found  = 1
                            break
                        state[i] = 0
            if not found:
                best_move=self.mcts(state)
        
        self.move+=1
        self.prev_move=best_move
        return best_move