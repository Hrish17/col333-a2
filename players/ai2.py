import time
import math
import random
import numpy as np
from helper import *
from typing import Tuple, List

# np.random.seed(0)

const = np.sqrt(1.5)
eps = 1e-12
moved=0
dim=6
maxx=2*dim-1
factor_h=1/100.0
dir1=[(-1,-1),(0,-1),(1,0),(1,1),(0,1),(-1,0)]
dir2=[(0,-1),(1,-1),(1,0),(0,1),(-1,1),(-1,0)]
dir3=[(-1,1),(0,1),(1,0),(0,-1),(-1,-1),(-1,0)]
allr=[(1,-2),(2,-1),(1,1),(-1,2),(-2,1),(-1,-1)]
alll=[(-1,-2),(1,-1),(2,1),(1,2),(-1,1),(-2,-1)]
all_l=[(-1,-2),(1,-1),(2,1),(0,2),(-1,1),(-2,-1)]
all_r=[(2,-1),(1,1),(-1,2),(-2,1),(-1,-1),(0,-2)]
all_c=[(-2,-1),(-1,-2),(1,-1),(1,1),(-1,2),(-2,1)]
rem_moves=None
def get_my_neighbours(state, x, y):
    neighs=[]
    if y>dim-1:
        for direction in dir2:
            nx, ny = x + direction[0], y + direction[1]
            if 0 <= nx < state.shape[0] and 0 <= ny < state.shape[1]:
                neighs.append((nx,ny))
    elif y<dim:
        for direction in dir1:
            nx, ny = x + direction[0], y + direction[1]
            if 0 <= nx < state.shape[0] and 0 <= ny < state.shape[1]:
                neighs.append((nx,ny))
    else:
        for direction in dir3:
            nx, ny = x + direction[0], y + direction[1]
            if 0 <= nx < state.shape[0] and 0 <= ny < state.shape[1]:
                neighs.append((nx,ny))
    
    if moved>3:
        if y>dim:
            for dir in allr:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    neighs.append((new_x,new_y))
        elif y<dim-2:
            for dir in alll:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    neighs.append((new_x,new_y))
                    
        elif y==dim:
            for dir in all_r:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    neighs.append((new_x,new_y))
                    
        elif y==dim-2:
            for dir in all_l:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    neighs.append((new_x,new_y))
        else:
            for dir in all_c:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    neighs.append((new_x,new_y))
                        
    return neighs

def add_moves_with_played_neighbors(valid_moves, state):
    filtered_moves_with_neighbors=[]
    for move in valid_moves:
        x, y = move
        all_neighs=get_my_neighbours(state, x, y)
        for mov in all_neighs:
            if state[mov]==1 or state[mov]==2:
                filtered_moves_with_neighbors.append(move)
                break
    return filtered_moves_with_neighbors

def modify(state,play_moves,move):
    valid_m=set(play_moves)
    neighs=get_my_neighbours(state,move[0],move[1])
    for m in neighs:
        if state[m]==0:
            if m in rem_moves:
                valid_m.add(m)
    return list(valid_m)

def get_valid_actions2(board: np.array):
    valid_moves = np.argwhere(board == 0)
    valid_moves = [tuple(move) for move in valid_moves]
    return add_moves_with_played_neighbors(valid_moves,board)


def hex_manhattan_distance(a,b):
    return factor_h*(abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[0]+ a[1]- b[0]- b[1])/3.0)

class Node:
    def __init__(self,state,p_num,play_moves,move,dep=1,par=None,last_opp_mov=None,last_my_mov=None):
        self.visits=0
        self.wins=0
        self.losses=0
        self.par=None
        self.child=[]
        self.player_num=p_num
        self.moves=play_moves
        self.par=par
        self.last_move = move
        self.is_expanded=False
        self.is_terminal=False
        self.opp_move=last_opp_mov
        self.val=0
        self.dep=dep
        self.state=state
        if move:
            self.moves=modify(state,self.moves,move)
            self.check()
            # if last_opp_mov:
            #     self.wins-=hex_manhattan_distance(move,last_opp_mov)*np.sqrt(self.dep**1.04)
            # if last_my_mov:
            #     self.wins-=hex_manhattan_distance(move,last_my_mov)*np.sqrt(self.dep**1.04)
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
        return ((self.wins-self.losses)/self.visits)+const*np.sqrt(np.log(self.par.visits)/(self.visits))
    
    def best_move(self):
        return self.child[np.argmax(self.ucbs)]
        
    def expand(self):
        self.is_expanded=True
        for move in self.moves:
            new_state=self.state.copy()
            new_state[move]=self.player_num
            new_moves=self.moves.copy()
            new_moves.remove(move)
            self.child.append(Node(new_state,3-self.player_num,new_moves,move,self.dep+1,par=self,last_my_mov=self.opp_move,last_opp_mov=self.last_move))   
        self.ucbs=[1e9 for _ in range(len(self.moves))]
    
    def get_win_ratio(self):
        return self.visits+(self.wins/self.visits)
    
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
                    return -1.0
                else :
                    return 1.0
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
                if reward==1.0:
                    node.losses+=1.0
                else:
                    node.wins+=1.0
            else:
                if reward==1.0:
                    node.wins += 1.0
                else:
                    node.losses += 1.0
            node.visits+=1
            if node.is_expanded:
                node.upd()
            node = node.par
            
    def mcts(self,state):
        start=time.time()
        moves=get_valid_actions2(state)
        print(f"moves at root :{len(moves)}")
        # print(moves)
        root=Node(state,self.player_number,moves,None)
        p_type = root.player_num
        it=0
        while time.time()-start<self.max_t-0.1:
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
        print(f"iterations are {it}")
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
        global moved
        global const
        global rem_moves
        rem_moves=set(get_valid_actions(state))
        print(f"move num {moved}")
        self.max_t = 12
        if moved > 2:
            self.max_t = 15.5
            if moved>6:
                self.max_t=21.5
                if moved > 10:
                    self.max_t = 22.5
                    if moved > 15:
                        self.max_t = 19.5
                        if moved > 20:
                            self.max_t = 15
                            if moved>25:
                                self.max_t = 8
                                if moved>30:
                                    self.max_t=4
 
        best_move=None
        start_move=(0,5)
        if moved==0:
            if state[start_move]==0:
                best_move=start_move
            else:
                best_move=(0,10) 
        else:
            found = False
            if moved>3:
                next_moves = get_valid_actions(state,3-self.player_number)
                for i in next_moves:
                    state[i] = self.player_number
                    check=check_win(state,i,self.player_number)
                    if check[0]:
                        best_move = i
                        found  = True
                        break
                    state[i] = 0
                if not found:
                    for i in next_moves:
                        state[i] = 3 - self.player_number
                        check=check_win(state,i,3-self.player_number)
                        if check[0]:
                            best_move = i
                            found  = True
                            break
                        state[i] = 0
            if not found:
                best_move=self.mcts(state)
        
        moved+=1
        return best_move