import time
import math
import random
import numpy as np
from helper import *
from typing import Tuple, List, Optional

# np.random.seed(0)

const = np.sqrt(2.2)
eps = 1e-12
side_allow=None
moved=0
dim=6
maxx=2*dim-1
factor_k=0.6
factor_h=1/8.0
decay=0.9
allowed=6.0
dir1=[(-1,-1),(0,-1),(1,0),(1,1),(0,1),(-1,0)]
dir2=[(0,-1),(1,-1),(1,0),(0,1),(-1,1),(-1,0)]
allr=[(1,-2),(2,-1),(1,1),(-1,2),(-2,1),(-1,-1)]
alll=[(-1,-2),(1,-1),(2,1),(1,2),(-1,1),(-2,-1)]
all_l=[(-1,-2),(1,-1),(2,1),(0,2),(-1,1),(-2,-1)]
all_r=[(2,-1),(1,1),(-1,2),(-2,1),(-1,-1),(0,-2)]
all_c=[(-2,-1),(-1,-2),(1,-1),(1,1),(-1,2),(-2,1)]
last_my_move=None
last_opp_move=None
orig_state=None
def has_played_neighbor(state, x, y):

    if y>=dim-1:
        for direction in dir2:
            nx, ny = x + direction[0], y + direction[1]
            if 0 <= nx < state.shape[0] and 0 <= ny < state.shape[1]:
                if state[nx,ny]%3:
                    return True
    else:
        for direction in dir1:
            nx, ny = x + direction[0], y + direction[1]
            if 0 <= nx < state.shape[0] and 0 <= ny < state.shape[1]:
                if state[nx, ny] == 1 or state[nx, ny] == 2:
                    return True
    if moved>10:
        if y>dim:
            for dir in allr:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    if state[(new_x,new_y)]%3:
                        return True
        elif y<dim-2:
            for dir in alll:
                new_x=y+dir[0]
                new_y=x+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    if state[(new_x,new_y)]%3:
                        return True
        elif y==dim:
            for dir in all_r:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    if state[(new_x,new_y)]%3:
                        return True
        elif y==dim-2:
            for dir in all_l:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    if state[(new_x,new_y)]%3:
                        return True
        else:
            for dir in all_c:
                new_x=x+dir[0]
                new_y=y+dir[1]
                if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                    if state[(new_x,new_y)]%3:
                        return True
                     
    return False

def add_moves_with_played_neighbors(valid_moves, state):
    filtered_moves_with_neighbors = set()
    for move in valid_moves:
        x, y = move
        if state[move]%3:
            filtered_moves_with_neighbors.add(move)
        if has_played_neighbor(state, x, y):
            filtered_moves_with_neighbors.add(move)
    return list(filtered_moves_with_neighbors)

def check_dist(x,y):
    if hex_manhattan_distance(last_my_move,(x,y))/factor_h>allowed:
        if hex_manhattan_distance(last_opp_move,(x,y))/factor_h>allowed:
            return False
    return True

def get_valid_actions2(board: np.array):

    valid_moves = np.argwhere(board == 0)
    valid_moves = np.array([tuple(move) for move in valid_moves])
    x_coords = valid_moves[:, 0]
    y_coords = valid_moves[:, 1]
    edge_mask = (x_coords < side_allow) | (x_coords >= board.shape[0] - side_allow) | (y_coords < side_allow) | (y_coords >= board.shape[1] - side_allow)
    filtered_moves = valid_moves[edge_mask]
    fil_moves=[]
    if moved<20:
        for move in filtered_moves:
            if check_dist(move[0],move[1]) or has_played_neighbor(board,move[0],move[1]):
                fil_moves.append(move)
    filtered_moves=fil_moves
    rem_moves=valid_moves[~edge_mask]
    rem_moves=[tuple(move) for move in rem_moves]
    added_moves=add_moves_with_played_neighbors(rem_moves,board)
    filtered_moves_list = [tuple(move) for move in filtered_moves]
    filtered_moves_list+=added_moves
    # print(filtered_moves_list)
    return filtered_moves_list


def hex_manhattan_distance(a,b):
    return factor_h*(abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[0]+ a[1]- b[0]- b[1])/2.0)

def kites(state,pos):
    num=0
    y=pos[1]
    x=pos[0]
    if y>dim:
        for dir in allr:
            new_x=x+dir[0]
            new_y=y+dir[1]
            if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                if state[(new_x,new_y)]%3:
                    num+=1
    elif y<dim-2:
        for dir in alll:
            new_x=y+dir[0]
            new_y=x+dir[1]
            if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                if state[(new_x,new_y)]%3:
                    num+=1
    elif y==dim:
        for dir in all_r:
            new_x=x+dir[0]
            new_y=y+dir[1]
            if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                if state[(new_x,new_y)]%3:
                    num+=1
    elif y==dim-2:
        for dir in all_l:
            new_x=x+dir[0]
            new_y=y+dir[1]
            if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                if state[(new_x,new_y)]%3:
                    num+=1
    else:
        for dir in all_c:
            new_x=x+dir[0]
            new_y=y+dir[1]
            if new_x>=0 and new_y>=0 and new_x<maxx and new_y<maxx:
                if state[(new_x,new_y)]%3:
                    num+=1
    if y>=dim-1:
        for direction in dir2:
            nx, ny = x + direction[0], y + direction[1]
            if 0 <= nx < state.shape[0] and 0 <= ny < state.shape[1]:
                if state[nx,ny]%3:
                    num+=0.2
    else:
        for direction in dir1:
            nx, ny = x + direction[0], y + direction[1]
            if 0 <= nx < state.shape[0] and 0 <= ny < state.shape[1]:
                if state[nx, ny] == 1 or state[nx, ny] == 2:
                    num+=0.2
    return factor_k*num

class Node:
    def __init__(self,p_num,movs,move,dep=1,par=None,last_opp_mov=None,last_my_mov=None):
        self.visits=0
        self.wins=0
        self.losses=0
        self.par=None
        self.child=[]
        self.player_num=p_num
        self.moves=movs
        self.par=par
        self.last_move = move
        self.is_expanded=False
        self.is_terminal=False
        self.opp_move=last_opp_mov
        self.val=0
        self.dep=dep
        if move:
            self.check()
            if last_opp_mov:
                self.wins-=hex_manhattan_distance(move,last_opp_mov)*np.sqrt(self.dep/2.0)
            if last_my_mov:
                self.wins-=hex_manhattan_distance(move,last_my_mov)
            x,y=move
            if x==0 and y%(dim-1)==0:
                self.wins+=3.0
            elif y%(2*(dim-1))==0 and x==dim-1:
                self.wins+=3.0
            elif x==2*(dim-1):
                self.wins+=3.0
            elif x==0 or y==0 or x-y==dim-1 or x+y==3*(dim-1) or y==2*(dim-1):
                self.wins+=1.0
    def check(self):
        new_state=orig_state.copy()
        curr=self
        while curr.last_move:
            new_state[curr.last_move]=3-curr.player_num
            curr=curr.par
        if self.last_move:
            self.wins+=kites(new_state,self.last_move)/np.sqrt(self.dep)
        ch=check_win(new_state,self.last_move,3-self.player_num)
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
            new_moves=self.moves.copy()
            new_moves.remove(move)
            self.child.append(Node(3-self.player_num,new_moves,move,self.dep+1,par=self,last_my_mov=self.opp_move,last_opp_mov=self.last_move))   
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

    def select(self,node:Node,state):
        while node.is_expanded and not node.is_terminal:
            node=node.best_move()
            state[node.last_move]=3-node.player_num
        return node,state
    
    def rollout(self,state,node:Node,p_type):
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
            
    def mcts(self,moves=None):
        global orig_state
        start=time.time()
        if not moves:
            moves=get_valid_actions2(orig_state)
        root=Node(self.player_number,moves,None,last_opp_mov=last_opp_move,last_my_mov=last_my_move)
        p_type = root.player_num
        it=0
        while time.time()-start<self.max_t:
            it+=1
            state=orig_state.copy()
            node,state=self.select(root,state)
            if node.is_terminal:
                if node.player_num==self.player_number:
                    self.back_prop(node,self.player_number,node.val)
                else:
                    self.back_prop(node,self.player_number,-1*node.val)
            else:
                node.expand()
                child:Node=random.choice(node.child)
                reward=self.rollout(state,child,p_type)
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
        global moved
        global const
        global side_allow
        global orig_state
        global last_my_move
        global last_opp_move
        global factor_k
        global decay
        global allowed
        if moved or np.any(state % 3 > 0):
            diff = np.where(orig_state != state)
            last_opp_move= (diff[0][0],diff[1][0])
        
        orig_state=state
        print(moved)
        self.max_t = 12
        if moved > 2:
            self.max_t = 15.5
            if moved>6:
                self.max_t=21.5
                if moved > 10:
                    allowed+=0.5
                    self.max_t = 23.5
                    if moved > 15:
                        self.max_t = 20.5
                        if moved > 20:
                            self.max_t = 16.5
                            if moved>25:
                                self.max_t = 11
                                if moved>30:
                                    self.max_t=5
        side_allow=3
        if moved>5:
            side_allow=2
            if moved > 25:
                side_allow=5
 
        best_move=None
        start_move=(1,5)
        if moved==0:
            if state[start_move]==0:
                best_move=start_move
            else:
                best_move=(1,1) 
        else:
            found = 0
            if moved>3:
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
                moves=get_valid_actions2(state)
                best_move=self.mcts(moves)
        
        moved+=1
        factor_k/=decay
        decay*=(1.02)
        last_my_move=best_move
        return best_move