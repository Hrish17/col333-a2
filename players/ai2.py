import time
import math
import random
import numpy as np
from helper import *



##I want to implement the MCTS algorithm for the AI player for havannah
##First implementing the node class for the tree






total_moves = 0
opponent = "Random"
current_remaining_time = 0
# def neighbours_2(i,j):
#     dim = 11
#     neighbours = []
#     if i-1>=0:
#         neighbours.append((i-1,j))
#     if i-1>=0 and j+1<dim:
#         neighbours.append((i-1,j+1))
#     if j+1<dim:
#         neighbours.append((i,j+1))
#     if i+1<dim:
#         neighbours.append((i+1,j))
#     if j-1>=0:
#         neighbours.append((i,j-1))
#     if i-1>=0 and j-1>=0:
#         neighbours.append((i-1,j-1))
#     if i-2>=0:
#         neighbours.append((i-2,j))
#     if i-2>=0 and j+1<dim:
#         neighbours.append((i-2,j+1))
#     if i-2>=0 and j+2<dim:
#         neighbours.append((i-2,j+2))
#     if i-1>=0 and j+2<dim:
#         neighbours.append((i-1,j+2))
#     if j+2<dim:
#         neighbours.append((i,j+2))
#     if i+1<dim and j+1<dim:
#         neighbours.append((i+1,j+1))
#     if i+2<dim:
#         neighbours.append((i+2,j))
#     if i+1<dim and j-1>=0:
#         neighbours.append((i+1,j-1))
#     if j-2>=0:
#         neighbours.append((i,j-2))
#     if i-1>=0 and j-2>=0:
#         neighbours.append((i-1,j-2))
#     if i-2>=0 and j-2>=0:
#         neighbours.append((i-2,j-2))
#     if i-2>=0 and j-1>=0:
#         neighbours.append((i-2,j-1))
#     return neighbours 

def my_neighbours(i,j):
    dim =11
    half = 5
    neighbours = []
    if j<4:
        if i-1>=0:
            neighbours.append((i-1,j))
        if j+1<dim:
            neighbours.append((i,j+1))
        if i+1<dim and j+1<dim:
            neighbours.append((i+1,j+1))
        if i+1<dim:
            neighbours.append((i+1,j))
        if j-1>=0:
            neighbours.append((i,j-1))
        if i-1>=0 and j-1>=0:
            neighbours.append((i-1,j-1))
        if i-1>=0 and j+1<dim:
            neighbours.append((i-1,j+1))
        if i+1<dim and j+2<dim:
            neighbours.append((i+1,j+2))
        if i+2<dim and j+1<dim:
            neighbours.append((i+2,j+1))
        if i+1<dim and j-1>=0:
            neighbours.append((i+1,j-1))
        if i-1>=0 and j-2>=0:
            neighbours.append((i-1,j-2))
        if i-2>=0 and j-1>=0:
            neighbours.append((i-2,j-1))
    elif j==5:
        if i-1>=0:
            neighbours.append((i-1,j))
        if i-1>=0 and j+1<dim:
            neighbours.append((i-1,j+1))
        if j+1<dim:
            neighbours.append((i,j+1))
        if i+1<dim:
            neighbours.append((i+1,j))
        if j-1>=0:
            neighbours.append((i,j-1))
        if i-1>=0 and j-1>=0:
            neighbours.append((i-1,j-1))
        if i-2>=0 and j+1<dim:
            neighbours.append((i-2,j+1))
        if i-1>=0 and j+2<dim:
            neighbours.append((i-1,j+2))
        if i+1<dim and j+1<dim:
            neighbours.append((i+1,j+1))
        if i+1<dim and j-1>=0:
            neighbours.append((i+1,j-1))
        if i-1>=0 and j-2>=0:
            neighbours.append((i-1,j-2))
        if i-2>=0 and j-1>=0:
            neighbours.append((i-2,j-1))
    elif j==4:
        if i-1>=0:
            neighbours.append((i-1,j))
        if j+1<dim:
            neighbours.append((i,j+1))
        if i+1<dim and j+1<dim:
            neighbours.append((i+1,j+1))
        if i+1<dim:
            neighbours.append((i+1,j))
        if j-1>=0:
            neighbours.append((i,j-1))
        if i-1>=0 and j-1>=0:
            neighbours.append((i-1,j-1))
        if i-1>=0 and j+1<dim:
            neighbours.append((i-1,j+1))
        if j+2<dim:
            neighbours.append((i,j+2))
        if i+2<dim and j+1<dim:
            neighbours.append((i+2,j+1))
        if i+1<dim and j-1>=0:
            neighbours.append((i+1,j-1))
        if i-1>=0 and j-2>=0:
            neighbours.append((i-1,j-2))
        if i-2>=0 and j-1>=0:
            neighbours.append((i-2,j-1))
    elif j==6:
        if i-1>=0:
            neighbours.append((i-1,j))
        if i-1>=0 and j+1<dim:
            neighbours.append((i-1,j+1))
        if j+1<dim:
            neighbours.append((i,j+1))
        if i+1<dim:
            neighbours.append((i+1,j))
        if i+1<dim and j-1>=0:
            neighbours.append((i+1,j-1))
        if j-1>=0:
            neighbours.append((i,j-1))
        if i-2>=0 and j+1<dim:
            neighbours.append((i-2,j+1))
        if i-1>=0 and j+2<dim:
            neighbours.append((i-1,j+2))
        if i+1<dim and j+1<dim:
            neighbours.append((i+1,j+1))
        if i+2<dim and j-1>=0:
            neighbours.append((i+2,j-1))
        if j-2>=0:
            neighbours.append((i,j-2))
        if i-1>=0 and j-1>=0:
            neighbours.append((i-1,j-1))
    else:
        if i-1>=0:
            neighbours.append((i-1,j))
        if i-1>=0 and j+1<dim:
            neighbours.append((i-1,j+1))
        if j+1<dim:
            neighbours.append((i,j+1))
        if i+1<dim:
            neighbours.append((i+1,j))
        if i+1<dim and j-1>=0:
            neighbours.append((i+1,j-1))
        if j-1>=0:
            neighbours.append((i,j-1))
        if i-2>=0 and j+1<dim:
            neighbours.append((i-2,j+1))
        if i-1>=0 and j+2<dim:
            neighbours.append((i-1,j+2))
        if i+1<dim and j+1<dim:
            neighbours.append((i+1,j+1))
        if i+2<dim and j-1>=0:
            neighbours.append((i+2,j-1))
        if i+1<dim and j-2>=0:
            neighbours.append((i+1,j-2))
        if i-1>=0 and j-1>=0:
            neighbours.append((i-1,j-1))
    
    return neighbours
    


class MTCS:
    def __init__(self,state,player):
        self.player = player
        self.state = state
        self.all_moves = get_valid_actions(state)
        self.root = node(state,None,player,None,self.all_moves)
        self.c = 2
        self.best_move = (0,0)
        print("Opponent",opponent)
        if opponent == "TA":
            if(total_moves<=2):
                self.max_time = 19
            elif total_moves<=9:
                self.max_time = 25
            else:
                self.max_time = 18
        elif opponent == "Group":
            print("current_remaining_time ",current_remaining_time)
            if total_moves<=6:
                self.max_time = 2*current_remaining_time/len((self.all_moves))
            elif total_moves <= 10:
                self.max_time = min(19,2*(current_remaining_time/len((self.all_moves)))*1.5)
            elif total_moves <= 15:
                self.max_time = min(19,2*(current_remaining_time/len((self.all_moves)))*2)
            elif total_moves <= 25:
                self.max_time = min(19,2*(current_remaining_time/len((self.all_moves)))*2)
            else:
                self.max_time = 2*current_remaining_time/(len(self.all_moves))
        else:
            self.max_time = 10
        self.iterations = 0

    def search(self):
        start_time = time.time()
        if(total_moves==1):
            if self.state[self.best_move]==0:
                return self.best_move
            else:
                return (1,4)
        
        # if opponent == "Group" and total_moves<=20:
        #     best_action = self.all_moves[0]
        #     cur_score = 0
        #     for action in self.all_moves:
        #         temp = self.score(action)
        #         if temp>cur_score:
        #             cur_score = temp
        #             best_action = action
        #     return best_action
        print("Max time",self.max_time)
        while time.time() - start_time < self.max_time:
            self.iterations+=1
            self.iteration()
        best_child = self.best_child(self.root)
        print("Iterations1 Time remaining_moves",self.iterations,self.max_time,len(self.root.children))
        return best_child.action

    ## function for traversal fo the tree
    def traverse(self,node):
        while node.expanded and not node.terminal:
            node = node.best_child()
        return node

    ##function for expansion of the tree
    def expand(self,node):
        node.expand()
    



    ##function for rollout
    def rollout(self,node):
        actions = node.actions.copy()
        temp_state = node.state.copy()
        # while not node.terminal:
        #     node = node.next_random_child()
        # if node.winner == self.player:
        #     return 1
        # elif node.winner == 0:
        #     return 0
        # else:
        #     return -1
        cur_player = node.player
        winning,way = check_win(temp_state,node.action,cur_player%2+1)
        if winning:
            if cur_player%2+1==self.player:
                return 1
            else:
                return -1
        
        while len(actions)!=0:
            action = random.choice(actions)
            actions.remove(action)
            temp_state[action[0],action[1]] = cur_player
            winning,way = check_win(temp_state,action,cur_player) 
            if winning:
                if cur_player == self.player:
                    return 1
                else:
                    return -1
            cur_player = cur_player%2+1
        return 0
    
    ##function for backpropagation
    def backpropagate(self,node,result):
        temp = node
        while node is not None:
            node.visits +=1
            if node.player == self.player:
                node.value -= result
            else:
                node.value += result
            node = node.parent
        while temp is not None:
            temp.update()
            temp = temp.parent
    
    ##function for iteration
    def iteration(self):
        node = self.traverse(self.root)
        result  = 0
        if node.terminal:
            if node.winner == self.player:
                result = 1
            elif node.winner == 0:
                result = 0
            else :
                result = -1
            self.backpropagate(node,result)
        else:
            self.expand(node)
            child = node.next_random_child()
            result = self.rollout(child) 
            self.backpropagate(child,result)


    def best_child(self,node):
        return max(node.children,key = lambda x: x.visits)
    


class node:
    def __init__(self,state,parent,player,action,all_moves):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.untried_actions = []
        self.player = player
        self.winner = None
        self.terminal = False
        self.expanded = False
        self.ucb = 1e15
        if total_moves<=25:
            self.c = 1
        else:
            self.c = np.sqrt(2)
        self.actions = all_moves.copy()
        if action is not None:
            self.actions.remove(action)
        self.action = action
        self.value = 0
        if self.action is not None:
            self.check_terminal()
    def dfs(self,i,j,visited,bool_array):
        if(visited[i,j]):
            return
        visited[i,j] = True

        neighbours = get_neighbours(self.state.shape[0],(i,j))
        for neighbour in neighbours:
            if is_valid(neighbour[0],neighbour[1],self.state.shape[0]):
                if bool_array[neighbour[0],neighbour[1]] and not visited[neighbour[0],neighbour[1]]:
                    self.dfs(neighbour[0],neighbour[1],visited,bool_array)
        return




    def connected_components(self):
        bool_array = self.state == self.player%2+1
        components = 0
        visited = np.zeros(self.state.shape)
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if not is_valid(i,j,self.state.shape[0]):
                    continue
                if not bool_array[i,j]:
                    continue
                else:
                    if not visited[i,j]:
                        components+=1
                        self.dfs(i,j,visited,bool_array)
        return components
                    


    def heuristic(self):
        all_corners = get_all_corners(self.state.shape[0])
        corner = 0
        for i in all_corners:
            if self.state[i[0],i[1]] == self.player%2+1:
                corner+=1
        components = self.connected_components()
        return -1*components
        # return 0
        # pass

    
    
    
    
    def check_terminal(self):
        winning,way = check_win(self.state,self.action,self.player%2+1)
        if winning:
            self.winner = self.player%2+1
            self.terminal = True
            return
        else:
            if(len(self.actions)==0):
                self.terminal = True
                self.winner = 0
                return
            else:
                self.terminal = False
                self.winner = None
                return
    



    def check(self,i,j):
        all_neighbours = get_neighbours(self.state.shape[0],(i,j))
        for neighbour in all_neighbours:
            if self.state[neighbour[0],neighbour[1]] !=0 and self.state[neighbour[0],neighbour[1]]!=3:
                return True
        
        return False
        pass
    
    def check_2(self,i,j):
        all_neighbours = my_neighbours(i,j)
        for neighbour in all_neighbours:
            if self.state[neighbour[0],neighbour[1]] !=0 and self.state[neighbour[0],neighbour[1]]!=3:
                return True

        return False
        pass




    # def check_2(self,i,j):
    #     all_neighbours = get_neighbours_2(i,j)
    #     count = 0
    #     for neighbour in all_neighbours:
    #         if self.state[neighbour[0],neighbour[1]] !=0 and self.state[neighbour[0],neighbour[1]]!=3:
    #            count+=1
    #     if count>=2:
    #         return True    
    #     return False

    def expand(self):
        self.expanded = True
        if opponent=="Group":
            if total_moves<=6:
                for action in self.actions:
                    if(self.check(action[0],action[1])):
                        temp_state = self.state.copy()
                        temp_state[action[0],action[1]] = self.player
                        self.children.append(node(temp_state,self,self.player%2+1,action,self.actions))
            elif total_moves<=25:
                for action in self.actions:
                    if self.check_2(action[0],action[1]):
                        temp_state = self.state.copy()
                        temp_state[action[0],action[1]] = self.player
                        self.children.append(node(temp_state,self,self.player%2+1,action,self.actions))
            else:
                for action in self.actions:
                    temp_state = self.state.copy()
                    temp_state[action[0],action[1]] = self.player
                    self.children.append(node(temp_state,self,self.player%2+1,action,self.actions))
        else:
            for action in self.actions:
                temp_state = self.state.copy()
                temp_state[action[0],action[1]] = self.player
                self.children.append(node(temp_state,self,self.player%2+1,action,self.actions))
    def next_random_child(self):
        random_action = random.choice(self.actions)
        temp_state = self.state.copy()
        temp_state[random_action[0],random_action[1]] = self.player
        return node(temp_state,self,self.player%2+1,random_action,self.actions)
    

    def update(self):
        # if self.parent is not None:
        #     self.ucb = self.value/self.visits + self.c*math.sqrt(math.log(self.parent.visits)/self.visits)
        for child in self.children:
            if child.visits !=0:
                child.ucb = child.value/child.visits + self.c*math.sqrt(math.log(self.visits)/child.visits)

    
    def best_child(self):
        return max(self.children,key = lambda x: x.ucb)
    




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
        # self.previous_state = np.zeroes((11,11))
        # self.opponent = "Random"
    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move

        # Parameters
        `state: Tuple[np.array]
            - a numpy array containing the state of the board using the following encoding:
            - the board maintains its same two dimensions
            - spaces that are unoccupied are marked as 0
            - spaces that are blocked are marked as 3
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them

        # Returns
        Tuple[int, int]: action (coordinates of a board cell)
        """
        global total_moves
        global opponent
        global current_remaining_time
        if(total_moves==0):
            total_remaining_time = fetch_remaining_time(self.timer,self.player_number)
            if total_remaining_time > 400:
                opponent = "Group"
            elif total_remaining_time > 300:
                opponent = "TA"
        total_moves+=1
        all_valid_actions = get_valid_actions(state)
        for action in all_valid_actions:
            state[action[0],action[1]] = self.player_number
            winning,way = check_win(state,action,self.player_number)
            state[action[0],action[1]] = 0
            if winning:
                return action
        
        for action in all_valid_actions:
            state[action[0],action[1]] = self.player_number%2+1
            winning,way = check_win(state,action,self.player_number%2+1)
            state[action[0],action[1]] = 0
            if winning:
                return action
        
        current_remaining_time = fetch_remaining_time(self.timer,self.player_number)
        ##Implementing the MCTS algorithm
        mcts = MTCS(state,self.player_number)
        action = mcts.search()
        return action
        # Do the rest of your implementation here
        # return (0,0)

