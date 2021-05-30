# Conducts a Monte-Carlo Tree Search
import math
import random
from copy import deepcopy
from env import OthelloEnv
import numpy as np

class MCTS():
    def __init__(self, state, env, model, explore_weight=0.5, sims=25):
        """Initialize a root node with its children and set explore weight"""
        self.model = model
        self.sims = sims
        self.root = Node(state, env.player_to_move, 1)
        self.turn = env.player_to_move
        self.expand(self.root)
        self.explore_weight = explore_weight  # TODO: Determine best explore_weight
        self.pi = np.zeros(66)

    def simulate(self):
        """Search and expand the tree for the specified number of iterations"""
        for _ in range(self.sims):
            # STEP 1: MOVE TO LEAF
            leaf = self.traverse()
            # STEP 2: EXPAND LEAF
            self.expand(leaf)
            # STEP 3: BACKFILL VALUE or RESULT
            # VALUE/RESULT are unbiased. backprop changes sign to POV 
            result, game_over = OthelloEnv(board=deepcopy(leaf.state), turn=leaf.turn).check_game_over()
            if game_over:
                self.backprop(leaf, result)
            else:
                _, value = self.model.predict(self.convert_to_model_input([leaf.state * leaf.turn])) 
                if leaf.turn == -1:
                    self.backprop(leaf, value * -1) # unbiased (no POV)
                self.backprop(leaf, value) # unbiased (no POV)

        self.update_policy()

    def update_policy(self):
        for x in self.root.children:
            action = x.action
            self.pi[action] = x.num_visited ** (1 - self.explore_weight)
        self.pi /= np.sum(self.pi)
        return self.pi

    def traverse(self):
        """For each state, select the action that maximizes UCB1 score until a leaf node is reached"""
        curr_node = self.root
        while len(curr_node.children) > 0:
            best_puct = -99
            next_node = None
            for x in curr_node.children:
                puct = self.puct(curr_node, x)
                if puct > best_puct:
                    best_puct = puct
                    next_node = x
            curr_node = next_node
        return curr_node

    def puct(self, parent, child):
        """Given (s, a) calculate its PUCT score"""
        if child.num_visited == 0:
            return child.prior * math.sqrt(parent.num_visited) / (child.num_visited + 1)
        U = -child.total_value/child.num_visited + child.prior * math.sqrt(parent.num_visited) / (child.num_visited + 1)
        return U

    def expand(self, node):
        """Given a node, populate its children attribute with possible actions"""
        temp_env = OthelloEnv(node.state, node.turn)

        _, game_over = temp_env.check_game_over()
        
        if not game_over:
            actions = temp_env.possible_actions()
            policy, _ = self.model.predict(temp_env.convert_to_model_input([node.state * node.turn]))
            
            legals = np.zeros(66)
            for x in actions:
                legals[x] = 1
            
            policy = policy[0] * legals
            policy /= np.sum(policy)

            for action in actions:
                state, _, _, turn = temp_env.step(action)
                node.children.append(Node(state, turn, policy[action], parent=node, action=action))
                temp_env.pop()


    def backprop(self, node, result):
        """Given a node and a result, update the total value and visit count for every parent node"""
        pointer = node
        while pointer:
            pointer.num_visited += 1
            pointer.total_value += result * pointer.turn
            pointer = pointer.parent

    def convert_to_model_input(self, boards):
        new_X = []

        for board in boards:
            new_board = np.reshape(board, (1, 8, 8))
            new_X.append(new_board)
        
        return np.array(new_X)
class Node():
    def __init__(self, state, turn, prior, parent=None, action=None):
        """Each node in the tree stores state, player turn, parent, action, total value, visit count, and children"""
        self.state = state
        self.turn = turn
        self.parent = parent
        self.prior = prior
        self.action = action
        self.total_value = 0
        self.num_visited = 0
        self.children = []

