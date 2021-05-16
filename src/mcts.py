# Conducts a Monte-Carlo Tree Search
import math
import random
from copy import deepcopy
from src.env import OthelloEnv


class MCTS():
    def __init__(self, state, env, explore_weight=0.5):
        """Initialize a root node with its children and set explore weight"""
        self.root = Node(state, env.player_to_move)
        self.expand(self.root)
        self.explore_weight = explore_weight  # TODO: Determine best explore_weight

    def search(self, env, model, iter=5):
        """Search and expand the tree for the specified number of iterations"""
        # TODO: Implement correct params. Change iter to a time limit
        for _ in range(iter):
            curr_node = self.traverse()
            if curr_node.num_visited > 0:
                curr_node = self.expand(curr_node)
            rollout_result = self.rollout(curr_node.state, curr_node.turn)
            self.backprop(curr_node, rollout_result)

        best_node = max(self.root.children, key=lambda node: node.total_value/node.num_visited)
        return best_node.action

    def traverse(self):
        """For each state, select the action that maximizes UCB1 score until a leaf node is reached"""
        curr_node = self.root
        while curr_node.children:
            curr_node = max(curr_node.children, key=self.ucb)
        return curr_node

    def ucb(self, node):
        """Given a node, calculate its UCB1 score"""
        if node.num_visited == 0:
            return math.inf
        exploit = node.total_value/node.num_visited
        explore = (math.log(node.parent.num_visited)/node.num_visited)**0.5
        return exploit + self.explore_weight*explore

    def expand(self, node):
        """Given a node, populate its children attribute with possible actions"""
        temp_env = OthelloEnv(node.state, node.turn)
        actions = temp_env.possible_actions()
        for action in actions:
            state, reward, done, turn = temp_env.step(action)
            node.children.append(Node(state, turn, parent=node, action=action))
            temp_env.pop()
        return node.children[0]  # Returns the first child

    def rollout(self, state, turn):
        """Given a state and a player turn, make random moves until the game is finished"""
        temp_env = OthelloEnv(state, turn)
        done = temp_env.check_game_over()[1]
        while not done:
            actions = temp_env.possible_actions()
            state, reward, done, turn = temp_env.step(random.choice(actions))
        return temp_env.reward  # TODO: Reflect magnitude of victory given a turn

    def backprop(self, node, result):
        """Given a node and a result, update the total value and visit count for every parent node"""
        pointer = deepcopy(node)
        while pointer:
            pointer.num_visited += 1
            pointer.total_value += result
            pointer = pointer.parent

    def pi(self):
        pass


class Node():
    def __init__(self, state, turn, parent=None, action=None):
        """Each node in the tree stores state, player turn, parent, action, total value, visit count, and children"""
        self.state = state
        self.turn = turn
        self.parent = parent
        self.action = action
        self.total_value = 0
        self.num_visited = 0
        self.children = []

