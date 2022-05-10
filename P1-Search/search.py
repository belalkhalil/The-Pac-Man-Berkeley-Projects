# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """    
    "*** YOUR CODE HERE ***"
    # initialized frontier (as a stack), expanded nodes and push beginner state which contains
    # coordinates and a list of actions (to get to this coordinate from the beginner state)
    frontier = util.Stack()
    expanded_nodes = set()
    beginnerState = [problem.getStartState(), []]
    frontier.push(beginnerState) 

    # iterate through graph by choosing top node safed in the stack to perform a DFS. Stop search 
    # when the goal state is found and return the path to the goal state or
    # when the frontier is empty and no goal state has been found. Only adds a a child node from the successor 
    # (with the information mentioned above) to the stack when it wasn't already expanded (is part of the expanded_nodes set)
    while True:
        if frontier.isEmpty():
            raise NameError("failure")
        node, path = frontier.pop()
        if problem.isGoalState(node):
            return path  
        expanded_nodes.add(node)
        for successor in problem.getSuccessors(node):
            child = successor[0]
            direction = successor[1]
            if child not in expanded_nodes:
                path_to_child = path + [direction]
                frontier.push([child, path_to_child]) 

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # initialized frontier (as a queue), expanded nodes and push beginner state which contains
    # coordinates and a list of actions (to get to this coordinate from the beginner state)
    frontier = util.Queue()
    expanded_nodes = list()
    beginnerState = [problem.getStartState(), []]
    frontier.push(beginnerState)

    # iterate through graph by choosing first node safed in the queue to perform a BFS. Stop search 
    # when the goal state is found and return the path to the goal state or
    # when the frontier is empty and no goal state has been found. Only adds a a child node from the successor
    # (with the information mentioned above)to the queue when it wasn't already expanded 
    # (is part of the expanded_nodes set)
    while True:
        if frontier.isEmpty():
            raise NameError("failure")
        node, path = frontier.pop()
        if problem.isGoalState(node):
            return path  
        expanded_nodes.append(node)
        for successor in problem.getSuccessors(node):
            child = successor[0]
            direction = successor[1]
            if child not in expanded_nodes:
                expanded_nodes.append(child)
                path_to_child = path + [direction]
                frontier.push([child, path_to_child]) 

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # initialized frontier (as a priority queue), expanded nodes and push beginner state which contains
    # [coordinates, a list of actions (to get to this coordinate from the beginner state) and the costs to
    # get to the coordinate], as well as the costs g(n) for the priority queue. 
    # g(n) = the cost of reaching n from the initial state in the search.
    frontier = util.PriorityQueue()
    expanded_nodes = set()
    beginnerState = [problem.getStartState(), [], 0]
    frontier.push(beginnerState, 0)

    # iterate through graph by choosing first node safed in the priority queue (node with the 
    # least cost) to perform a UCS. Stop search when the goal state is found and return the 
    # path to the goal state or when the frontier is empty and no goal state has been found. Only adds 
    # a a child node from the successor (with the information mentioned above) to the priority queue when it wasn't
    # already expanded (is part of the expanded_nodes set)
    while True:
        if frontier.isEmpty():
            raise NameError("failure")
        node, path, cost = frontier.pop()
        if problem.isGoalState(node):
            return path  
        if node not in expanded_nodes:
            expanded_nodes.add(node)
            for successor in problem.getSuccessors(node):
                child = successor[0]
                if child not in expanded_nodes:
                    direction = successor[1]
                    path_to_child = path + [direction]
                    cost_to_child = cost + successor[2]
                    frontier.push([child, path_to_child, cost_to_child], cost_to_child) 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # initialized frontier (as a priority queue), expanded nodes and push beginner state which contains
    # [coordinates, a list of actions (to get to this coordinate from the beginner state) and the costs to
    # get to the coordinate], as well as the costs for the priority queue (defined as f(n) = g(n) + h(n)).
    # g(n) = the cost of reaching n from the initial state in the search. 
    # h(n) = a heuristic value of n, that is the estimated cost of reaching a goal from n
    frontier = util.PriorityQueue()
    expanded_nodes = []
    beginnerState = [problem.getStartState(), [], 0]
    frontier.push(beginnerState, 0)

    # iterate through graph by choosing first node safed in the priority queue (node with the 
    # least cost) to perform a A* search. Stop search when the goal state is found and return the 
    # path to the goal state or when the frontier is empty and no goal state has been found. Only adds 
    # a child node from the successor (with the information mentioned above) to the priority queue when 
    # it wasn't already expanded (is part of the expanded_nodes set)
    while True:
        if frontier.isEmpty():
            raise NameError("failure")
        node, path, cost = frontier.pop()
        if problem.isGoalState(node):
            return path  
        if node not in expanded_nodes:
            expanded_nodes.append(node)
            for successor in problem.getSuccessors(node):
                child = successor[0]
                if child not in expanded_nodes:
                    direction = successor[1]
                    path_to_child = path + [direction]
                    cost_to_child = cost + successor[2]
                    frontier.push([child, path_to_child, cost_to_child], cost_to_child + heuristic(child, problem)) 

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
