# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # simple implementation ignores ghost states as thier behaviour is randomized 
        # and focuses on getting to the next closest available Food
        # code passed all test cases

        #get list of all available food
        availableFood = newFood.asList()
        # get shortest distance to each available food using manhattanDistance from util
        manhattanFoodDistance = [util.manhattanDistance(food, newPos) for food in availableFood]
        
        # if there is still food
        if manhattanFoodDistance:
            # get the nearest food available
            nearestFood = min(manhattanFoodDistance)
        else:
            # no food available, all food is eaten. nearestFood could be any value aside from zero
            nearestFood = -100

        # get the reciprocal value of the distance of the next closest food
        moveToward = 1 / nearestFood
        
        # add this value to the score
        return successorGameState.getScore() + moveToward 

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1 

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(0, self.depth, gameState)[1]

    def minimax(self, agent, depth, gameState):
        # Pacman is always agent 0, and the agents move in 
        # order of increasing agent index. If it reaches the last agent
        # it will start with Pacman again.   
        if agent + 1 == gameState.getNumAgents():
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agent + 1, depth

        # Check if out of possible depth or a terminal state is reached.    
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), 'Stop'

        # (Pacman) maximizes and returns a list with the structure [v, action]. 
        # Internally it's maximizing v which its initialized as -inf and then updated recursively
        # return is a list to also return the action related to v (needed in getAction())
        if agent == 0:
            v = [float('-inf'), None]
            for action in gameState.getLegalActions(agent):
                next_state = gameState.generateSuccessor(agent, action)
                successor = [self.minimax(next_agent, next_depth, next_state)[0], action]
                v = max(v, successor, key=lambda x:x[0])
            return v

        # (Ghost) minimizes and returns a list with the structure [v, action]. 
        # Internally it's minimizing v which its initialized as inf and then updated recursively
        # return is a list to also return the action related to v (needed in getAction())
        else:  
            v = [float('inf'), None]
            for action in gameState.getLegalActions(agent):
                next_state = gameState.generateSuccessor(agent, action)
                successor = [self.minimax(next_agent, next_depth, next_state)[0], action]
                v = min(v, successor, key=lambda x:x[0])
            return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alpha_beta(0, self.depth, gameState, float('-inf'), float('inf'))[1]

    def alpha_beta(self, agent, depth, gameState, a, b):
        # Pacman is always agent 0, and the agents move in 
        # order of increasing agent index. If it reaches the last agent
        # it will start with Pacman again.   
        if agent + 1 == gameState.getNumAgents():
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agent + 1, depth

        # Check if out of possible depth or a terminal state is reached.  
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), 'Stop'
        # (Pacman) maximizes with alpha-beta pruning and returns a list with the structure [v, action].
        # Internally it's maximizing v which its initialized as -inf and then updated recursively
        # return is a list to also return the action related to v (needed in getAction())
        if agent == 0:
            v = [float('-inf'), None]
            for action in gameState.getLegalActions(agent):
                next_state = gameState.generateSuccessor(agent, action)
                successor = [self.alpha_beta(next_agent, next_depth, next_state, a, b)[0], action]
                v = max(v, successor, key=lambda x:x[0])
                
        # Alpha-beta pruning: Beta-Cut
                if v[0] > b: 
                    return v
                a = max(a,v[0])
            return v

        else:  
        # (Ghost) minimizes with alpha-beta pruning and returns a list with the structure [v, action]. 
        # Internally it's minimizing v which its initialized as inf and then updated recursively
        # return is a list to also return the action related to v (needed in getAction())
            v = [float('inf'), None]
            for action in gameState.getLegalActions(agent):
                next_state = gameState.generateSuccessor(agent, action)
                successor = [self.alpha_beta(next_agent, next_depth, next_state, a, b)[0], action]
                v = min(v, successor, key=lambda x:x[0])

        # Alpha-beta pruning: Alpha-Cut
                if v[0] < a: 
                    return v
                b = min(b, v[0])
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return  self.expectimax(0, 0, gameState)[1]                         #Calling expectimax for PacMan and the current depth is 0

    def expectimax(self, agent,  depth, gameState):
        if agent == gameState.getNumAgents():                               #the agents at this depth are finished, we need to restart with pacman the next level
            agent, depth = 0, depth +1                                      #next_agent is pacman, and the depth is incremented of 1 unit

        if depth >= self.depth or gameState.isLose() or gameState.isWin():   #check if the agent is out of his possible depth or we reach one of the two terminal states
            return self.evaluationFunction(gameState),None                   #returning the self.evaluationFunction
            
        
        #MAX AGENT
        if agent == 0:                                                                              #if it is Pacman, he chooses the best action among all the possible actions
            v = [float('-inf'), None]                                                               #we search for the max value, so we set the initial value to - infinite
            for action in gameState.getLegalActions(agent):                                         #we iterate for each possible actions it could make
                next_state = gameState.generateSuccessor(agent, action)                             #we create the new gameState, after Pacman move.
                successor = [self.expectimax(agent +1, depth, next_state)[0], action]               #for each successor of the state we perform expectimax
                v = max(v, successor, key=lambda x:x[0])
            return v

        #EXP AGENT
        else:
            v = [float('0'), None]
            exp_score = v[0]                                                                        #expected value of the successot
            for action in gameState.getLegalActions(agent):                                         #iterate among each possible action
                number_of_actions = float(len(gameState.getLegalActions(agent)))                    #number of possible actions
                next_state = gameState.generateSuccessor(agent, action)                             #state after agent did the action
                successor = [float(self.expectimax(agent + 1, depth, next_state)[0]), action]       #for each successor we call expectimax
                exp_score += (successor[0]/number_of_actions)                                       #adding component to the expected value


            v = [exp_score, None]                                                                   #compute v -> it's value now is the expected value of its successors  
            return v

    
    
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    '''
    We evaluate in different terms the distance from food, the distance from ghosts, the gameScore...
    and we assign a different weight for every term, to be able to assign a "priority" to each of them 
    '''

    
    newPacPos = currentGameState.getPacmanPosition()                                                            #next PacMan position 
    availableFood = currentGameState.getFood().asList()                                                         #take all the food still existing 
    next_ghosts_scared_timers = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]    #storing all the scaredTimer for each ghost 


    closest_food_distance = min( manhattan_distance(newPacPos, nextFood) for nextFood in availableFood) if availableFood else 0        #taking the minimum value, corrisponding to the manhattan distance to the closest food
    ghost_distance = min( manhattan_distance(newPacPos, ghost.configuration.pos) for ghost in currentGameState.getGhostStates())       #taking the minimum value, corrisponding to the manhattan distance to the closest ghost
    scared_time = min(next_ghosts_scared_timers)



    game_score = currentGameState.getScore() * 0.5                                                             #if the gameState is getting higher, it is better
    
    remaining_food_feature = 1/(len(availableFood)+0.1)                                                         #taking the reciproce of the number of remaining food (+0.1, so we couldn't devide by zero, instead of using an if statement)
    closest_food_feature = 0.2 / (closest_food_distance + 0.1)                                                  #we need to be close as possible to food   (for a greather distance, we will have a small value)

    
    ghost_distance_feature = - 1/ (ghost_distance + 0.1) if scared_time == 0 else 0.3 / (ghost_distance + 0.1)  #we need to be as far as possible from ghosts if they are not scared 
    power_pellets_feature = scared_time * 0.5                                                                   #it is good to try to eat ghosts if they are scared, but it's not always worth it 
    
    
    
    

    #sum every weighted term to return the evaluation function value
    return game_score + remaining_food_feature  + closest_food_feature + ghost_distance_feature + power_pellets_feature 


# Abbreviation
better = betterEvaluationFunction
    