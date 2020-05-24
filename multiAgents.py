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
        food = currentGameState.getFood()
        curPosition = list(successorGameState.getPacmanPosition())

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(curPosition) and (state.scaredTimer == 0):
                return float("-Inf")

        lstDistance = list(map(lambda foodPosition: -1*(manhattanDistance(curPosition, foodPosition)), foodList))

        return max(lstDistance)

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
        def minValue(state, agentIndex, depth):
            ghostActions = state.getLegalActions(agentIndex)
            if len(ghostActions) <= 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state)

            if agentIndex == state.getNumAgents() - 1:
                return min(maxValue(state.generateSuccessor(agentIndex, action), depth) for action in ghostActions)
            else:
                return min(minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in
                           ghostActions)

        def maxValue(state, depth):
            pacmanActions = state.getLegalActions(0)
            if len(pacmanActions) <= 0 or depth == self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)

            return max(minValue(state.generateSuccessor(0, action), 1, depth + 1) for action in pacmanActions)

        bestAction = max(gameState.getLegalActions(0),
            key=lambda action: minValue(gameState.generateSuccessor(0, action), 1, 1))
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        Inf = float('inf')

        def minValue(state, agentIndex, depth, a, b):
            ghostActions = state.getLegalActions(agentIndex)
            if len(ghostActions) <= 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state)

            v = Inf
            for action in ghostActions:
                ghostState = state.generateSuccessor(agentIndex, action)

                if agentIndex == state.getNumAgents() - 1:
                    value = maxValue(ghostState, depth, a, b)
                else:
                    value = minValue(ghostState, agentIndex + 1, depth, a, b)

                v = min(v, value)
                if v < a:
                    return v
                b = min(b, v)
            return v

        def maxValue(state, depth, a, b):
            pacmanActions = state.getLegalActions(0)
            if len(pacmanActions) <= 0 or depth == self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)

            v = -Inf
            if depth == 0:
                bestAction = pacmanActions[0]
            for action in pacmanActions:
                pacmanState = state.generateSuccessor(0, action)
                value = minValue(pacmanState, 0 + 1, depth + 1, a, b)
                if value > v:
                    v = value
                    if depth == 0:
                        bestAction = action
                if v > b:
                    return v
                a = max(a, v)

            if depth == 0:
                return bestAction
            return v

        bestAction = maxValue(gameState, 0, -Inf, Inf)
        return bestAction

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
        def maxValue(state, depth):
            pacmanActions = state.getLegalActions(0)
            if len(pacmanActions) <= 0 or depth == self.depth or state.isLose() or state.isWin():
                return self.evaluationFunction(state)

            v = max(expectValue(state.generateSuccessor(0, action), 0 + 1, depth + 1) for action in pacmanActions)
            return v

        def expectValue(state, agentIndex, depth):
            ghostActions = state.getLegalActions(agentIndex)
            if len(ghostActions) <= 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state)

            probability = 1.0 / len(ghostActions)
            v = 0
            for action in ghostActions:
                ghostState = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    v += maxValue(ghostState, depth) * probability
                else:
                    v += expectValue(ghostState, agentIndex + 1, depth) * probability
            return v

        pacmanActions = gameState.getLegalActions()
        bestAction = max(pacmanActions, key=lambda action: expectValue(gameState.generateSuccessor(0, action), 1, 1))
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = list(currentGameState.getPacmanPosition())
    foodPositions = currentGameState.getFood().asList()
    lstDistance = list(map(lambda foodPosition: manhattanDistance(foodPosition, pacmanPosition), foodPositions))
    return currentGameState.getScore() + 1.0/min(lstDistance) if lstDistance else currentGameState.getScore() + 2

# Abbreviation
better = betterEvaluationFunction
