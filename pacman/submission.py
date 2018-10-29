from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent
import pdb

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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
    self.debug = False
    self.calledBefore = False

  def postProcess(self, value, action):
    """
    Post-processing of values and actions.
    """
    if self.debug and not self.calledBefore:
      self.calledBefore = True
      print("Value of initial state is {} with action {}.".format(value, action))

  def GeneralizedTreeSearch(self, state, player, max_depth, alpha=None, beta=None, pacman_reducer=max, ghost_reducer=min):
    """
    Computes the value of state using minimax algorithm with a maximum-depth and alpha beta pruning. 

    Args:
      state: The current state for which we're computing the value.
      max_depth: The maximum full-round plays to execute before using the evaluation function.
      player: the player for which to compute the value.
      alpha: the lower bound on the value of the max node. We stop expanding the actions in a min
        node as soon as we find a value <= alpha, since we know that the max node above will just choose
        alpha in this case.
      beta: the upper bound on the value of the min node. We stop expanding the actions in a max
        node as soon as we find a value >= beta, since we know that the min node above will just choose
        beta as the value in this case.
        If None, no pruning is applied.
      pacman_reducer: The reducer function f((currValue, currAction), (accValue, accAction)) -> (accValue, accAction)
        to reduce Pacman values computed for future states at each iteration. On the first iteration, accValue is -inf
        and accAction is None.
      ghost_reducer: The reducer function f((currValue, currAction), (accValue, accAction)) -> (accValue, accAction)
        to reduce Ghost values computed for future states at each iteration. On the first iteration, accValue is inf
        and the accAction is None.
      TODO(luis): Note that alpha/beta pruning is not supported with anything other than min/max reducers.
    """
    legalActions = state.getLegalActions(player)
    if state.isWin() or state.isLose() or not legalActions or max_depth == 0:
      return (self.evaluationFunction(state), None)
    nextPlayerIndex = (player + 1) % state.getNumAgents()
    nextDepth = (max_depth - 1) if player == (state.getNumAgents() - 1) else max_depth
    optimizer, optimalValue = (pacman_reducer, -float("inf")) if player == self.index else (ghost_reducer, float("inf"))
    # For max players, we prune using beta and need to update alpha (with new max).
    # For min players, we prune using alpha and need to update beta (with new min)
    pruneThreshold, thresholdToUpdate = (beta, alpha) if player == self.index else (alpha, beta)
    optimalAction = None
    for action in legalActions:
      nextState = state.generateSuccessor(player, action)
      nextStateValue, _ = self.GeneralizedTreeSearch(
        nextState, nextPlayerIndex, nextDepth,
        thresholdToUpdate if player == self.index else alpha,
        thresholdToUpdate if player != self.index else beta,
        pacman_reducer=pacman_reducer, ghost_reducer=ghost_reducer)
      optimalValue, optimalAction = optimizer((nextStateValue, action), (optimalValue, optimalAction))
      # Note here that for alpha, max we have value >= beta implies max(value, beta) == value
      # Similarly, for beta, min we have value <= alpha implies min(value, alpha) == value
      # As such, this is a condition where we immediately prune.
      if pruneThreshold is not None and optimizer(pruneThreshold, optimalValue) == optimalValue:
        return (optimalValue, optimalAction)
      # We update our threshold to update. alpha in the max case, and beta in the min case, with the new
      # tighter bound based on the optimal value.
      thresholdToUpdate = optimizer(thresholdToUpdate, optimalValue) if thresholdToUpdate is not None else None
    return (optimalValue, optimalAction)
    

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    value, action = self.GeneralizedTreeSearch(
      state=gameState, player=self.index, max_depth=self.depth)
    self.postProcess(value, action)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    value, action = self.GeneralizedTreeSearch(
      state=gameState, player=self.index, max_depth=self.depth,
      alpha=-float("inf"), beta=float("inf"))
    self.postProcess(value, action)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    def streamingAverage(curr, acc):
      """
      Computes the streaming average for the ghosts to be used in expectimax search. We overload the
      unused "action" parameter in the tuple to keep count of the number of actions, which we use to
      then compute the streamingAverage over the returned values.
      """
      currValue, _ = curr

      streamingAverage, numActions = acc
      # This happens when we receive our very first action for this round.
      if numActions is None:
        numActions = 0
        streamingAverage = 0.0
      numActions += 1
      newAverage = (streamingAverage * numActions + currValue) / float(numActions)
      return (newAverage, numActions)

    value, action = self.GeneralizedTreeSearch(
      state=gameState, player=self.index, max_depth=self.depth,
      ghost_reducer=streamingAverage)
    self.postProcess(value, action)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: The idea behind the below is as follows.

    Losing is always avoided, significantly, by heavily penalizing all such states.
    Winning is heavily rewarded (above just a simple score), so that PacMan always tries to win.

    For everything else, we use BFS to compute distances to food items, scared ghosts, and
    normal ghosts from pacman. We then use a simply formula (arrived at after experimentation)
    to combine these features into a single store.

    Intuitively, here are the properties we're generally looking for:
      1. A higher game score implies a higher evaluation score.
      2. Closer food items imply higher scores.
      3. A lower number of capsules implies a higher score.
      4. A lower number of food items left implies a higher score.
      5. Closer scared ghosts imply a higher score (because we can eat them)
      6. Further active ghosts imply a higher score (because they won't eat us)
  """

  # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
  SCARED_TIMER = 40
  walls = currentGameState.getWalls()
  food = currentGameState.getFood()
  capsules = set(currentGameState.getCapsules())
  scaredGhosts = {util.nearestPoint(ghost.getPosition()) : ghost.scaredTimer
    for ghost in currentGameState.getGhostStates() if ghost.scaredTimer}
  adversarialGhosts = {ghost.getPosition() for ghost in currentGameState.getGhostStates() if not ghost.scaredTimer}
  def greedyApproachCost(pacmanPos, eaten, reward=10, is_edible=lambda x,y: food[x][y], maxRestarts=None):
    """If we were to take a greedy route, ignoring ghosts, what is our cost"""
    def end():
      if len(eaten.values()) > 0:
        costs, rewards = zip(*eaten.values())
        return sum(costs) - sum(rewards)
      return 0
    if maxRestarts is not None and maxRestarts == 0:
      return end()
    queue = util.Queue()
    queue.push((pacmanPos, 0))
    visited = {}
    while not queue.isEmpty():
      (x,y), distance = queue.pop()
      if (x,y) not in eaten and is_edible(x,y):
        eaten[(x,y)] = (distance, reward)
        return greedyApproachCost(pacmanPos=(x,y), eaten=eaten,
          reward=reward, is_edible=is_edible, maxRestarts=maxRestarts if maxRestarts is None else maxRestarts - 1)
          
      visited[(x,y)] = distance
      actions = Actions.getLegalNeighbors((x,y), walls)
      random.shuffle(actions)
      for neighbor in actions:
        if neighbor not in visited:
          queue.push((neighbor, distance + 1))
        
    return end()

  def minDistanceToAllReachablePoints(pos):
    """Basically BFS to find how many steps it'll take to get places.
    
    Returns a function that given a loc returns the minimum distance from pos to loc.
    """
    queue = util.Queue()
    queue.push((pos, 0))
    visited = {}
    while not queue.isEmpty():
      curr, distance = queue.pop()
      visited[curr] = distance
      for neighbor in Actions.getLegalNeighbors(curr, walls):
        if neighbor not in visited:
          queue.push((neighbor, distance + 1))
    return lambda loc: visited[util.nearestPoint(loc)]

  # distanceFunction = lambda loc: util.manhattanDistance(pos, loc)
  distanceFunction = minDistanceToAllReachablePoints(currentGameState.getPacmanPosition())
  def winEvaluationScore():
    greedyScore = scoreEvaluationFunction(currentGameState)
    greedyScore += sum({200 - distanceFunction(pos) for pos in scaredGhosts})
    #if len(adversarialGhosts) == 2:
      # Punish if the ghosts get too close to each other.
    #  ghosts = list(adversarialGhosts)
    #  greedyScore -= (greedyScore / 10.0) / (1 + minDistanceToAllReachablePoints(ghosts[0])(ghosts[1]))
    greedyScore -= greedyApproachCost(pacmanPos=currentGameState.getPacmanPosition(), eaten={}, maxRestarts=None)
    return greedyScore

  def loseEvaluationScore():
    if not adversarialGhosts: return -1000
    closestGhost = min([distanceFunction(pos) for pos in adversarialGhosts])
    # The closer the ghost, the more we want this state.
    return -closestGhost

  # Compute an upperbound on our score.
  maxFromFood = 10 * food.count()
  maxForWin = 500
  maxForKillingGhosts = 400*len(currentGameState.getCapsules()) + 200*len(scaredGhosts)
  maxPossibleScore = scoreEvaluationFunction(currentGameState) + maxFromFood + maxForWin + maxForKillingGhosts

  # We know we can normally do well. Decide if we want to win our lose at this point.
  #if maxPossibleScore > 1200:
  return winEvaluationScore()
  #return loseEvaluationScore()
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
