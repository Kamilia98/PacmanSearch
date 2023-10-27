# search.py
# ---------

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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
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
    # Get the initial state from the problem
    start = problem.getStartState()
    print(start)

    # Check if the initial state is already the goal state
    if problem.isGoalState(start):
        return []

    # Initialize a stack for DFS and a list to keep track of visited nodes
    myQueue = util.Stack()
    visited = []

    # Push the initial node and an empty list of actions onto the stack
    myQueue.push((start, []))

    # Start the DFS loop
    while myQueue:
        # Pop the current node and its associated actions from the stack
        currentNode, actions = myQueue.pop()

        # Check if the current node has not been visited before
        if currentNode not in visited:
            visited.append(currentNode)  # Mark the node as visited
            # Check if the current node is the goal state

            if problem.isGoalState(currentNode):
                return actions  # Return the list of actions to reach the goal state

            for nextNode, action, cost in problem.getSuccessors(currentNode):
                # Create a new list of actions by appending the current action.
                newAction = actions + [action]

                # Push the next node and the new list of actions onto the stack for further exploration.
                myQueue.push((nextNode, newAction))

    util.raiseNotDefined()


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Get the starting state of the problem.
    start = problem.getStartState()

    # Check if the starting state is already a goal state.
    if problem.isGoalState(start):
        return []  # If it's a goal state, no need to search, return an empty list

    # Initialize a queue for BFS and a list to keep track of visited nodes
    myQueue = util.Queue()
    visited = []

    # Initialize the queue with the starting state and and empty list of actions
    myQueue.push((start, []))

    # Start the BFS loop.
    while myQueue:
        # Get the current node and its associated actions from the queue
        currentNode, actions = myQueue.pop()

        # Check if the current node has not been visited before
        if currentNode not in visited:
            visited.append(currentNode)  # Mark the current node as visite

            # Check if the current node is a goal state.
            if problem.isGoalState(currentNode):
                return actions  # If it's a goal state, return the list of actions

            # Explore the successors of the current node.
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                # Create a new list of actions by appending the current action.
                newAction = actions + [action]

                # Push the next node and the new list of actions onto the queue for further exploration.
                myQueue.push((nextNode, newAction))

    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Get the initial state of the problem
    start = problem.getStartState()

    # Check if the initial state is the goal state
    if problem.isGoalState(start):
        return []  # If it is, return an empty list, as we're already at the goal.

    # Create a list to keep track of visited nodes
    visited = []

    # Create a priority queue for uniform cost search
    myQueue = util.PriorityQueue()

    # Add the start node with its associated action and cost to the priority queue with a priority of 0
    myQueue.push((start, [], 0), 0)

    # Main loop for uniform cost search
    while myQueue:
        # Get the node with the lowest cost from the priority queue
        currentNode, actions, currentCost = myQueue.pop()

        # Check if the current node has been visited before
        if currentNode not in visited:
            visited.append(currentNode)  # Mark the current node as visited

            # Check if the current node is the goal state
            if problem.isGoalState(currentNode):
                return actions  # If it is, return the list of actions taken to reach the goal

            # Explore the successors of the current node
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                # Create a new list of actions by adding the current action
                newAction = actions + [action]

                # Calculate the new cost by adding the current cost
                newCost = currentCost + cost

                # The priority is set to the new cost for uniform cost search
                priority = newCost

                # Add the next node with its new action and cost to the priority queue
                myQueue.push((nextNode, newAction, newCost), priority)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Get the initial state of the problem
    start = problem.getStartState()

    # Check if the initial state is the goal state
    if problem.isGoalState(start):
        return []

    # Initialize a priority queue for the open set and a list to track visited nodes
    myQueue = util.PriorityQueue()
    visited = []

    # Push the initial state, empty actions list, and zero cost to the open set with priority 0
    myQueue.push((start, [], 0), 0)

    # Main A* search loop.
    while myQueue:
        # Pop the node with the lowest priority (lowest estimated total cost)
        currentNode, actions, currentCost = myQueue.pop()

        # Check if the current node has already been visited to avoid revisiting it
        if currentNode not in visited:
            visited.append(currentNode)

            # Check if the current node is the goal state
            if problem.isGoalState(currentNode):
                return actions

            # Explore the successors of the current node
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                # Create new action list and update the cost
                newAction = actions + [action]
                newCost = currentCost + cost

                # Calculate the estimated total cost using the heuristic function
                heuristicCost = newCost + heuristic(nextNode, problem)

                # Push the next node onto the priority queue with its heuristic cost
                myQueue.push((nextNode, newAction, newCost), heuristicCost)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
