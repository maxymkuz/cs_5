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
from treeSearch import treeSearch


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
    # fringe = None  # use corresponding data structure implemented in util.py
    # return treeSearch(problem, fringe)

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    print('\n\n\n')
    start_node = problem.getStartState()
    if problem.isGoalState(start_node):
        return []

    stk = util.Stack()
    visited = set()

    stk.push((start_node, []))

    while not stk.isEmpty():
        curr, actions = stk.pop()
        if curr not in visited:
            visited.add(curr)

            if problem.isGoalState(curr):
                return actions

            for next_pos, action, _ in problem.getSuccessors(curr):
                next_step = actions + [action]
                stk.push((next_pos, next_step))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    start_node = problem.getStartState()
    if problem.isGoalState(start_node):
        return []

    queue = util.Queue()
    visited = set()

    queue.push((start_node, []))

    while not queue.isEmpty():
        node, actions = queue.pop()

        if node not in visited:
            visited.add(node)

            if problem.isGoalState(node):
                return actions

            for next_pos, action, _ in problem.getSuccessors(node):
                next_step = actions + [action]
                queue.push((next_pos, next_step))


def greedSearch(problem):
    """Search the node that has the lowest heuristic first."""
    def func(state):
        # !!! Your code here
        return None

    fringe = util.PriorityQueueWithFunction(func)
    return treeSearch(problem, fringe)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    def func(state):
        # !!! Your code here
        return None

    start_node = problem.getStartState()
    if problem.isGoalState(start_node):
        return []

    visited = set()

    fringe = util.PriorityQueueWithFunction(func)
    return treeSearch(problem, fringe)


def aStarSearch(problem):
    """Search the node that has the lowest combined cost and heuristic first."""
    def func(state):
        # !!! Your code here
        return None

    fringe = util.PriorityQueueWithFunction(func)
    return treeSearch(problem, fringe)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
greed = greedSearch
