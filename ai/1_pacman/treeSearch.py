def treeSearch(problem, fringe):
    visited = dict()

    initial_state = problem.getStartState()
    previous_state = None
    initial_action = None
    initial_cost = 0

    fringe.push((previous_state, initial_state, initial_action, initial_cost))

    while not fringe.isEmpty():
        previous_state, state, action, state_cost = fringe.pop()

        if state in visited:
            continue

        visited[state] = (previous_state, action)

        if problem.isGoalState(state):
            solution = []

            # !!! Your code here

            return solution

        for new_state, action, action_cost in problem.getSuccessors(state):
            if new_state not in visited:
                fringe.push((state, new_state, action, state_cost + action_cost))

    return []
