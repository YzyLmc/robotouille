from robotouille.env import RobotouilleEnv

def _goto(env: RobotouilleEnv, object: str):
    """
    Go to a location that enables interaction with the object.
    The function is not a skill per se, but is called by other skills
    """
    # Find the object location
    state = env.current_state
    for literal, is_true in state.predicates.items():
            if is_true and literal.name == "item_on": # On top of a station
                item_name = literal.params[0].name
                stack_number[item_name] = 1
                item_station[item_name] = literal.params[1].name

    # Find the action from available actions

    # Execute the action


def Pick(env: RobotouilleEnv, args):
    """
    
    """

    pass

def Place():
    pass

def Stack():
    pass

def Cut():
    pass

def Cook():
    pass