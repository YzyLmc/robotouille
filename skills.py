"""
All skills for the burger environment. Each skill take the strings as arguments and directly execute on the robotouille env.

Object
|---item: lettuce1, patty1, etc.
|---station: cuttingboard1, stove1, etc.
"""

from robotouille.env import RobotouilleEnv
from backend.object import Object
from backend.state import State

class SkillManager:
    """
    Take in a robotouille env, record the items and their locations (stack order) in the environment.
    """
    def __init__(self, env: RobotouilleEnv):
        self.env = env
        self.stack_number, self.item_station, self.held_item = self.calculate_item_stack()
        self.objects: list[Object] = self.env.current_state.objects

    def calculate_item_stack(self):
        """
        Calculate item stacks and store in dictionaries. 
        The first dictionary stores the stack number for each item, and the second dictionary stores the station each item is at.
        """
        state: State = self.env.current_state
        ## Calculate item stacks
        stack_list = [] # In the form (x, y) such that x is stacked on y
        stack_number: dict[Object, int] = {} # Stores the item item and current stack number
        item_station: dict[Object, Object] = {}
        for literal, is_true in state.predicates.items():
            if is_true and literal.name == "item_on": # On top of a station
                item = literal.params[0]
                stack_number[item] = 1
                item_station[item] = literal.params[1]
            if is_true and literal.name == 'atop': # On top of an item
                stack = (literal.params[0], literal.params[1])
                stack_list.append(stack)
            if is_true and literal.name == "has_item":
                held_item = literal.params[1]
        
        while len(stack_list) > 0:
            i = 0
            while i < len(stack_list):
                item_above, item_below = stack_list[i]
                if item_below in stack_number:
                    stack_list.remove(stack_list[i])
                    stack_number[item_above] = stack_number[item_below] + 1
                    item_station[item_above] = item_station[item_below]
                else:
                    i += 1

        return stack_number, item_station, held_item

    def _goto(self, object: Object):
        """
        Go to a location that enables interaction with the object.
        This function is not a skill per se, but is called by other skills
        """
        # Find the object location
        item_station: Object = self.item_station[object]

        # Find the action from available actions
        action_str = "move"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        ## Look for valid parameter combinations
        for a in valid_action:
            if item_station == a[1]["s2"]:
                self.env.step(a)
                return True
        assert False, "This should never happen since the agent is free to move to anywhere anytime"

    # Pick, Place, and Stack will change the stack_number and item_station
    def Pick(self,  args: tuple[str]):
        """
        Pick up the item regardless of its station.
        Precondition:
        - The item is on top of the stack
        """
        item_name: str = args[0]
        item: Object = [i for i in self.objects if i.name == item_name][0]
        # If the object is not on top of the stack, the skill will fail
        if self.stack_number[item] < max([self.stack_number[o] for o in self.stack_number if self.item_station[o] == self.item_station[item_name]]): # Stack number less than the highest one on the stack
            return False
        if self.held_item: # Already holding an item
            return False
        
        if not self._goto(item):
            return False
        action_str = "pick-up-item"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        if valid_action:
            for a in valid_action:
                if item == a[1]["i1"]:
                    self.env.step(a)
                    self.stack_number, self.item_station, self.held_item = self.calculate_item_stack()
                    return True
        assert False, "Precondition missed edge cases"
    
    def Place(self, args: tuple[str]):
        """
        Go to the location of the station, and place the item on top.
        Precondition:
        - The station is empty.
        - The item is being hold by the agent.
        """
        item_name, station_name = args
        item: Object = [i for i in self.objects if i.name == item_name][0]
        if [i for i in self.objects if self.item_station[i] == station_name]: # The station is not empty
            return False
        if not self.held_item: # No item is not being held
            return False
        elif item_name not in self.held_item.name: # The item is not being held by the agent
            return False

        if not self._goto(item):
            return False
        action_str = "place-item"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        if valid_action:
            for a in valid_action:
                if item == a[1]["i1"] and station_name in a[1]["s1"].name:
                    self.env.step(a)
                    self.stack_number, self.item_station = self.calculate_item_stack()
                    return True
        assert False, "Precondition missed edge cases"
    
    def Stack(self, args: tuple[str]):
        """
        Go to the location of the second item, and place the first item on top.
        Precondition:
        - The second item is on top of the stack.
        - The first item is being held by the agent.
        """
        item1_name, item2_name = args
        item1: Object = [i for i in self.objects if i.name == item1_name][0]
        item2: Object = [i for i in self.objects if i.name == item2_name][0]
        if self.stack_number[item2] < max([self.stack_number[o] for o in self.stack_number if self.item_station[o] == self.item_station[item2_name]]): # The second item is not on top of the stack
            return False
        if not self.held_item: # No item is not being held
            return False
        elif item1 != self.held_item: # The item is not being held by the agent
            return False

        if not self._goto(item2):
            return False
        action_str = "stack"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        if valid_action:
            for a in valid_action:
                if item1 == a[1]["i1"] and item2 == a[1]["i2"]:
                    self.env.step(a)
                    self.stack_number, self.item_station = self.calculate_item_stack()
                    return True
        assert False, "Precondition missed edge cases"
    
    # Cut and Cook will change the object state, tracked by predicates
    def Cut(self, args: tuple[str]):
        """
        Cut the item with going to its station.
        Precondition:
        - The item is on top of the cuttingboard.
        - The item is cuttable.
        - The agent is not holding anything.
        - There is nothing else on top of the item.
        """
        pass

    def Cook(self, args: tuple[str]):
        """
        Cook the item with going to its station.
        Precondition:
        - The item is on top of the stove.
        - The item is cookable.
        - There is nothing else on top of the item.
        """
        pass