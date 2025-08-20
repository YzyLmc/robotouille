"""
All skills for the burger environment. Each skill take the strings as arguments and directly execute on the robotouille env.

Object
|---item: lettuce1, patty1, etc.
|---station: cuttingboard1, stove1, etc.
"""
import datetime
import os

from robotouille.env import RobotouilleEnv
from backend.object import Object
from backend.state import State
from playground import render_img

import hydra
from omegaconf import DictConfig, OmegaConf
from robotouille.robotouille_env import create_robotouille_env

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
        held_item: Object = None
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

    def _goto(self, object: Object, is_station:bool=False):
        """
        Go to a location that enables interaction with the object.
        This function is not a skill per se, but is called by other skills
        """
        # Find the object location
        item_station: Object = self.item_station[object] if not is_station else object

        # Find the action from available actions
        action_str = "move"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        ## Look for valid parameter combinations
        for a in valid_action:
            if item_station == a[1]["s1"]:
                return True
            if item_station == a[1]["s2"]:
                self.env.step([a])
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
        item: Object = [i for i in self.objects if item_name in i.name][0]
        # If the object is not on top of the stack, the skill will fail
        if self.stack_number[item] < max([self.stack_number[o] for o in self.stack_number if self.item_station[o] == self.item_station[item]]): # Stack number less than the highest one on the stack
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
                    self.env.step([a])
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
        item: Object = [i for i in self.objects if item_name in i.name][0]
        station: Object = [s for s in self.objects if station_name in s.name][0]
        if [i for i in self.item_station if station_name == self.item_station[i].name]: # The station is not empty
            return False
        if not self.held_item: # No item is not being held
            return False
        elif item_name not in self.held_item.name: # The item is not being held by the agent
            return False

        if not self._goto(station, is_station=True):
            return False
        action_str = "place-item"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        if valid_action:
            for a in valid_action:
                if item == a[1]["i1"] and station_name in a[1]["s1"].name:
                    self.env.step([a])
                    self.stack_number, self.item_station, self.held_item = self.calculate_item_stack()
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
        item1: Object = [i for i in self.objects if item1_name in i.name][0]
        item2: Object = [i for i in self.objects if item2_name in i.name][0]
        if self.stack_number[item2] < max([self.stack_number[o] for o in self.stack_number if self.item_station[o] == self.item_station[item2]]): # The second item is not on top of the stack
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
                    self.env.step([a])
                    self.stack_number, self.item_station, self.held_item = self.calculate_item_stack()
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
        item_name: str = args[0]
        item: Object = [i for i in self.objects if item_name in i.name][0]
        if self.stack_number[item] < max([self.stack_number[o] for o in self.stack_number if self.item_station[o] == self.item_station[item]]): # Stack number less than the highest one on the stack
            return False
        if self.held_item: # Already holding an item
            return False
        if "board" not in self.item_station[item].name: # The item is not on top of the cuttingboard
            return False
        # The item is not cuttable
        for literal, is_true in self.env.current_state.predicates.items():
            if literal.params[0] == item and literal.name == "iscuttable":
                if not is_true:
                    return False
        
        if not self._goto(item):
            return False
        action_str = "cut"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        if valid_action:
            for a in valid_action:
                if item == a[1]["i1"]:
                    for _ in range(3):self.env.step([a]) # You somehow have to cut it three times
                    return True
        assert False, "Precondition missed edge cases"

    def Cook(self, args: tuple[str]):
        """
        Cook the item with going to its station.
        Precondition:
        - The item is on top of the stove.
        - The item is cookable.
        - There is nothing else on top of the item.
        """
        item_name: str = args[0]
        item: Object = [i for i in self.objects if item_name in i.name][0]
        if self.stack_number[item] < max([self.stack_number[o] for o in self.stack_number if self.item_station[o] == self.item_station[item]]): # Stack number less than the highest one on the stack
            return False
        if "stove" not in self.item_station[item].name: # The item is not on top of the cuttingboard
            return False
        # The item is not cookable
        for literal, is_true in self.env.current_state.predicates.items():
            if literal.params[0] == item and literal.name == "iscookable":
                if not is_true:
                    return False
                
        if not self._goto(item):
            return False
        action_str = "cook"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        if valid_action:
            for a in valid_action:
                if item == a[1]["i1"]:
                    self.env.step([a])
                    # Wait for three timesteps after start cooking
                    wait = [a for a in self.env.current_state.get_valid_actions_and_str()[0] if a[0].name == "wait"][0]
                    for _ in range(3): self.env.step([wait])
                    return True
        assert False, "Precondition missed edge cases"

    def execute_skill(self, skill: str):
        """
        Ground a skill string into actual functions for execution.
        E.g., "Pick(item1)" will be executed as self.Pick(args=("item1",))
        """
        # separate the skill name from the arguments
        skill, args = skill.split("(")
        args = args[:-1] # remove the closing parenthesis
        args = tuple(args.split(', '))

        # execute the skill
        if skill == "Pick":
            self.Pick(args)
        elif skill == "Place":
            self.Place(args)
        elif skill == "Stack":
            self.Stack(args)
        elif skill == "Cut":
            self.Cut(args)
        elif skill == "Cook":
            self.Cook(args)
        else:
            assert False, f"Unknown skill: {skill}"

def run_skill_sequence_and_record(skill_manager: SkillManager, skill_sequence: list[str], save_path: str):
    """
    Run a skill sequence.
    """
    # Use current time as save path
    time_now = datetime.datetime.now()
    save_path = save_path + str(time_now.year) + "-" + str(time_now.month) + "-" + str(time_now.day) + "-" + str(time_now.hour) + "-" + str(time_now.minute)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, skill in enumerate(skill_sequence):
        file_name = f"{save_path}/{i}.png"
        render_img(skill_manager.env, skill_manager.env.current_state, file_name)
        skill_manager.execute_skill(skill)
        
    file_name = f"{save_path}/{i+1}.png"
    render_img(skill_manager.env, skill_manager.env.current_state, file_name)

def test_roll_out(environment_name: str, **kwargs):
    '''Minimal script for testing action rollout and screen shot'''
    # Initialize environment
    seed = kwargs.get('seed', None)
    env = create_robotouille_env(environment_name, seed)
    obs, info = env.reset()

    # Initialize skill manager
    skill_manager = SkillManager(env)
    skill_sequence = [
        "Pick(lettuce)",
        "Place(lettuce, board)",
        "Cut(lettuce)",
        "Pick(patty)",
        "Place(patty, stove)",
        "Cook(patty)",
        "Pick(patty)",
        "Stack(patty, bottombun)",
        "Pick(lettuce)",
        "Stack(lettuce, patty)",
        "Pick(topbun)",
        "Stack(topbun, lettuce)"
    ]
    save_path = "test_run/"
    # Run skill sequence
    run_skill_sequence_and_record(skill_manager, skill_sequence, save_path)

@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg: DictConfig) -> None:
    if not cfg.evaluation.evaluate:
        kwargs = OmegaConf.to_container(cfg.game, resolve=True)
        kwargs['llm_kwargs'] = OmegaConf.to_container(cfg.llm, resolve=True)
        environment_name = kwargs.pop('environment_name')
        test_roll_out(environment_name, **kwargs)

if __name__ == "__main__":
    main()