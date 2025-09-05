"""
All skills for the burger environment. Each skill take the strings as arguments and directly execute on the robotouille env.

Object
|---item: lettuce1, patty1, etc.
|---station: cuttingboard1, stove1, etc.
"""
import datetime
import os
import string
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from robotouille.robotouille.env import RobotouilleEnv
from backend.object import Object
from backend.state import State
from robotouille.utils.helper_functions import save_to_file, load_from_file

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
                stack_number[item] = 0
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
        if self.held_item: # Already holding an item
            return False
        # If the object is not on top of the stack, the skill will fail
        if self.stack_number[item] < max([self.stack_number[o] for o in self.stack_number if self.item_station[o] == self.item_station[item]]): # Stack number less than the highest one on the stack
            return False
        
        if not self._goto(item):
            return False
        action_str = "pick-up-item" if self.stack_number[item] == 0 else "unstack"
        valid_action, _ = self.env.current_state.get_valid_actions_and_str()
        valid_action = [a for a in valid_action if action_str in a[0].name]
        if valid_action:
            for a in valid_action:
                if item == a[1]["i1"]:
                    self.env.step([a])
                    self.stack_number, self.item_station, self.held_item = self.calculate_item_stack()
                    return True
        assert False, f"Precondition missed edge cases: {args}"
    
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
        if [i for i in self.item_station if station_name in self.item_station[i].name]: # The station is not empty
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
        assert False, f"Precondition missed edge cases: {args}"
    
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
        if not self.held_item: # No item is not being held
            return False
        elif item1 != self.held_item: # The item is not being held by the agent
            return False
        if self.stack_number[item2] < max([self.stack_number[o] for o in self.stack_number if self.item_station[o] == self.item_station[item2]]): # The second item is not on top of the stack
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
        assert False, f"Precondition missed edge cases: {args}"
    
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
        if self.held_item: # If the item is being held
            if item == self.held_item: return False
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
        assert False, f"Precondition missed edge cases: {args}"

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
        if self.held_item: # If the item is being held
            if item == self.held_item: return False
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
        assert False, f"Precondition missed edge cases: {args}"

    def execute_skill(self, skill):
        """
        Ground a skill string into actual functions for execution.
        E.g., "Pick(item1)" will be executed as self.Pick(args=("item1",))
        """
        if type(skill) is str:
            # separate the skill name from the arguments
            skill, args = skill.split("(")
            args = args[:-1] # remove the closing parenthesis
            args = tuple([arg.strip().lower() for arg in args.split(',')])
        else: # Skill type
            args = skill.params
            skill = skill.name
        # change to lowercase
        args = tuple([arg.lower() for arg in args])
        
        # execute the skill
        if skill == "Pick":
            return self.Pick(args)
        elif skill == "Place":
            return self.Place(args)
        elif skill == "Stack":
            return self.Stack(args)
        elif skill == "Cut":
            return self.Cut(args)
        elif skill == "Cook":
            return self.Cook(args)
        else:
            assert False, f"Unknown skill: {skill}"

def run_skill_sequence_and_record(skill_manager: SkillManager, skill_sequence, save_path: str):
    """
    Run a skill sequence.
    skill_sequence: list[Skill]
    """
    # Use current time as save path
    time_now = datetime.datetime.now()
    dir_name = str(time_now.year) + "-" + str(time_now.month) + "-" + str(time_now.day) + "-" + str(time_now.hour) + "-" + str(time_now.minute) + "-" + str(time_now.second)
    img_save_path = save_path + dir_name
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    
    # Init state
    transitions = {}
    file_name = f"{img_save_path}/0.jpg"
    render_img(skill_manager.env, skill_manager.env.current_state, file_name)
    transitions[str(0)] = {
        'skill': None,
        'image': file_name[3:],
        'success': None
    }
    # After each skill execution
    for i, skill in enumerate(skill_sequence):
        file_name = f"{img_save_path}/{i+1}.jpg"
        suc = skill_manager.execute_skill(skill)
        render_img(skill_manager.env, skill_manager.env.current_state, file_name)
        transitions[str(i+1)] = {
            'skill': skill,
            'image': file_name[3:], # remove ../
            'success': suc
        }
    
    # if log file exists, merge new data
    task_log_fpath = save_path + "/tasks.yaml"
    if os.path.exists(task_log_fpath):
        task_log = load_from_file(task_log_fpath)
    else:
        task_log = {}
    task_log[dir_name] = transitions
    save_to_file(task_log, task_log_fpath)
    
    return file_name
        

# Rendering function
def get_env_asset_path(asset_name: str, assert_exists: bool = True) -> str:
    """Return the absolute path to env asset."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, "envs", "assets")
    path = os.path.join(asset_dir_path, asset_name)
    if assert_exists:
        assert os.path.exists(path), f"Env asset not found: {asset_name} under {asset_dir_path}."
    return path
def render_img(env: RobotouilleEnv, state: State, file_name=None):
    '''
    Rendering function separated from qt interface
    '''
    name_to_img = {
            'topbun':
            mpimg.imread(get_env_asset_path("imgs/top_bun.png")),
            'bottombun':
            mpimg.imread(get_env_asset_path("imgs/bottom_bun.png")),
            'cheese':
            mpimg.imread(get_env_asset_path("imgs/cheese.png")),
            'lettuce':
            mpimg.imread(get_env_asset_path("imgs/uncut_lettuce.png")),
            'lettuce_cut':
            mpimg.imread(get_env_asset_path("imgs/cut_lettuce.png")),
            'patty':
            mpimg.imread(
                get_env_asset_path("imgs/realistic_patty_full.png")),
            'patty_cooked':
            mpimg.imread(
                get_env_asset_path("imgs/realistic_patty_full_cooked.png"))
        }
    
    def get_item_from_name(item_name):
        if 'patty' in item_name:
            img = name_to_img['patty_cooked'] if item_name in cooked_items else name_to_img['patty']
        elif 'lettuce' in item_name:
            img = name_to_img['lettuce_cut'] if item_name in cut_items else name_to_img['lettuce']
        else:
            img = name_to_img[item_name.rstrip(string.digits)]
        return img
    
    cut_items = []
    cooked_items = []

    layout = env.renderer.layout
    num_cols, num_rows = len(layout[0]), len(layout)
    figsize = (num_cols * 2, num_rows * 2)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=216)
    fontsize = 14

    # Plot vertical lines
    for i in range(num_cols + 1):
        ax.axvline(x=i, color="k", linestyle="-")

    # Plot horizontal lines
    for i in range(num_rows + 1):
        ax.axhline(y=i, color="k", linestyle="-")

    # Plot robot
    # player_pose = defaultdict(dict) # store the location and directions just in case
    vec2dir = {
        (-1, 0): "left",
        (1, 0): "right",
        (0, 1): "up",
        (0, -1): "down"
    }

    players: list[Object] = state.get_players()
    for player in players:
        player_pos = None
        held_item_name = None
        for literal, is_true in state.predicates.items():
            if is_true and literal.name == "loc" and literal.params[0].name == player.name:
                player_station = literal.params[1].name
                station_pos = env.renderer.canvas._get_station_position(player_station)
                player_pos = env.renderer.canvas.player_pose[player.name]["position"]
                player_pos, player_direction = env.renderer.canvas._move_player_to_station(player_pos, tuple(station_pos), layout)
                env.renderer.canvas.player_pose[player.name] = {"position": player_pos, "direction": player_direction}

                x, y = player_pos
                x, y = x, num_rows - y -1 # NOTE: y-axis is flipped in matplotlib for everything
                robot_img = mpimg.imread(
                    get_env_asset_path(f"imgs/robot_{vec2dir[player_direction]}.png"))
                img_size = (0.7, 0.7)
                ax.imshow(robot_img,
                        extent=[
                            x + (1 - img_size[0]) / 2, x + (1 + img_size[0]) / 2,
                            y + (1 - img_size[1]) / 2, y + (1 + img_size[1]) / 2
                        ])
                if True: # captions underneath
                    ax.text(x + 1 / 2,
                            y + (1 - img_size[1]) / 2,
                            player.name,
                            fontsize=fontsize,
                            color="red",
                            ha="center",
                            va="top",
                            bbox=dict(facecolor="black",
                                    alpha=0.5,
                                    boxstyle="square,pad=0.0"))
            # store the name item on robot if holding any
            if is_true and literal.name == "has_item" and literal.params[0].name == player.name:
                # player_pos = self.player_pose[player.name]["position"]
                held_item_name = literal.params[1].name

            # store cut item and cooked item
            if is_true and literal.name == "iscooked":
                cooked_items.append(literal.params[0].name)
            if is_true and literal.name == "iscut":
                cut_items.append(literal.params[0].name)

        # Plot grill, cutting board
        img_size = (0.7, 0.7)
        for i, row in enumerate(layout):
            for j, col in enumerate(row):
                if col is not None:
                    draw = False
                    x, y = j, num_rows - i - 1
                    if 'stove' in col:
                        img = mpimg.imread(get_env_asset_path("imgs/grill.png"))
                        draw = True
                    if 'board' in col:
                        img = mpimg.imread(get_env_asset_path("imgs/cutting_board.png"))
                        draw = True
                    
                    if draw:
                        ax.imshow(img, extent=[x, x + 1, y, y + 1])
                        if True:
                            ax.text(x + 1 / 2,
                                    y + (1 - img_size[1]) / 2,
                                    col,
                                    fontsize=fontsize,
                                    color="red",
                                    ha="center",
                                    va="top",
                                    bbox=dict(facecolor="black",
                                            alpha=0.5,
                                            boxstyle="square,pad=0.0"))
        # Plot items                   
        held_img_size = (0.6, 0.6)
        img_size = (0.7, 0.7)
        
        ## Held item if any
        if held_item_name:
            offset = held_img_size[1] * (1 / 2)
            img = get_item_from_name(held_item_name)
            x, y = player_pos
            x, y = x, num_rows - y -1
            extent = [
                x + (1 - held_img_size[0]) * (1 / 2),
                x + (1 + held_img_size[0]) * (1 / 2), y + offset,
                y + held_img_size[1] + offset
            ]
            ax.imshow(img, extent=extent)
            if True:
                # If the robot is on the right edge, put text labels for
                # held items on the left side so that they don't extend past
                # the edge of the grid and make the image larger.
                if x == num_cols - 1:
                    horizontal_align = "right"
                    text_x = x + (1 - held_img_size[0]) * (1 / 2)
                else:
                    horizontal_align = "left"
                    text_x = x + (1 + held_img_size[0]) * (1 / 2)
                ax.text(text_x,
                        y + offset + held_img_size[1] / 2,
                        held_item_name,
                        fontsize=fontsize,
                        color="red",
                        ha=horizontal_align,
                        va="top",
                        bbox=dict(facecolor="black",
                                alpha=0.5,
                                boxstyle="square,pad=0.0"))

            # Calculate item stacks
        stack_list = [] # In the form (x, y) such that x is stacked on y
        stack_number = {} # Stores the item item and current stack number
        item_station = {}

        for literal, is_true in state.predicates.items():
            if is_true and literal.name == "item_on": # On top of a station
                item_name = literal.params[0].name
                stack_number[item_name] = 1
                item_station[item_name] = literal.params[1].name
                x, y = env.renderer.canvas._get_station_position(item_station[item_name])
                x, y = x, num_rows - y -1
                # Place the item slightly above the station

                extent = [
                    x + (1 - img_size[0]) * (1 / 2),
                    x + (1 + img_size[0]) * (1 / 2), y + (1 - img_size[1]) / 2,
                    y + (1 + img_size[1]) / 2
                ]
                
                img = get_item_from_name(item_name)
                ax.imshow(img, extent=extent, zorder=stack_number[item_name])

            if is_true and literal.name == 'atop': # On top of an item
                stack = (literal.params[0].name, literal.params[1].name)
                stack_list.append(stack)

            # Add stacked items
        while len(stack_list) > 0:
            i = 0
            while i < len(stack_list):
                item_above, item_below = stack_list[i]
                if item_below in stack_number:
                    stack_list.remove(stack_list[i])
                    stack_number[item_above] = stack_number[item_below] + 1
                    item_station[item_above] = item_station[item_below]
                    # Get location of station
                    for literal, is_true in state.predicates.items():
                        if is_true and literal.name == "atop" and literal.params[0].name == item_above:
                            station_pos = env.renderer.canvas._get_station_position(item_station[item_below])
                            x, y = station_pos[0], station_pos[1]
                            x, y = x, num_rows - y - 1
                            offset = 0.1 * stack_number[item_above]
                            extent = [
                                x + (1 - img_size[0]) * (1 / 2),
                                x + (1 + img_size[0]) * (1 / 2),
                                y + (1 - img_size[1]) / 2 + offset,
                                y + (1 + img_size[1]) / 2 + offset
                            ]
                            img = get_item_from_name(item_above)
                            ax.imshow(img, extent=extent, zorder=stack_number[item_above])

                            break
                else:
                    i += 1

        # Labeling
        if True:
            for item_name in stack_number:
                stack_i = {it:s for it, s in item_station.items() if s == item_station[item_name]}
                # print(stack_i)
                station_pos = env.renderer.canvas._get_station_position(item_station[item_name])
                x, y = station_pos[0], station_pos[1]
                x, y = x, num_rows - y - 1
                # On cuttingboard or grill, place item label on top
                if "stove" in item_station[item_name] or "board" in item_station[item_name]:
                    # Nothing on top
                    if len(stack_i) == 1: # Table is invisible
                        # print(item_name)
                        ax.text(x,
                            y + (1 - img_size[1]) / 2,
                            item_name,
                            fontsize=fontsize,
                            color="red",
                            ha="center",
                            va="top",
                            bbox=dict(facecolor="black",
                                    alpha=0.5,
                                    boxstyle="square,pad=0.0"))
                    # More than 1 item in the stack
                    else:
                        ax.text(x,
                                y + (0.1 * stack_number[item_name]) + (1 - img_size[1]) / 2,
                                item_name,
                                fontsize=fontsize,
                                color="red",
                                ha="center",
                                va="top",
                                bbox=dict(facecolor="black",
                                        alpha=0.5,
                                        boxstyle="square,pad=0.0"))
                # No station below, place item label underneath
                else:
                    # Nothing on top or bottom
                    if len(stack_i) == 1: # table is invisible
                        ax.text(x + 1 / 2,
                                        y + (1 + img_size[1]) / 2,
                                        item_name,
                                        fontsize=fontsize,
                                        color="red",
                                        ha="center",
                                        va="bottom",
                                        bbox=dict(facecolor="black",
                                                alpha=0.5,
                                                boxstyle="square,pad=0.0"))
                    else:
                        ax.text(x,
                                y + (0.1 * stack_number[item_name]) + (1 - img_size[1]) / 2,
                                item_name,
                                fontsize=fontsize,
                                color="red",
                                ha="left",
                                va="bottom",
                                bbox=dict(facecolor="black",
                                        alpha=0.5,
                                        boxstyle="square,pad=0.0"))
                    

        # Draw background
        floor_img = mpimg.imread(
            get_env_asset_path("imgs/floorwood.png"))
        for y in range(num_rows):
            for x in range(num_cols):
                ax.imshow(floor_img, extent=[x, x + 1, y, y + 1], zorder=-1)

        ax.set_xlim(0, num_cols)
        ax.set_ylim(0, num_rows)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    else:
        plt.savefig("my_plot.jpg")
    plt.close('all')
    # return fig
