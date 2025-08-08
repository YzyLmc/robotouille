import hydra
import os
import json

from omegaconf import DictConfig, OmegaConf

import math
import pygame
from typing import Dict, Any, List
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict

from agents import NAME_TO_AGENT

from utils.video_recorder import record_video
from robotouille.robotouille_env import create_robotouille_env
from robotouille.env import RobotouilleEnv
from backend.state import State
from backend.object import Object
import utils_temp

# from robotouille.robotouille_simulator import run_robotouille


def run_robotouille(environment_name: str, agent_name: str, **kwargs: Dict[str, Any]):
    """Runs the provided Robotouille environment with the given agent.

    Parameters:
        environment_name (str):
            The name of the environment to run.
            Find environment names under environments/env_generator/examples
        agent_name (str):
            The name of the agent to run. Use "human" for Pygame human input.
            Find agent names under agents/__init__.py
        kwargs (Dict[str, Any]):
            Optional parameters to run Robotouille with including:
                - seed (int):
                    The seed for the environment.
                - max_steps (int):
                    The maximum number of steps to run the environment.
                - noisy_randomization (bool):
                    Whether to use noisy randomization.
                    See environments/env_generator/README.md for more information.
                - render_mode (str):
                    The render mode to use. Can be "human" or "rgb_array".
                - record (bool):
                    Whether to record a video of the run.
                - fourcc_str (str):
                    The fourcc string to use for the video codec.
                - video_path (str):
                    The filename for the file to save the video to.
                - video_fps (int):
                    The frames per second for the video.
                - llm_kwargs (Dict[str, Any]):
                    The kwargs for the LLM agent.
                    - log_path (str):
                        The path to the log file to write to.
    
    Returns:
        done (bool):
            Whether the environment is done.
        steps (int):
            The number of steps taken in the environment.
    """
    # Initialize environment
    seed = kwargs.get('seed', None)
    noisy_randomization = kwargs.get('noisy_randomization', False)
    env = create_robotouille_env(environment_name, seed, noisy_randomization)
    renderer = env.renderer
    # Initialize agent
    llm_kwargs = kwargs.get('llm_kwargs', {})
    agent = NAME_TO_AGENT[agent_name](llm_kwargs)
    agent_done_cond = lambda a: a.is_done() if a is not None else False
    agent_retry_cond = lambda a, steps_left: a.is_retry(steps_left) if a is not None else False

    render_mode = kwargs.get('render_mode', 'human')
    record = kwargs.get('record', False)

    obs, info = env.reset()
    done = False
    steps = 0
    if kwargs.get('max_steps'):
        max_steps = kwargs.get('max_steps')
    elif kwargs.get('max_steps_multiplier'):
        agent = NAME_TO_AGENT['bfs'](None)
        optimal_plan = agent.propose_actions(obs, env)
        max_steps = math.ceil(len(optimal_plan) * kwargs.get('max_steps_multiplier'))
    else:
        assert False, "Must provide either max_steps or max_steps_multiplier in kwargs"
    imgs = []
    queued_actions = []
    stochastic_done = False
    while not done and not agent_done_cond(agent) and steps < max_steps:
        render_img(env, env.current_state)
        img = env.render(render_mode)
        if record:
            imgs.append(img)
        
        if len(queued_actions) == 0:
            # Retrieve action(s) from agent output
            proposed_actions = agent.propose_actions(obs, env)
            if proposed_actions: 
                print(proposed_actions)
                print('state', env.current_state.predicates)
            if len(proposed_actions) == 0:
                # Reprompt agent for action(s)
                continue
            action, param_arg_dict = proposed_actions[0]
            queued_actions = proposed_actions[1:]
        else:
            action, param_arg_dict = queued_actions.pop(0)
        
        # Assign action to players
        # We only have one player
        actions = []
        current_state = env.current_state
        print(current_state.get_valid_actions_and_str())
        for player in current_state.get_players():
            if player == current_state.current_player:
                actions.append((action, param_arg_dict))
            else:
                actions.append((None, None))
        
        # Step environment
        obs, reward, done, info = env.step(actions)

        if kwargs.get("stochastic") and not stochastic_done and random.random() < 0.1:
            # Randomly set one cut ingredient to be uncut
            cut_predicates = [p for p in env.current_state.predicates if p.name == 'iscut']
            for predicate in cut_predicates:
                if env.current_state.predicates[predicate]:
                    env.current_state.predicates[predicate] = False
                    stochastic_done = True
                    break
        
        steps += 1
        if agent_retry_cond(agent, math.floor(max_steps - steps)):
            steps = 0
            obs, info = env.reset()
            queued_actions = []
    
    img = env.render(render_mode, close=True)
    if record:
        imgs.append(img)
        filename = kwargs.get('video_path', 'recorded_video.mp4')
        fourcc_str = kwargs.get('fourcc_str', 'avc1')
        fps = kwargs.get('video_fps', 3) # Videos with FPS < 3 on MP4 will appear corrupted (all green)
        record_video(imgs, filename, fourcc_str, fps)
    
    return done, steps

def render_img(env: RobotouilleEnv, state: State):
    '''
    Rendering function separated from qt interface
    '''
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
                    utils_temp.get_env_asset_path(f"imgs/robot_{vec2dir[player_direction]}.png"))
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
        #     # draw item on robot if holding any
        #     if is_true and literal.name == "has_item" and literal.params[0].name == player.name:
        #         player_pos = self.player_pose[player.name]["position"]
        #         held_item_name = literal.params[1].name
        # if held_item_name:
        #     self._draw_item_image(surface, held_item_name, obs, player_pos * self.pix_square_size)

        # Plot grill, cutting board
        for i, row in enumerate(layout):
            for j, col in enumerate(row):
                if col is not None:
                    draw = False
                    x, y = j, num_rows - i - 1
                    if 'stove' in col:
                        img = mpimg.imread(utils_temp.get_env_asset_path("imgs/grill.png"))
                        draw = True
                    if 'board' in col:
                        img = mpimg.imread(utils_temp.get_env_asset_path("imgs/cutting_board.png"))
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
        # type_to_img = {
        #     self._top_bun_type:
        #     mpimg.imread("predicators/envs/assets/imgs/top_bun.png"),
        #     self._bottom_bun_type:
        #     mpimg.imread("predicators/envs/assets/imgs/bottom_bun.png"),
        #     self._cheese_type:
        #     mpimg.imread("predicators/envs/assets/imgs/cheese.png"),
        #     self._tomato_type:
        #     mpimg.imread(utils.get_env_asset_path("imgs/uncut_lettuce.png")),
        #     self._patty_type:
        #     mpimg.imread(
        #         utils.get_env_asset_path("imgs/realistic_raw_patty_full.png"))
        # }
        held_img_size = (0.3, 0.3)
        offset = held_img_size[1] * (1 / 3)

    
        # Draw background
        floor_img = mpimg.imread(
            utils_temp.get_env_asset_path("imgs/floorwood.png"))
        for y in range(num_rows):
            for x in range(num_cols):
                ax.imshow(floor_img, extent=[x, x + 1, y, y + 1], zorder=-1)

        ax.set_xlim(0, num_cols)
        ax.set_ylim(0, num_rows)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
    plt.savefig("my_plot.png")



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.evaluation.evaluate:
        kwargs = OmegaConf.to_container(cfg.game, resolve=True)
        kwargs['llm_kwargs'] = OmegaConf.to_container(cfg.llm, resolve=True)
        environment_name = kwargs.pop('environment_name')
        agent_name = kwargs.pop('agent_name')
        run_robotouille(environment_name, agent_name, **kwargs)
    # else:
    #     evaluate(cfg)

if __name__ == "__main__":
    main()