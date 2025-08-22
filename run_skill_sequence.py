import sys
import argparse

import hydra
from omegaconf import DictConfig, OmegaConf

from robotouille.robotouille_env import create_robotouille_env
from skills import SkillManager, run_skill_sequence_and_record
from utils.helper_functions import save_to_file, load_from_file

def exec_and_record(environment_name: str, args, **kwargs):
    '''Minimal script for testing action rollout and screen shot'''
    # Initialize environment
    seed = kwargs.get('seed', None)
    env = create_robotouille_env(environment_name, seed)
    obs, info = env.reset()

    # Initialize skill manager
    skill_manager = SkillManager(env)
    skill_sequence = load_from_file(args.skill_sequence_fpath)
    # Run skill sequence
    run_skill_sequence_and_record(skill_manager, skill_sequence, args.save_path)

@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg: DictConfig) -> None:
    if not cfg.evaluation.evaluate:
        kwargs = OmegaConf.to_container(cfg.game, resolve=True)
        kwargs['llm_kwargs'] = OmegaConf.to_container(cfg.llm, resolve=True)
        environment_name = kwargs.pop('environment_name')
        exec_and_record(environment_name, args, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill_sequence_fpath", type=str, default="../test_tasks/burger_1.json", help="Path to the skill sequence YAML file.")
    parser.add_argument("--save_path", type=str, default="../test_tasks/burger/", help="Path to save the execution results.")
    args = parser.parse_args()

    main()