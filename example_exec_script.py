import hydra
from omegaconf import DictConfig, OmegaConf
from robotouille.robotouille_env import create_robotouille_env
from skills import SkillManager, run_skill_sequence_and_record

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