"""Run scene shards with stock Habitat-Lab or an existing split-aware installation."""

import inspect
import logging
from collections import defaultdict
from types import SimpleNamespace

import habitat


def _configure_actions(config):
    """Register the agent's double right turn as one environment action."""
    if "TURN_RIGHT_2" not in config.TASK.POSSIBLE_ACTIONS:
        return
    if "TURN_RIGHT_2" in config.TASK.ACTIONS:
        return

    from habitat.core.registry import registry
    from habitat.tasks.nav.nav import TurnRightAction

    if registry.get_task_action("E2NavDoubleRight") is None:
        @registry.register_task_action(name="E2NavDoubleRight")
        class DoubleRight(TurnRightAction):
            def step(self, *args, **kwargs):
                super().step(*args, **kwargs)
                return super().step(*args, **kwargs)

    config.TASK.ACTIONS.TURN_RIGHT_2 = habitat.Config({"TYPE": "E2NavDoubleRight"})


def evaluate(agent, config, experiment):
    bounds = (experiment.split_l, experiment.split_r)
    if "split_l" in inspect.signature(habitat.Challenge).parameters:
        challenge = habitat.Challenge(eval_remote=False, split_l=bounds[0], split_r=bounds[1])
        return challenge.submit(agent)

    config = config.clone()
    config.defrost()
    from habitat.core.registry import registry

    dataset = registry.get_dataset(config.DATASET.TYPE)
    scenes = config.DATASET.CONTENT_SCENES
    if "*" in scenes:
        scenes = sorted(dataset.get_scenes_to_load(config.DATASET))
    left, right = bounds
    if not 0 <= left < right or not scenes[left:right]:
        raise ValueError(f"Empty or invalid scene range [{left}, {right}) for {len(scenes)} scenes")
    config.DATASET.CONTENT_SCENES = scenes[left:right]
    if config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID < 0:
        config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = experiment.cuda_id
    _configure_actions(config)
    config.freeze()

    with habitat.Env(config=config) as env:
        agent.simulator = SimpleNamespace(_env=env)
        first = max(1, experiment.episode_l)
        last = min(len(env.episodes) + 1, experiment.episode_r)
        if first >= last:
            raise ValueError(f"No episodes in [{first}, {last}) for the selected scenes")
        totals = defaultdict(float)
        for number in range(1, last):
            observations = env.reset()
            if number < first:
                continue
            agent.episode_id = number - 1
            agent.reset()
            while not env.episode_over:
                agent.update_metrics(env.get_metrics())
                observations = env.step(agent.act(observations))
            for name, value in env.get_metrics().items():
                totals[name] += value
        averages = {name: total / (last - first) for name, total in totals.items()}
        logging.getLogger(__name__).info("Evaluation metrics: %s", averages)
        return averages
