import re
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
import gym
from gym import spaces
from gym.wrappers.env_checker import PassiveEnvChecker
from tests.envs.test_envs import PASSIVE_CHECK_IGNORE_WARNING
from tests.envs.utils import all_testing_initialised_envs
from tests.testing_env import GenericTestEnv

@pytest.mark.parametrize(
    "env",
    all_testing_initialised_envs,
    ids=[env.spec.id for env in all_testing_initialised_envs],
)
def test_passive_checker_wrapper_warnings(env: gym.Env):
    with warnings.catch_warnings(record=True) as caught_warnings:
        checker_env = PassiveEnvChecker(env)
        checker_env.reset()
        checker_env.step(checker_env.action_space.sample())
        
        # Check for render
        if "render_modes" in checker_env.env.metadata:
            render_modes = checker_env.env.metadata["render_modes"]
            if render_modes and isinstance(render_modes, (list, tuple)):
                try:
                    checker_env.render(mode=render_modes[0])
                except Exception as e:
                    warnings.warn(f"Render check failed: {str(e)}")
        
        checker_env.close()
    
    for warning in caught_warnings:
        if warning.message.args[0] not in PASSIVE_CHECK_IGNORE_WARNING:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")

@pytest.mark.parametrize(
    "env, message",
    [
        (
            GenericTestEnv(action_space=None),
            "The environment must specify an action space. https://www.gymlibrary.dev/content/environment_creation/",
        ),
        (
            GenericTestEnv(action_space="error"),
            "action space does not inherit from `gym.spaces.Space`, actual type: <class 'str'>",
        ),
        (
            GenericTestEnv(observation_space=None),
            "The environment must specify an observation space. https://www.gymlibrary.dev/content/environment_creation/",
        ),
        (
            GenericTestEnv(observation_space="error"),
            "observation space does not inherit from `gym.spaces.Space`, actual type: <class 'str'>",
        ),
    ],
)
def test_initialise_failures(env: GenericTestEnv, message: str):
    with pytest.raises(AssertionError, match=f"^{re.escape(message)}$"):
        PassiveEnvChecker(env)
    env.close()

def _reset_failure(self, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    return np.array([-1.0], dtype=np.float32), {}

def _step_failure(self, action: Any) -> str:
    return "error"

def test_api_failures():
    env = GenericTestEnv(
        reset_fn=_reset_failure,
        step_fn=_step_failure,
        metadata={"render_modes": "error"},
    )
    env = PassiveEnvChecker(env)
    assert env.checked_reset is False
    assert env.checked_step is False
    assert env.checked_render is False

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "The obs returned by the `reset()` method is not within the observation space"
        ),
    ):
        env.reset()
    assert env.checked_reset

    with pytest.raises(
        AssertionError,
        match="Expects step result to be a tuple, actual type: <class 'str'>",
    ):
        env.step(env.action_space.sample())
    assert env.checked_step

    with pytest.warns(
        UserWarning,
        match=r"Expects the render_modes to be a sequence \(i\.e\. list, tuple\), actual type: <class 'str'>",
    ):
        env.render()
    assert env.checked_render

    env.close()

def test_space_checks():
    # Test for proper action and observation spaces
    proper_env = GenericTestEnv(
        action_space=spaces.Discrete(2),
        observation_space=spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
    )
    PassiveEnvChecker(proper_env)

    # Test for improper action space
    with pytest.raises(AssertionError, match="action space must be a gym.spaces.Space"):
        improper_action_env = GenericTestEnv(action_space="not a space")
        PassiveEnvChecker(improper_action_env)

    # Test for improper observation space
    with pytest.raises(AssertionError, match="observation space must be a gym.spaces.Space"):
        improper_obs_env = GenericTestEnv(observation_space="not a space")
        PassiveEnvChecker(improper_obs_env)

def test_render_modes_check():
    # Test for proper render modes
    proper_render_env = GenericTestEnv(metadata={"render_modes": ["human", "rgb_array"]})
    checker = PassiveEnvChecker(proper_render_env)
    assert checker.checked_render is False
    checker.render()
    assert checker.checked_render is True

    # Test for improper render modes
    improper_render_env = GenericTestEnv(metadata={"render_modes": "human"})
    checker = PassiveEnvChecker(improper_render_env)
    with pytest.warns(UserWarning, match="Expects the render_modes to be a sequence"):
        checker.render()

def test_reset_return_type():
    env = GenericTestEnv()
    checker = PassiveEnvChecker(env)
    
    obs, info = checker.reset()
    assert isinstance(obs, np.ndarray), "Reset should return a numpy array as observation"
    assert isinstance(info, dict), "Reset should return a dict as info"

def test_step_return_type():
    env = GenericTestEnv()
    checker = PassiveEnvChecker(env)
    
    checker.reset()
    obs, reward, done, truncated, info = checker.step(env.action_space.sample())
    assert isinstance(obs, np.ndarray), "Step should return a numpy array as observation"
    assert isinstance(reward, (int, float)), "Step should return a number as reward"
    assert isinstance(done, bool), "Step should return a boolean as done flag"
    assert isinstance(truncated, bool), "Step should return a boolean as truncated flag"
    assert isinstance(info, dict), "Step should return a dict as info"

if __name__ == "__main__":
    pytest.main([__file__])
