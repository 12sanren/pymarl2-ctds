REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_teach import EpisodeRunnerTeach
REGISTRY["episode_teach"] = EpisodeRunnerTeach

from .parallel_runner_teach import ParallelRunnerTeach
REGISTRY["parallel_teach"] = ParallelRunnerTeach
