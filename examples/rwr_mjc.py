from rllab.algos.erwr import ERWR
from rllab.envs.mujoco.peg_env import PegEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import instrument
import sys

instrument.stub(globals())

env = normalize(PegEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 42 hidden units.
    hidden_sizes=(42, 42)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

vg = instrument.VariantGenerator()
vg.add("seed", range(3))

variants = vg.variants()

for variant in [variants[2]]:
    algo = ERWR(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=45000,
        max_path_length=100,
        step_size=0.01,
        n_itr=1000,
        discount=0.99,
        exp_prefix="rwr-peg-search",
        exp_name="seed_{0}".format(variant["seed"]),
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=8,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used,
        seed=variant["seed"],
        exp_prefix="rwr_peg_search",
        exp_name="seed_{0}".format(variant["seed"]),
    )