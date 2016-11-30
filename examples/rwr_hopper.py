from rllab.algos.erwr import ERWR
from rllab.envs.gym_env import GymEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import instrument
import sys

instrument.stub(globals())

env = normalize(GymEnv("Hopper-v1", record_video=False))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 42 hidden units.
    hidden_sizes=(42, 42)
)

# baseline = LinearFeatureBaseline(env_spec=env.spec)
baseline = ZeroBaseline(env_spec=env.spec)

vg = instrument.VariantGenerator()
vg.add("seed", range(1))

variants = vg.variants()

for variant in variants:
    algo = ERWR(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=500,
        n_itr=300,
        discount=0.99,
        exp_prefix="rwr-hopper-search",
        exp_name="seed_{0}".format(variant["seed"]),
        plot=True,
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used,
        seed=variant["seed"],
        exp_prefix="rwr_hopper_full_traj_exp_nodone_search",
        exp_name="seed_{0}".format(variant["seed"]),
        plot=True,
    )
