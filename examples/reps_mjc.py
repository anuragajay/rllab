from rllab.algos.reps import REPS
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
vg.add("seed", [1,2,3])
vg.add("step_size",[0.1])

variants = vg.variants()
num = eval(sys.argv[1])

print "#Experiments number:", num
variant = variants[num]


algo = REPS(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=90000,
    max_path_length=100,
    step_size=variant["step_size"],
    n_itr=1000,
    discount=0.99,
    plot=True,
)

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=8,
    plot=True,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used,
    seed=variant["seed"],
    exp_prefix="reps_peg_search",
    exp_name="seed_{0}".format(variant["seed"]),
)