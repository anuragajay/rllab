from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.gripper_env import GripperEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import instrument
import sys

instrument.stub(globals())

env = normalize(GripperEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 42 hidden units.
    hidden_sizes=(42, 42)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

vg = instrument.VariantGenerator()
vg.add("seed", [1,2,3])
vg.add("step_size",[0.1])#,0.01,0.001])

variants = vg.variants()
num = eval(sys.argv[1])

print "#Experiments number:", num
variant = variants[num]


algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=90000,
    max_path_length=100,
    n_itr=1000,
    discount=0.99,
    step_size=variant["step_size"],
    optimizer_args={
        'cg_iters': 100
    },
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
    exp_prefix="trpo_grip_search",
    exp_name="seed_{0}_step_size_{1}".format(variant["seed"], variant["step_size"]),
)