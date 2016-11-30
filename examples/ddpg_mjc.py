from rllab.algos.ddpg import DDPG
from rllab.envs.mujoco.peg_env import PegEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.misc import instrument
import sys

instrument.stub(globals())

env = normalize(PegEnv(), normalize_reward=True)

policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(42, 42)
)

qf = ContinuousMLPQFunction(
    env_spec=env.spec,
    hidden_sizes=(42, 42)
)

vg = instrument.VariantGenerator()
vg.add("scale_reward", [0.01])#, 0.001, 0.1])
vg.add("policy_learning_rate", [1e-4])#, 1e-3, 1e-5])
vg.add("qf_learning_rate", [1e-3]) #, 1e-3, 1e-4])
vg.add("decay_period", [1E+6, 1E+5, 1E+4, 1E+3, 1E+7, 1E+8, 1E+9, 1E+10])

variants = vg.variants()
num = eval(sys.argv[1])

print "#Experiments number:", num
variant = variants[num]

# es = OUStrategy(env_spec=env.spec, theta=0.15, sigma=0.3)
es = GaussianStrategy(env_spec=env.spec, max_sigma=1.0, min_sigma=0.1, decay_period=variant["decay_period"])

algo = DDPG(
    env=env,
    policy=policy,
    es=es,
    qf=qf,
    batch_size=35,
    max_path_length=100,
    epoch_length=5000,
    min_pool_size=10000,
    n_epochs=100,
    discount=0.99,
    scale_reward=variant["scale_reward"],
    soft_target_tau=1e-3,
    qf_learning_rate=variant["qf_learning_rate"],
    policy_learning_rate=variant["policy_learning_rate"],
    #Uncomment both lines (this and the plot parameter below) to enable plotting
    plot=True,
    eval_samples=5000,
)

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    exp_prefix="dpg_search",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    exp_name=str(num),
    seed=2,
    plot=True,
)
