from algorithms.sac import SAC
from algorithms.sac_aug import SAC_AUG
from algorithms.soda import SODA
from algorithms.soda_aug import SODA_AUG
from algorithms.drq import DrQ
from algorithms.drq_aug import DrQ_AUG
from algorithms.svea_c import SVEA_C
from algorithms.svea_c_aug import SVEA_C_AUG
from algorithms.svea_o import SVEA_O
from algorithms.svea_o_aug import SVEA_O_AUG

algorithm = {
	'sac': SAC,
	'sac_aug':SAC_AUG,
	'soda': SODA,
	'soda_aug':SODA_AUG,
	'drq': DrQ,
	'drq_aug': DrQ_AUG,
	'svea_c':SVEA_C,
	'svea_c_aug':SVEA_C_AUG,
	'svea_o':SVEA_O,
	'svea_o_aug':SVEA_O_AUG
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
