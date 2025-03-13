# prior_factory.py
from omtra.priors.register import train_prior_register, inference_prior_register
from functools import partial

def get_prior(task_cls, config_prior=None, train=False) -> dict:
    """
    Get the prior distribution function for all modalitities for a given task class.
    :param task_cls: The task class (e.g., TaskA)
    :param config_prior: Optional config override dict.
    :return: A dictionary with keys for each modality and values that are the prior distribution functions.
    """
    if train:
        register = train_prior_register
    else:
        register = inference_prior_register

    task_name = task_cls.name

    prior_fn_output = {}

    if 'ligand_identity' in task_cls.modalities_present:
        prior_fn_output['lig'] = {}
        default_lig_cfg: dict = task_cls.priors['lig'] # should be a dictionary with a key for each ligand modality
        for lig_modality in default_lig_cfg:

            # get the type of prior
            try:
                prior_fn_key = config_prior[task_name]['lig'][lig_modality]['type']
            except KeyError:
                prior_fn_key = default_lig_cfg[lig_modality]['type']

            # get any params
            try:
                prior_params = config_prior[task_name]['lig'][lig_modality]['params']
            except KeyError:
                prior_params = default_lig_cfg[lig_modality].get('params', {})

            prior_fn = register[prior_fn_key]
            prior_fn = partial(prior_fn, **prior_params)
            prior_fn_output['lig'][lig_modality] = (prior_fn_key, prior_fn)

    if 'pharmacophore' in task_cls.modalities_present:
        prior_fn_output['pharm'] = {}
        default_pharm_cfg: dict = task_cls.priors['pharm']
        for pharm_modality in default_pharm_cfg:
            try:
                prior_fn_key = config_prior[task_name]['pharm'][pharm_modality]['type']
            except KeyError:
                prior_fn_key = default_pharm_cfg[pharm_modality]['type']

            try:
                prior_params = config_prior[task_name]['pharm'][pharm_modality]['params']
            except KeyError:
                prior_params = default_pharm_cfg[pharm_modality].get('params', {})

            prior_fn = register[prior_fn_key]
            prior_fn = partial(prior_fn, **prior_params)
            prior_fn_output['pharm'][pharm_modality] = (prior_fn_key, prior_fn)

    if 'protein' in task_cls.modalities_present:
        default_prot_cfg: dict = task_cls.priors['protein']
        # for now, protein is only one modalitiy, this is entirely subject to change
        try:
            prior_fn_key = config_prior[task_name]['protein']['type']
        except KeyError:
            prior_fn_key = default_prot_cfg['type']
        
        try:
            prior_params = config_prior[task_name]['protein']['params']
        except KeyError:
            prior_params = default_prot_cfg.get('params', {})

        prior_fn = register[prior_fn_key]
        prior_fn = partial(prior_fn, **prior_params)
        prior_fn_output['protein'] = (prior_fn_key, prior_fn)

    return prior_fn_output