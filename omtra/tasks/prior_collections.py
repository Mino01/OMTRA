# store common formats for priors


# typical prior setup for de novo ligand
denovo_ligand = {
    'x': {
        'type': 'gaussian',
        'params': {'ot': True}
    }
}
for modality in 'ace':
    denovo_ligand[modality] = dict(type='masked')

# typical prior setup for ligand conformer
ligand_conformer = {
    'x': {
        'type': 'gaussian',
        'params': {'ot': True}
    }
}
for modality in 'ace':
    ligand_conformer[modality] = dict(type='fixed')

# typical prior setup for de novo pharmacophore
denovo_pharmacophore = {
    'x': {
        'type': 'gaussian',
        'params': {'ot': True}
    },
    'a': dict(type='masked'),
    'v': dict(type='gaussian')
}


# fixed pharmacophore
fixed_pharmacophore = {
    'x': dict(type='fixed'),
    'a': dict(type='fixed'),
    'v': dict(type='fixed')
}