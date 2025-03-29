# One Model to Rule Them All

A multi-task generative model for small-molecule structure-based drug design. 

# Building the Environment

For now:
```bash
git clone https://github.com/gnina/OMTRA.git
cd OMTRA
mamba create -n omtra python=3.11
mamba activate omtra
chmod +x build_env.sh
./build_env.sh
```

# TODO:
- [ ] add npnde to protein modality group
- [ ] how to model npnde? need to maintain edge features? writing npndes back out requires we have edges? but we don't really need to model npnde edges?
- [ ] finish adding conditional paths to task definitions
- [ ] there are entities that are in our systems but not explicitly modeled as modalities (protein atom type) because they are always fixed. the effect is that protein atom types are not actually incorporated into the system state at time t, and not "tracked" by our task class. we either need to incorporate these into our modality system and make them always fixed, or we need to hard-code treatment of protein identity... i think the former is preferable.
- [ ] have we handled pharm vec features appropriately in VF class?
- [ ] do we initialize pharm vec features?
- [ ] we don't create node output heads for pharm vec features
- [ ] task embedding
- [ ] prot element + name embedding?
- [ ] are atom name embeddings set correctly? what about residue typing?
- [ ] need to embed protein types?
- [ ] how does heterogvpconv create messaging and update functions?