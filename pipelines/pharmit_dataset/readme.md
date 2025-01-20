# The Pharmit Dataset Pipeline

Pharmit is a tool built/maintained by the Koes group for pharmacophore based screening. Because of this, our hosted pharmit website actually contains an extremely large dataset of 3D drug-like ligands. This directory is where we will build a pipeline for processing this pharmit dataset down into the standardized tensor format that will be ingested by omtra. 

# Where is the data and what does it look like?

The pharmit dataset is located on jabba (`jabba.csb.pitt.edu`). On jabba there are many data directories located at `/`:

```console
icd3@jabba:~/OMTRA$ ls -lh / | grep data*
drwxr-xr-x    2 dkoes root 4.0K May 22  2019 data
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data00
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data01
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data02
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data03
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data04
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data05
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data06
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data07
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data08
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data09
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data10
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data11
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data12
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data13
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data14
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data15
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data16
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data17
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data18
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data19
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data20
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data21
drwxr-xr-x   11 dkoes root 4.0K Oct 21 14:12 data22
drwxr-xr-x    5 dkoes root 4.0K May 25  2019 data23
```

All the data directories with a number after the word data contain...data (`/data/` is empty 🤷) 

At the top level, each data directory look *something* like this:

```console
icd3@jabba:~/OMTRA$ ls -lh /data00
total 59G
drwxrwxr-x 51136 dkoes dkoes 1.1M Jan 20 12:17 conformers
drwxrwxr-x    47 dkoes dkoes 4.0K Oct 24 05:48 databases
drwx------     2 root  root   16K May 24  2019 lost+found
-rw-rw-r--     1 dkoes dkoes  59G May 24  2019 molportsdfs.tar
```

They seem to look slightly different for different datasets. The consistent file structure that we care about is:


```console
/dataXX
├── conformers
│   ├── 0
    |   ├── 0.sdf.gz
    |   ├── 1.sdf.gz
    |   └── ...
│   ├── 1
    |   ├── 0.sdf.gz
    |   ├── 1.sdf.gz
    |   └── ...
│   └── ...
└── databases
    ├── the structure here is less clear to me
```

Although it seems that the .sdf.gz files are not labeled consecutively from 0-> within a directory but I don't know the pattern for the numbers.

# What does an indiviudal conformer file look like?

We are going to inspect `/data01/conformers/0/100.sdf.gz` and `/data01/conformers/0/1012.sdf.gz`

It seems each .sdf.gz file contains conformers for just 1 molecule. The number of conformers is variable but we'll just take the first.

The files also contain a pharmacophore for each ligand. Here is one molecule from `100.sdf.gz` unzipped:

```plaintext
100
 OpenBabel06051510463D

 12 12  0  0  0  0  0  0  0  0999 V2000
    1.3074    2.1239   -0.8834 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.3832    0.6720   -0.8940 C   0  0  1  0  0  0  0  0  0  0  0  0
    2.3130    0.1888    0.1712 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1091    0.5400    1.5004 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.2885   -0.5429   -0.1384 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0200    0.0669   -0.7618 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8117    0.3751    0.3261 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0800   -0.2039    0.4269 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8794    0.0957    1.4783 F   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5230   -1.0920   -0.5575 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.6980   -1.4011   -1.6428 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4292   -0.8225   -1.7444 C   0  0  0  0  0  0  0  0  0  0  0  0
  2  1  1  1  0  0  0
  2  3  1  0  0  0  0
  2  6  1  0  0  0  0
  3  4  1  0  0  0  0
  3  5  2  0  0  0  0
  6  7  2  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  8 10  2  0  0  0  0
 10 11  1  0  0  0  0
 11 12  2  0  0  0  0
 12  6  1  0  0  0  0
M  END
>  <pharmacophore>
Aromatic -1.25365 -0.512917 -0.658917 [ -0.389643 0.763975 -0.514315 ] [ 0.389643 -0.763975 0.514315 ]
HydrogenDonor 1.3074 2.1239 -0.8834 [ 0.919654 0.382457 -0.0892422 ] [ -0.406299 0.3063 0.860872 ]
HydrogenAcceptor 1.3074 2.1239 -0.8834 [ 0.919654 0.382457 -0.0892422 ] [ -0.406299 0.3063 0.860872 ]
HydrogenAcceptor 2.1091 0.54 1.5004 [ 0.627703 -0.328287 0.705845 ]
HydrogenAcceptor 3.2885 -0.5429 -0.1384 [ 0.77537 -0.581587 -0.246084 ]
NegativeIon 2.5702 0.0619667 0.511067
Hydrophobic -1.25365 -0.512917 -0.658917
Hydrophobic -2.8794 0.0957 1.4783


$$$$
```

So we have to write code to parse the pharmacophore information out of the sdf files, too. I had chatgpt write a script to do this. It's slightly incorrect; some of the pharmacophores have 3 floats and then subsequent floats in brackets. I don't know what these mean, they may be important. But this code will only take the first 3 floats.

```python
def parse_pharmacophore_data(pharmacophore_data):
    """
    Parses pharmacophore data into a dictionary.
    Args:
        pharmacophore_data (str): Raw pharmacophore data as a string.

    Returns:
        dict: Parsed pharmacophore data with types as keys and lists of tuples as values.
    """
    parsed_data = {}
    lines = pharmacophore_data.splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            key = parts[0]  # Pharmacophore type
            try:
                coordinates = tuple(map(float, parts[1:4]))  # Extract the 3 float values
                print(line)
                print(coordinates)
                if key not in parsed_data:
                    parsed_data[key] = []
                parsed_data[key].append(coordinates)
            except ValueError:
                print(f"Skipping line due to parsing error: {line}")

    return parsed_data

def read_sdf_and_print_pharmacophore(file_path):
    # Open and read the SDF file
    suppl = Chem.SDMolSupplier(file_path)

    if not suppl:
        print("Failed to read the SDF file.")
        return

    for mol in suppl:
        if mol is None:
            print("Failed to parse a molecule.")
            continue

        # Retrieve the pharmacophore data from the molecule properties
        pharmacophore_data = mol.GetProp("pharmacophore") if mol.HasProp("pharmacophore") else None

        if pharmacophore_data:
            print("Raw Pharmacophore Data:")
            print(pharmacophore_data)

            # Parse the pharmacophore data into a dictionary
            parsed_data = parse_pharmacophore_data(pharmacophore_data)
            print("Parsed Pharmacophore Data:")
            print(parsed_data)
        else:
            print("No pharmacophore data found in the molecule.")
```