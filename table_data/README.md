# The original table files
To reproduce the results from the paper, you can direclty use the pretrained models or extracted features feature we provided in the [main download script](https://github.com/megagonlabs/sato/blob/master/download_data.sh). Here we also provide the raw and processed table we used in our experiements. `viznet_tables` is a subset of the [VizNet](https://viznet.media.mit.edu/) corpus, and `sato_tables` contains the same set of tables but filtered to only include columns with valid semantic types. Please see below for dataset details.

## Download the files

Run the script and you will see `sata_tables` and `viznet_tables`.

```
$ bash download.sh
```


## `sato_tables`: Original Sato Tables (used in the paper)

The directory contains two directories `all` and `multionly`, which correspond to All tables `D` and Multi-column tables `D_multi` in the paper. The tables are split into 5 disjoint subset for cross-validation. Each table is stored in a CSV format, so you can load the files using a standard way (e.g., Pandas `read_csv()`.)

```
sato_tables/
└── all
    ├── K0
    ├── K1
    ├── K2
    ├── K3    
    └── K4
└── multionly
    ├── K0
    ├── K1
    ├── K2
    ├── K3    
    └── K4

```

For each CV, the same block id as the cv index was used as the test data. That is,

```
CV0: Train K1, 2, 3, 4, Test K0
CV1: Train K0, 2, 3, 4, Test K1
CV2: Train K0, 1, 3, 4, Test K2
CV3: Train K0, 1, 2, 4, Test K3
CV4: Train K0, 1, 2, 3, Test K4
```


## `viznet_tables`: Original VizNet Tables (before filtering & label assignment)

In case you're interestd in using the original tables that are extracted from the VizNet corpus before filtering and semantic type assignment, please refer to `viznet_tables`.


`viznet_tables` stores the original table files in a CSV format. `K{0-4}` contains all the original tables and `K{0-4}_multi-col` contains a subset of the tables that have more than one columns (i.e., multi-column tables in the experiments.)


```
viznet_tables/
├── webtables1
│   ├── K0
│   ├── K0_multi-col
│   ├── K1
│   ├── K1_multi-col
│   ├── K2
│   ├── K2_multi-col
│   ├── K3
│   ├── K3_multi-col
│   ├── K4
│   └── K4_multi-col
└── webtables2
    ├── K0
    ├── K0_multi-col
    ├── K1
    ├── K1_multi-col
    ├── K2
    ├── K2_multi-col
    ├── K3
    ├── K3_multi-col
    ├── K4
    └── K4_multi-col
```


