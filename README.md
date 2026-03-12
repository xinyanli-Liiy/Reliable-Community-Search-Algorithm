# Implementation for Reliable Community Search

This repository is a reference implementation of the algorithms proposed in:

**"Efficient Reliable Community Search Algorithm in Temporal Networks"**

---

## Data Source

Popular temporal graph datasets are available at:

- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)
- [Network Data Repository](https://networkrepository.com/networks.php)

---

## Data Preparation

Data is organized in **`.gml` format**, where:

- Each file represents a **one-timestamp graph instance**  
- Each edge has a **`weight` attribute**

---

## Requirements

- Python 3.12  
- `networkx == 3.3`  
- `click == 8.1.7`  

Install dependencies via:

```bash
pip install networkx==3.3 click==8.1.7

| Parameter    | Description                                                     |
| ------------ | --------------------------------------------------------------- |
| Dataset name | Name of the dataset folder (string)                             |
| θ (Theta)    | Edge weight threshold, float in [0,1]                           |
| k (K)        | k-core constraint (integer)                                     |
| query        | Query vertex                                                    |
| α (Alpha)    | Balance factor for community size and duration (positive float) |
| T_s          | Start timestamp of query interval (inclusive, integer)          |
| T_e          | End timestamp of query interval (exclusive, integer)            |


Example Input
Dataset name(str): Bitcoin_otc
Theta(float): 0.5
K(int): 2
query(str): 1
Alpha(float): 1
T_s(int): 0
T_e(int): 10
Example Output
(Bitcoin_otc-0.5-2-1-1.0_WCF.txt)
Index construction time: 4.894751 s
Running time of query: 0.267192 s

Found 1 optimal subgraph(s):
========================================
Option 1:
  Vertices Size: 430
  Edges Size:    1235
Nodes List:
========================================
Run the Code

Execute the main script:

python run.py
