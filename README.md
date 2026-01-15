# Moe-Gen-Code
This repository contains the code implementation related to Moe-Gen
```

Moe-Gen-Code/
├── Centry_Inject/         # Distribution analysis tools
│   ├── dis_main.py        # Main script for distribution evaluation
│   └── dis_tool.py        # Tools for distribution analysis
├── DataTool/              # Data processing utilities
│   ├── DataTool.py        # Data conversion and processing functions
│   └── LoadData.py        # Data loading functions for all datasets
├── DefenceCompare/        # Defense method comparison tools
│   └── CW_Compare.py      # Closed-world defense comparison script
├── Defence_Model/         # Implementation of various defense models
│   ├── Alert/             # Alert defense model
│   ├── AWA/               # AWA defense model
│   ├── DFD/               # DFD defense model
│   ├── Moe-Gen/           # Moe-Gen defense model (main contribution)
|   |   ├──Evaluate_Gen.py    # Evaluate trained Moe-Gen script
|   |   ├──Moe_Gen_def.py     # interface provided defence tool method
|   |   ├──Train_Moe_Gen.py   # train Moe-Gen script
|   |   ├──Gen_Model          # Moe-Gen util
|   |   └──Gen_Save           # provided trained Moe-Gen in AWF100
│   └── WalkieTalkie/      # WalkieTalkie defense model
├── OpenWorld/             # Open-world defense evaluation
│   └── OpenWorld_Compare.py # Open-world defense comparison script
└── WF_Model/              # Website fingerprinting models
    ├── CFModel_Loder.py   # Classifier model loading utilities
    ├── test_ClassfyModel.py # Model evaluation functions
    └── Train_CFModel_NoDef.py # Model training script
```

# PluggableTransport - Moegen（Code

```
├── common/           # Common utility libraries
│   ├── csrand/       # Cryptographically secure random number generation
│   ├── drbg/         # Deterministic random bit generator
│   ├── log/          # Logging system
│   ├── ntor/         # NTor handshake protocol implementation
│   ├── probdist/     # Probability distribution implementation
│   ├── replayfilter/ # Replay attack filter
│   └── socks5/       # SOCKS5 protocol implementation
├── git.torproject.org/ # Tor project dependencies
│   └── pluggable-transports/
│       └── goptlib/  # Tor pluggable transport library
├── internal/         # Internal cryptographic libraries
│   ├── edwards25519/ # Edwards25519 elliptic curve implementation
│   └── extra25519/   # Additional 25519 elliptic curve functionality
├── obfs4proxy/       # Main program entry
│   └── obfs4proxy.go # Program main entry
├── transports/       # Transport protocol implementations
│   ├── base/         # Transport protocol base interfaces
│   ├── moegen/       # Moegen transport protocol implementation
│   ├── obfs4/        # Obfs4 transport protocol implementation
│   └── transports.go # Transport protocol registration and management
├── LICENSE           # License file
├── LICENSE-GPL3.txt  # GPL3 license file
├── go.mod            # Go module dependencies
└── go.sum            # Go module checksums
```
# DataSet
The DataSet directory contains a small subset of AWF100, which includes the original burst sequences and their corresponding perturbed sequences generated using the DFD and Walkie-Talkie methods. Due to GitHub's upload limitations, each class in this uploaded subset is restricted to 100 samples.


