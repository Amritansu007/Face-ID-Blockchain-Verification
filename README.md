# Face ID + Blockchain Verification

A pipeline that detects a face from a photo, finds a real matching social
media post via reverse image search, and writes that match to a blockchain
as a tamper-evident, verifiable record.

```
Face image → Part A (Face Detection) → Part B (Reverse Image Search) → Part C (Blockchain Record) → Verified output
```

## Overview

Given an input photo, this project:
1. Detects and crops the face
2. Independently verifies the face against candidate matches found online,
   ranks them, and returns the best real match with an explainable evidence
   score — or safely reports no reliable match instead of guessing
3. Writes a tamper-evident record of that match to an Ethereum testnet,
   then re-reads and verifies it, proving the record hasn't been altered

No hardcoded results at any stage — every match is a genuine search, and
no unverified match is ever written to the blockchain.

## Team & Components

| Part | Owner | What it does | Details |
|------|-------|---------------|---------|
| **Part A** — Face Detection & Encoding | Krishiv Rathi | Detects a face in the input photo, crops it, computes a face encoding | [`Integrated/README_partA.md`](Integrated/README_partA.md) |
| **Part B** — Reverse Image Search & Evidence Verification | Somanshu Vyas | Independently verifies candidate matches using multi-signal computer vision scoring, returns the best real match or a safe "no reliable match" result | [`Integrated/README_partB.md`](Integrated/README_partB.md) |
| **Part C** — Blockchain Record & Verification | Amritansu Singh | Writes a tamper-evident hash of the verified match to a smart contract on Sepolia testnet, then re-verifies it on-chain | [`Integrated/README_partC.md`](Integrated/README_partC.md) |

## Running the Full Pipeline

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
python main.py path/to/photo.jpg
```

See each part's own README (linked above) for component-specific setup —
API keys, wallet/testnet configuration, and dependencies.

## Repo Structure

```
.
├── partA/          # Face detection
├── partB/          # Reverse image search
├── partC/          # Blockchain record + verification
├── main.py         # Integrates all three parts end to end
├── requirements.txt
├── .env.example
└── README.md       # This file
```

## Demo

[Link to full screen recording of the end-to-end pipeline — input photo →
detected face → matched post found → blockchain record written →
re-verified on screen]

## Known Limitations

- Reverse image search quality depends on how discoverable the input photo
  already is online
- Testnet blockchain data isn't permanent and depends on faucet/RPC uptime
- See each part's own README for component-specific limitations
