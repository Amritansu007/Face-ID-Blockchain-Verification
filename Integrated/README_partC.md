# Part C — Blockchain Verification

> **Pipeline**: Face Detection (Part A) → Reverse Image Search & Verification (Part B) → **Blockchain Verification (Part C)**
>
> This module receives a verified match payload from Part B, writes a tamper-evident
> record to a Solidity smart contract on the **Ethereum Sepolia testnet**,
> and verifies the record by reading it back on-chain and recomputing the cryptographic hash.

---

## File Structure

```
.
├── contracts/
│   └── StoreRecord.sol      # Solidity ^0.8.19 — deploy via Remix
├── partC.py                  # Blockchain verification engine (web3.py + dotenv)
├── demo_match.json           # Sample VERIFIED_MATCH payload
├── .env.example              # Template for secrets
├── .gitignore
├── requirements.txt
└── README_partC.md           # Documentation for Part C
```

---

## Target Chain

| Property | Value |
|----------|-------|
| Network  | Ethereum Sepolia (testnet) |
| Chain ID | 11155111 |
| Explorer | <https://sepolia.etherscan.io> |
| Faucet   | <https://sepoliafaucet.com> or <https://faucets.chain.link/sepolia> |

---

## Setup

### 1. Deploy the Contract (or use deployed address)

1. Open [Remix IDE](https://remix.ethereum.org).
2. Paste `contracts/StoreRecord.sol` into a new file.
3. Compile with Solidity **^0.8.19**.
4. Deploy using **Injected Provider – MetaMask** on Sepolia.
5. Copy the deployed **contract address** and the **ABI** (Compilation Details → ABI).

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure the environment variables:
```ini
RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_ALCHEMY_KEY
PRIVATE_KEY=your_metamask_private_key_without_0x_or_with_0x
CONTRACT_ADDRESS=0xYourDeployedContractAddress
```

---

## How It Works

1. **Fail-safe Gate**: If `match_status != "VERIFIED_MATCH"`, the transaction is skipped entirely (`SKIPPED`). Unverified matches never touch the chain.
2. **Deterministic Hashing**:
   $$\text{canonical} = \text{image\_hash\_sha256} : \text{matched\_url} : \text{evidence\_score (1 decimal)} : \text{timestamp}$$
   $$\text{dataHash} = \text{SHA-256}(\text{canonical})$$
3. **On-Chain Write**: Calls `storeRecord(bytes32 dataHash, string url, uint256 timestamp)` on Sepolia.
4. **On-Chain Verification**: Calls `getRecord(dataHash)` to verify data stored on the blockchain matches the local payload.
