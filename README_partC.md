# Part C — Blockchain Verification

> **Pipeline**: Face Detection → Reverse Image Search → **Blockchain Verification**
>
> This module receives a match payload from Part B, writes a tamper-evident
> record to a Solidity smart contract on the **Ethereum Sepolia testnet**,
> and verifies the record by reading it back and recomputing the hash.

---

## File Structure

```
FaceID-Blockchain-PartC/
├── contracts/
│   └── StoreRecord.sol      # Solidity ^0.8.19 — deploy via Remix
├── partC.py                  # Python script (web3.py + dotenv)
├── demo_match.json           # Sample VERIFIED_MATCH payload
├── .env.example              # Template for secrets
├── .gitignore
├── requirements.txt
└── README_partC.md           # ← you are here
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

### 1. Deploy the contract

1. Open [Remix IDE](https://remix.ethereum.org).
2. Paste `contracts/StoreRecord.sol` into a new file.
3. Compile with Solidity **^0.8.19**.
4. Deploy using **Injected Provider – MetaMask** on Sepolia.
5. Copy the deployed **contract address** and the **ABI** (Compilation
   Details → ABI → copy icon).

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in the four variables:

| Variable | Where to get it |
|----------|-----------------|
| `RPC_URL` | Alchemy / Infura / QuickNode — create a free Sepolia app |
| `PRIVATE_KEY` | MetaMask → Account Details → Export Private Key |
| `CONTRACT_ADDRESS` | Remix deploy output |
| `CONTRACT_ABI` | Remix → Compilation Details → ABI (paste as one-line JSON) |

> ⚠️ **Never commit `.env`** — it contains your private key.

### 3. Install Python dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

---

## How to Run

```bash
python partC.py
```

The script loads `demo_match.json`, checks that `match_status` is
`VERIFIED_MATCH`, builds a deterministic SHA-256 hash, sends a
`storeRecord` transaction, waits for the receipt, reads the record back via
`getRecord`, and prints a JSON result.

### Example output

```
============================================================
  Part C — Blockchain Verification
============================================================

Input payload:
{ … }

⏳ Sending storeRecord tx from 0xYourAddress …
✅ Tx mined: 0xabc123…  (status=1)

============================================================
  Result
============================================================
{
  "status": "STORED_AND_VERIFIED",
  "tx_hash": "0xabc123…",
  "block_number": 12345678,
  "data_hash": "0x…",
  "verification": {
    "hash_match": true,
    "indicator": "✅ Verified",
    "on_chain_url": "https://instagram.com/johndoe/avatar",
    "on_chain_timestamp": 1741192800,
    "on_chain_submitter": "0xYourAddress",
    "recomputed_hash": "0x…",
    "expected_hash": "0x…"
  }
}
```

---

## How It Works

1. **Fail-safe gate** — If `match_status` ≠ `VERIFIED_MATCH`, the
   transaction is skipped entirely (status `SKIPPED`). Unverified matches
   never touch the chain.

2. **Deterministic hashing** — The canonical string
   `{image_hash_sha256}:{matched_url}:{evidence_score:.1f}:{timestamp}`
   is SHA-256'd. The `:.1f` format guarantees `88.5` always hashes the
   same way regardless of floating-point representation.

3. **On-chain storage** — `storeRecord(bytes32, string, uint256)` writes
   the hash, URL, and timestamp. The contract reverts if the same
   `dataHash` has already been stored (no silent overwrites).

4. **Verification** — `getRecord` reads back the on-chain data; the script
   recomputes the hash from the original payload and compares.

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| **Testnet data isn't permanent** | Sepolia may be reset or deprecated. Records are for demonstration only. |
| **Faucet dependency** | You need Sepolia ETH for gas. Faucets may rate-limit or go offline. |
| **RPC dependency** | The script needs a working Sepolia RPC endpoint (Alchemy, Infura, etc.). |
| **Duplicate-hash reverts** | Submitting the same `dataHash` twice will revert. This is intentional (tamper evidence) but means re-runs with unchanged data will fail. |
| **No on-chain access control** | Anyone can call `storeRecord`. For production, consider adding `Ownable` or an allow-list. |
| **Single-record verification** | The script verifies one match at a time. Batch support is left for future work. |

---

## Integration with Parts A & B

Replace `demo_match.json` with the live JSON output from Part B's reverse
image search. The expected schema:

```jsonc
{
  "match_status": "VERIFIED_MATCH",   // or "NO_RELIABLE_MATCH"
  "matched_url": "https://…",
  "title": "…",
  "source": "Instagram",
  "evidence_score": 88.5,
  "source_image": "cropped_face.jpg",
  "image_hash_sha256": "9cb452b8…",
  "timestamp": 1741192800
}
```

Simply call `store_and_verify(match_dict)` from your pipeline code:

```python
from partC import store_and_verify

result = store_and_verify(part_b_output)
print(result["status"])  # STORED_AND_VERIFIED | SKIPPED | TX_FAILED
```

---

## License

MIT — hackathon project, use at your own risk.
