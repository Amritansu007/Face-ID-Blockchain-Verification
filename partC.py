"""
partC.py — Blockchain Verification (Part C)
FaceID Hackathon Pipeline: Face Detection → Reverse Image Search → Blockchain Verification

Writes tamper-evident records to an Ethereum Sepolia testnet contract and
verifies them by reading back on-chain data and recomputing the hash.
"""

import hashlib
import json
import logging
import os
import re
import sys
import warnings

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress dotenv warnings on unquoted multiline variables
warnings.filterwarnings("ignore")
logging.getLogger("dotenv").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from web3 import Web3

# ---------------------------------------------------------------------------
# Default StoreRecord ABI (used as fallback or when .env ABI has formatting issues)
# ---------------------------------------------------------------------------

STORE_RECORD_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
            {"indexed": False, "internalType": "string", "name": "url", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"indexed": True, "internalType": "address", "name": "submittedBy", "type": "address"}
        ],
        "name": "RecordStored",
        "type": "event"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
            {"internalType": "string", "name": "url", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "storeRecord",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "dataHash", "type": "bytes32"}
        ],
        "name": "getRecord",
        "outputs": [
            {"internalType": "string", "name": "url", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "address", "name": "submittedBy", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# ---------------------------------------------------------------------------
# Environment & Web3 setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env in the working directory

RPC_URL          = os.getenv("RPC_URL")
PRIVATE_KEY      = os.getenv("PRIVATE_KEY")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
CONTRACT_ABI     = os.getenv("CONTRACT_ABI")  # JSON string or multiline in .env


def _get_web3() -> Web3:
    """Return a connected Web3 instance or exit with a clear error."""
    if not RPC_URL:
        sys.exit("ERROR: RPC_URL is not set in .env")
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        sys.exit(f"ERROR: Cannot connect to RPC at {RPC_URL}")
    return w3


def _resolve_abi() -> list:
    """
    Resolve the contract ABI gracefully:
    1. Try parsing CONTRACT_ABI directly if valid JSON.
    2. Try extracting multiline JSON from .env if unquoted.
    3. Fall back to standard StoreRecord ABI.
    """
    if CONTRACT_ABI:
        try:
            return json.loads(CONTRACT_ABI)
        except Exception:
            pass

    env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                raw_env = f.read()
            match = re.search(r"CONTRACT_ABI\s*=\s*(\[[\s\S]*?\])(?:\r?\n\s*[A-Za-z0-9_]+\s*=|\s*$)", raw_env)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass

    return STORE_RECORD_ABI


def _get_contract(w3: Web3):
    """Return a Contract instance bound to the deployed StoreRecord address."""
    if not CONTRACT_ADDRESS:
        sys.exit("ERROR: CONTRACT_ADDRESS is not set in .env")
    abi = _resolve_abi()
    return w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=abi,
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def build_data_hash(match: dict) -> bytes:
    """
    Produce a deterministic SHA-256 hash of the match payload.

    The canonical string is:
        {image_hash_sha256}:{matched_url}:{evidence_score:.1f}:{timestamp}

    evidence_score is always formatted to exactly 1 decimal place so that
    re-hashing later yields an identical digest.
    """
    canonical = (
        f"{match['image_hash_sha256']}:"
        f"{match['matched_url']}:"
        f"{match['evidence_score']:.1f}:"
        f"{match['timestamp']}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).digest()   # 32 bytes


def _send_signed_tx(w3: Web3, tx, private_key: str) -> dict:
    """
    Sign and broadcast a transaction; wait for the receipt.

    Handles the attribute-name difference between web3.py v5
    (signed_tx.rawTransaction) and v6+ (signed_tx.raw_transaction).
    """
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)

    # web3.py v6+ uses .raw_transaction; v5 uses .rawTransaction
    raw = getattr(signed_tx, "raw_transaction", None) or getattr(
        signed_tx, "rawTransaction", None
    )
    if raw is None:
        raise AttributeError(
            "Cannot locate raw transaction bytes on the signed tx object — "
            "check your web3.py version."
        )

    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    return receipt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_and_verify(match: dict) -> dict:
    """
    End-to-end flow: guard → hash → store on-chain → verify read-back.

    Returns a result dict suitable for JSON serialisation.
    """
    # ── HARD FAIL-SAFE ──────────────────────────────────────────────────
    if match.get("match_status") != "VERIFIED_MATCH":
        return {
            "status": "SKIPPED",
            "reason": (
                f"match_status is '{match.get('match_status')}' — "
                "only VERIFIED_MATCH records are written on-chain."
            ),
        }

    # ── Setup ───────────────────────────────────────────────────────────
    w3       = _get_web3()
    contract = _get_contract(w3)
    account  = w3.eth.account.from_key(PRIVATE_KEY)
    sender   = account.address

    # ── Build hash ──────────────────────────────────────────────────────
    data_hash = build_data_hash(match)

    # ── Build & send storeRecord transaction ────────────────────────────
    nonce = w3.eth.get_transaction_count(sender)

    # Build fee params — use EIP-1559 fields on supported chains (web3 v6+/v8),
    # fall back to legacy gasPrice for older web3 versions or non-1559 chains.
    tx_params: dict = {
        "from":  sender,
        "nonce": nonce,
        "gas":   300_000,
    }
    try:
        latest = w3.eth.get_block("latest")
        if "baseFeePerGas" in latest:
            # EIP-1559 chain — set maxFeePerGas / maxPriorityFeePerGas
            base_fee = latest["baseFeePerGas"]
            tx_params["maxPriorityFeePerGas"] = w3.to_wei(1.5, "gwei")
            tx_params["maxFeePerGas"] = base_fee * 2 + tx_params["maxPriorityFeePerGas"]
        else:
            tx_params["gasPrice"] = w3.eth.gas_price
    except Exception:
        tx_params["gasPrice"] = w3.eth.gas_price

    tx = contract.functions.storeRecord(
        data_hash,
        match["matched_url"],
        match["timestamp"],
    ).build_transaction(tx_params)

    print(f"⏳ Sending storeRecord tx from {sender} …")
    receipt = _send_signed_tx(w3, tx, PRIVATE_KEY)

    tx_hash_hex = receipt["transactionHash"].hex()
    success     = receipt["status"] == 1

    print(f"{'✅' if success else '❌'} Tx mined: {tx_hash_hex}  (status={receipt['status']})")

    if not success:
        existing_info = None
        try:
            existing_info = verify_record(data_hash, match, w3=w3, contract=contract)
        except Exception:
            pass

        return {
            "status":  "TX_FAILED_DUPLICATE" if existing_info else "TX_FAILED",
            "tx_hash": tx_hash_hex,
            "detail":  (
                "Transaction reverted — record already exists on-chain for this dataHash."
                if existing_info
                else "Transaction reverted — check contract execution logic."
            ),
            "data_hash": "0x" + data_hash.hex(),
            "existing_record_verification": existing_info,
        }

    # ── Verify ──────────────────────────────────────────────────────────
    verification = verify_record(data_hash, match, w3=w3, contract=contract)

    return {
        "status":       "STORED_AND_VERIFIED" if verification["hash_match"] else "STORED_BUT_MISMATCH",
        "tx_hash":      tx_hash_hex,
        "block_number": receipt["blockNumber"],
        "data_hash":    "0x" + data_hash.hex(),
        "verification": verification,
    }


def verify_record(
    data_hash: bytes,
    original_match: dict,
    *,
    w3=None,
    contract=None,
) -> dict:
    """
    Read a record back from the chain and verify it against the original
    match payload by recomputing the hash.

    Parameters
    ----------
    data_hash : bytes
        The 32-byte SHA-256 digest that was used as the mapping key.
    original_match : dict
        The same match dict used to produce ``data_hash``.
    w3, contract : optional
        Re-use existing objects; created automatically if omitted.
    """
    if w3 is None:
        w3 = _get_web3()
    if contract is None:
        contract = _get_contract(w3)

    url, timestamp, submitted_by = contract.functions.getRecord(data_hash).call()

    # Recompute hash from the original match to confirm integrity
    recomputed = build_data_hash(original_match)
    hashes_match = recomputed == data_hash

    return {
        "hash_match":    hashes_match,
        "indicator":     "✅ Verified" if hashes_match else "❌ Hash mismatch",
        "on_chain_url":       url,
        "on_chain_timestamp": timestamp,
        "on_chain_submitter": submitted_by,
        "recomputed_hash":    "0x" + recomputed.hex(),
        "expected_hash":      "0x" + data_hash.hex(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    demo_path = os.path.join(os.path.dirname(__file__) or ".", "demo_match.json")

    if not os.path.exists(demo_path):
        sys.exit(f"ERROR: {demo_path} not found — create it first.")

    with open(demo_path, "r", encoding="utf-8") as f:
        match_payload = json.load(f)

    # Optional --fresh argument to test writing a new on-chain record with a new timestamp
    if "--fresh" in sys.argv:
        match_payload["timestamp"] = int(time.time())
        print("⚡ '--fresh' flag detected — using current timestamp for unique on-chain write.")

    print("=" * 60)
    print("  Part C — Blockchain Verification")
    print("=" * 60)
    print(f"\nInput payload:\n{json.dumps(match_payload, indent=2)}\n")

    result = store_and_verify(match_payload)

    print("\n" + "=" * 60)
    print("  Result")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
