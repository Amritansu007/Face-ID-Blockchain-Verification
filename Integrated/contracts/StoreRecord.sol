// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title StoreRecord
 * @dev Stores tamper-evident records of verified face reverse search matches.
 * Each record is keyed by a deterministic SHA-256 dataHash:
 * sha256(image_hash_sha256 : matched_url : evidence_score : timestamp)
 */
contract StoreRecord {
    struct Record {
        string url;
        uint256 timestamp;
        address submittedBy;
    }

    // Mapping from dataHash to on-chain Record
    mapping(bytes32 => Record) private records;

    // Emitted whenever a verified record is anchored on-chain
    event RecordStored(
        bytes32 indexed dataHash,
        string url,
        uint256 timestamp,
        address indexed submittedBy
    );

    /**
     * @notice Store a verified record on-chain.
     * @dev Reverts if a record with the same dataHash already exists (tamper-evident prevention of overwrites).
     * @param dataHash 32-byte SHA-256 hash of the verification payload
     * @param url Matched public URL found by reverse image search
     * @param timestamp Unix timestamp of the verification event
     */
    function storeRecord(
        bytes32 dataHash,
        string calldata url,
        uint256 timestamp
    ) external {
        require(records[dataHash].timestamp == 0, "Record already exists for this dataHash");
        require(bytes(url).length > 0, "URL cannot be empty");
        require(timestamp > 0, "Invalid timestamp");

        records[dataHash] = Record({
            url: url,
            timestamp: timestamp,
            submittedBy: msg.sender
        });

        emit RecordStored(dataHash, url, timestamp, msg.sender);
    }

    /**
     * @notice Retrieve a stored record by its dataHash.
     * @param dataHash 32-byte SHA-256 hash
     * @return url The stored matched URL
     * @return timestamp The timestamp when the match occurred
     * @return submittedBy The Ethereum address that submitted the transaction
     */
    function getRecord(bytes32 dataHash)
        external
        view
        returns (
            string memory url,
            uint256 timestamp,
            address submittedBy
        )
    {
        Record storage rec = records[dataHash];
        require(rec.timestamp != 0, "Record not found");
        return (rec.url, rec.timestamp, rec.submittedBy);
    }

    /**
     * @notice Check if a record exists for a given dataHash without reverting.
     */
    function recordExists(bytes32 dataHash) external view returns (bool) {
        return records[dataHash].timestamp != 0;
    }
}
