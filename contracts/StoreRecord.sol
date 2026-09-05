// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title StoreRecord
 * @notice Part C of the FaceID hackathon pipeline.
 *         Stores tamper-evident records keyed by a SHA-256 data hash.
 *         Each hash may only be written once — duplicates revert.
 */
contract StoreRecord {

    struct Record {
        string  url;
        uint256 timestamp;
        address submittedBy;
        bool    exists;          // guard against silent overwrites
    }

    mapping(bytes32 => Record) private records;

    event RecordStored(
        bytes32 indexed dataHash,
        string  url,
        uint256 timestamp,
        address indexed submittedBy
    );

    /**
     * @notice Write a new record on-chain.
     * @param dataHash   SHA-256 digest that uniquely identifies the match.
     * @param url        The matched URL from the reverse-image search.
     * @param timestamp  Unix epoch of the original match.
     */
    function storeRecord(
        bytes32 dataHash,
        string memory url,
        uint256 timestamp
    ) public {
        require(!records[dataHash].exists, "Record already exists for this dataHash");

        records[dataHash] = Record({
            url:         url,
            timestamp:   timestamp,
            submittedBy: msg.sender,
            exists:      true
        });

        emit RecordStored(dataHash, url, timestamp, msg.sender);
    }

    /**
     * @notice Read back a previously stored record.
     * @param dataHash  The same SHA-256 digest used during storage.
     * @return url         The matched URL.
     * @return timestamp   The Unix epoch stored with the record.
     * @return submittedBy The address that submitted the record.
     */
    function getRecord(bytes32 dataHash)
        public
        view
        returns (
            string memory url,
            uint256 timestamp,
            address submittedBy
        )
    {
        Record storage r = records[dataHash];
        require(r.exists, "No record found for this dataHash");
        return (r.url, r.timestamp, r.submittedBy);
    }
}
