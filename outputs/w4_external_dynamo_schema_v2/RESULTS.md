# Official Dynamo historical scNT-seq releases

Downloaded the exact Dropbox URLs in Dynamo commit `368dc55e63dc5bbd100d16c566cc5206c0307658`.
Total bytes: 617,385,142. Stanford/Figshare alternatives failed; historical official mirrors succeeded.

- Splicing: 13,476 × 44,021, layers `spliced/unspliced`, no embeddings or inferred velocities.
- Labeling: 3060 × 24,078, layers `new/total`; an author-selected subset, not raw unselected counts.
- File SHA256 values and URLs: `downloads.json`; HDF5 schemas: `schema.json`.
- Historical labeling values were not used in the independent GEO-count pairing.

Schema v1 was interrupted by JSON serialization of a NumPy integer. Downloads were valid;
v2 reused the files, corrected serialization, and completed. No biological analysis performed.
