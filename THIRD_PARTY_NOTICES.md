# Third-party source and notices

ICORE includes reviewed source copies so the native builds do not need to download
solver dependencies. The archives are unchanged; only their outer filenames are
made readable. Exact version identifiers and automatic integrity checks are internal
to the build helpers, not user settings.

| Component | Purpose | Retained license / notice |
|---|---|---|
| BLASFEO | Packed small/medium matrix kernels | `icore/vendor/blasfeo/BLASFEO_LICENSE.txt` |
| HPIPM | Comparison solver | `icore/vendor/hpipm/LICENSE.txt` |
| QOCO | Comparison solver | `icore/vendor/qoco/LICENSE` |
| QDLDL in the QOCO archive | Sparse factorization | `icore/vendor/qoco/QDLDL_LICENSE` |
| QDLDL in ICORE's chain path | Augmented sparse recovery | `icore/src/chain_sparse/qdldl/LICENSE` |
| AMD / SuiteSparse_config | Sparse ordering support | `icore/src/chain_sparse/amd/AMD-License.txt`, `SuiteSparse-config-License.txt`, and `LICENSE-NOTICE.txt` |
| Retained QOCO adaptation notices | Chain sparse support provenance | `icore/src/chain_sparse/QOCO-License.txt` |

`icore/src/chain_sparse/MODIFICATIONS.txt` documents retained local adaptations.
`icore/vendor/blasfeo/patches/` contains the existing reviewed numerical repairs.
The original notices inside the source archives remain intact. This document is a
navigation aid, not a replacement for those licenses or a new ICORE license.
