# IP / legal compliance record — embedreckon

- **Original name:** "embedreckon" was coined for this project.
  Collision-checked against GitHub, PyPI, and npm on 2026-11-22 in
  the landscape scan and re-verified the same day at build time
  (GitHub repository search: 0 results; PyPI: 404; npm: 404). Full
  record in [NAME_CHECK.md](NAME_CHECK.md).
- **Original code and copy:** all code and documentation in this
  repository were written for this project. The motivating idea —
  that a vector index should be audited against the provenance it
  declares, screened for mixed embedding spaces, and certified
  before a migration cutover — came from a landscape analysis of
  vector stores, embedding providers, migration runbooks and
  retrieval-evaluation tools; no third-party code, text, or
  documentation was copied from any of them. The seed scenario (a
  model update silently breaking search over a large index) was
  taken as an idea only, per the reference rule.
- **No entrant code reused, no entrant dependency:** v1 imports
  nothing outside the Python standard library. In particular:
  - **Vector stores (Qdrant, Weaviate, Milvus, Chroma, LanceDB,
    Vespa, OpenSearch — Apache-2.0 / BSD-3; pgvector — PostgreSQL
    License; Pinecone — proprietary):** not used, not imported,
    not vendored. embedreckon reads *exports* a user produces from
    their own store; it connects to nothing.
  - **Elasticsearch (AGPLv3 / SSPL / Elastic License 2.0):** no
    code used, no dependency; reading exports constrains nothing.
  - **Embedding providers (OpenAI, Voyage, Cohere, Google —
    proprietary APIs):** no calls are made in v1 and the tool
    never holds API keys. The compatibility probe consumes paired
    embeddings the user supplies, generated under the user's own
    accounts and each provider's terms.
  - **Eval tools and frameworks (RAGAS, TruLens, DeepEval,
    Promptfoo, LangChain, LlamaIndex, MTEB — Apache-2.0 / MIT):**
    no code used; their documented behaviour informed the
    landscape scan only. (The scan flags DeepEval's LICENSE as
    not directly verified; nothing here depends on it.)
- **The population-only / no-attribution boundary is an original
  design decision, recorded here as such.** The landscape scan's
  dual-use flag: embedding-space translation and per-vector model
  attribution are dual-use capabilities (the Vec2Vec line of
  research demonstrates attribute inference from unattributed
  vector databases). embedreckon's verdict model therefore stamps
  *populations and partitions only*, never attributes an
  individual vector to a named model, and ships **no translator**:
  the compatibility probe measures a pair the user *declares*, on
  pairs the user *supplies*, and its alignment map is computed
  inside the measurement and discarded — it is not an exportable
  artefact. This boundary is stated in every report the tool
  emits (the `refusal_boundary` header) and in the README.
- **Kill criterion (iii) re-checked at build time (2026-11-22):**
  no vector store or eval vendor ships an independent mixed-space
  audit or pre-cutover migration certification emitting
  per-partition verdicts. The nearest adjacent entrant is an
  ingest-side signing/policy gateway (write-time prevention,
  waitlist-stage): different layer, no code or copy used.
- **LICENSE:** MIT, in the tree. v1 has zero third-party
  dependencies, so there is no dependency-license surface.
