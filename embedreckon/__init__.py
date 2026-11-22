"""embedreckon -- the embedding-space auditor and migration certifier.

Audits a vector index somebody else's pipeline built: stamps every
partition CONSISTENT / MIXED / UNVERIFIABLE against a declared
provenance manifest, screens unprovenanced populations for
mixed-space contamination at population level under a published
refusal boundary, measures whether two embedding models' spaces can
be aligned (and with what measured loss), and certifies or refuses a
migration before cutover on completeness, single-space and golden-set
retrieval equivalence.

Pure Python standard library. Offline. No model calls.
"""

__version__ = "0.1.0"

REFUSAL_BOUNDARY = (
    "embedreckon stamps populations and partitions, never individual "
    "vectors; it never attributes a vector to a named model. A "
    "statistical null result is SINGLE-SPACE-CONSISTENT, never "
    "\"verified single-space\": contamination whose negative-tail "
    "signal falls in the registered refusal band (at fixture scale, "
    "roughly the 1-2% foreign-tranche range) stamps UNVERIFIABLE, "
    "not a guessed MIXED and not a false CONSISTENT. Statistics "
    "alone never certify a migration: certification requires the "
    "provenance audit, the completeness check and measured "
    "retrieval equivalence to all pass; any leg that cannot be "
    "evaluated refuses the certificate."
)
