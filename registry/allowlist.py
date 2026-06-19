"""
Allowlist of trusted domains and DOI prefixes.

Sources on this list receive:
- trusted_source = True
- credibility floor of 0.90 (bypasses metadata classification)
- Included because they are authoritative, peer-reviewed, or government/WHO institutional sources.

Maintenance:
- Add new entries to the appropriate set below.
- DOI prefixes: format is "10.XXXX" (the registrant prefix, without the suffix).
"""

from __future__ import annotations
from typing import FrozenSet


# Trusted domains 

TRUSTED_DOMAINS: FrozenSet[str] = frozenset({
    # WHO / International health organizations
    "who.int",
    "euro.who.int",
    "paho.org",

    # US Government health agencies
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "cdc.gov",
    "fda.gov",
    "hhs.gov",
    "cms.gov",
    "ahrq.gov",
    "niddk.nih.gov",
    "nhlbi.nih.gov",
    "nci.nih.gov",

    # US Government statistical / policy agencies
    "bls.gov",
    "census.gov",
    "bea.gov",
    "data.gov",
    "gao.gov",
    "cbo.gov",
    "federalreserve.gov",
    "usda.gov",
    "epa.gov",
    "energy.gov",

    # UK health / government
    "gov.uk",
    "nhs.uk",
    "nice.org.uk",

    # EU / ECDC
    "ecdc.europa.eu",
    "europa.eu",
    "ema.europa.eu",

    # Cochrane / evidence synthesis
    "cochrane.org",
    "cochranelibrary.com",

    # Major academic indexing / preprint (with caveat: preprints are not peer-reviewed)
    "pubmed.gov",
    "ncbi.nlm.nih.gov",

    # IPCC / scientific panels
    "ipcc.ch",
    "ipbes.net",

    # Major statistical bodies
    "worldbank.org",
    "imf.org",
    "oecd.org",
    "un.org",
    "ourworldindata.org",  

    # Standards bodies
    "iso.org",
    "nist.gov",
    "ieee.org",
})

#    Trusted DOI prefixes                                                       ─
# These are Crossref registrant prefixes for major academic publishers
# whose peer-review standards are considered high.

TRUSTED_DOI_PREFIXES: FrozenSet[str] = frozenset({
    # Nature Publishing Group
    "10.1038",
    # NEJM
    "10.1056",
    # JAMA Network
    "10.1001",
    # Lancet
    "10.1016",
    # BMJ
    "10.1136",
    # Cochrane Reviews
    "10.1002",
    # PLOS
    "10.1371",
    # Cell Press
    "10.1016",
    # Science / AAAS
    "10.1126",
    # PNAS
    "10.1073",
    # Annals of Internal Medicine
    "10.7326",
    # Oxford University Press
    "10.1093",
    # Cambridge University Press
    "10.1017",
    # Springer / BMC
    "10.1007",
    "10.1186",
    # Wiley
    "10.1111",
    "10.1002",
    # Taylor & Francis
    "10.1080",
    # American Psychological Association
    "10.1037",
    # American Medical Association
    "10.1001",
})


# Lookup functions                                                           ─

def is_trusted_domain(domain: str | None) -> bool:
    """
    Check if a domain is on the allowlist.
    Handles subdomains: 'pubmed.ncbi.nlm.nih.gov' matches 'nih.gov'.
    """
    if not domain:
        return False

    domain = domain.lower().strip()

    # Direct match
    if domain in TRUSTED_DOMAINS:
        return True

    # Subdomain match: check if domain ends with any trusted domain
    for trusted in TRUSTED_DOMAINS:
        if domain.endswith(f".{trusted}") or domain == trusted:
            return True

    return False


def is_trusted_doi(doi: str | None) -> bool:
    """
    Check if a DOI's registrant prefix is on the trusted list.
    Input: full DOI like '10.1038/s41586-021-03658-1'
    """
    if not doi:
        return False

    doi = doi.strip().lower()
    # Extract the prefix (format: 10.XXXX/...)
    parts = doi.split("/")
    if len(parts) < 2:
        return False

    prefix = parts[0]  # e.g. "10.1038"
    return prefix in TRUSTED_DOI_PREFIXES