"""
Denylist of known bad domains and predatory journal publishers.

Sources on this list are DROPPED immediately by the Source Scorer agent.
No further processing occurs. The drop is logged to audit_trail.

Seeded from:
- Beall's List of Predatory Journals and Publishers (archived versions)
- Known misinformation / pseudoscience domains
- Retraction Watch flagged publishers

Maintenance:
- Add domains (lowercase, no 'www.') to the appropriate set below.
- This list is intentionally conservative — when in doubt, leave a domain off
  the denylist and let the scoring pipeline assess it.
"""

from __future__ import annotations
from typing import FrozenSet


# Predatory journal publishers 
# Seeded from Beall's List key entries. NOT an exhaustive reproduction.

PREDATORY_JOURNAL_DOMAINS: FrozenSet[str] = frozenset({
    # OMICS Group / iMedPub cluster
    "omicsonline.org",
    "omicsgroup.org",
    "imedpub.com",
    "esciencecentral.org",
    "hilarispublisher.com",
    "longdom.org",
    "scitechnol.com",
    "gavinpublishers.com",

    # Scientific Research Publishing (SCIRP)
    "scirp.org",

    # Medwell Journals
    "medwelljournals.com",

    # Academic Journals
    "academicjournals.org",

    # Science Publishing Group
    "sciencepublishinggroup.com",

    # Insight Medical Publishing
    "insightmedicaljournal.com",

    # IJSciRP cluster
    "ijsrp.org",

    # World Academic Publishing
    "worldacademicpublishing.com",

    # Other frequently flagged
    "thescipub.com",
    "idosi.org",
    "pubs.sciepub.com",
    "globalscienceresearchjournals.org",
    "journalofscienceandresearch.com",
    "ijser.org",
})


# Misinformation / pseudoscience domains 

MISINFORMATION_DOMAINS: FrozenSet[str] = frozenset({
    # Health misinformation
    "naturalnews.com",
    "mercola.com",
    "childrenshealthdefense.org",
    "greenmedinfo.com",
    "infowars.com",
    "beforeitsnews.com",
    "globalresearch.ca",
    "activistpost.com",
    "theepochtimes.com",   # flagged by multiple media literacy orgs for health misinfo
    "ageofautism.com",

    # Anti-science / climate denial
    "co2coalition.org",
    "globalwarmingpolicy.org",
    "climatedepot.com",
    "wattsupwiththat.com",
    "heartland.org",

    # Known conspiracy / unreliable news
    "zerohedge.com",
    "whatreallyhappened.com",
    "veteranstoday.com",
    "yournewswire.com",
    "newspunch.com",
    "100percentfedup.com",
    "conservativetreehouse.com",
    "thegatewaypundit.com",
})


# Combined denylist 

DENYLIST: FrozenSet[str] = PREDATORY_JOURNAL_DOMAINS | MISINFORMATION_DOMAINS


# Lookup functions 

def is_denylisted(domain: str | None) -> bool:
    """
    Check if a domain is on the denylist.
    Handles 'www.' prefix and subdomains.

    Returns True if the domain should be dropped.
    """
    if not domain:
        return False

    domain = domain.lower().strip()

    # Strip www. prefix
    if domain.startswith("www."):
        domain = domain[4:]

    # Direct match
    if domain in DENYLIST:
        return True

    # Check if domain is a subdomain of a denylisted domain
    for bad_domain in DENYLIST:
        if domain.endswith(f".{bad_domain}"):
            return True

    return False


def get_denylist_reason(domain: str | None) -> str:
    """Return a human-readable reason for the denylist hit."""
    if not domain:
        return "unknown"

    domain = domain.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]

    for bad_domain in PREDATORY_JOURNAL_DOMAINS:
        if domain == bad_domain or domain.endswith(f".{bad_domain}"):
            return "predatory_journal"

    for bad_domain in MISINFORMATION_DOMAINS:
        if domain == bad_domain or domain.endswith(f".{bad_domain}"):
            return "misinformation_source"

    return "denylisted"