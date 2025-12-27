"""
Domain Registry: Central registration and lookup for all 66 domains.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scallop_titans.data.generators.base_generator import BaseDomainGenerator


# Registry maps domain name -> generator class
_DOMAIN_REGISTRY: dict[str, type["BaseDomainGenerator"]] = {}
_CATEGORY_REGISTRY: dict[str, list[str]] = {}


def register_domain(category: str):
    """Decorator to register a domain generator."""
    def decorator(cls: type["BaseDomainGenerator"]):
        domain_name = cls.DOMAIN_NAME
        if not domain_name:
            raise ValueError(f"Generator {cls.__name__} must define DOMAIN_NAME")
            
        _DOMAIN_REGISTRY[domain_name] = cls
        
        if category not in _CATEGORY_REGISTRY:
            _CATEGORY_REGISTRY[category] = []
        _CATEGORY_REGISTRY[category].append(domain_name)
        
        return cls
    return decorator


def get_domain_generator(domain_name: str) -> type["BaseDomainGenerator"]:
    """Get a domain generator class by name."""
    if domain_name not in _DOMAIN_REGISTRY:
        raise KeyError(f"Unknown domain: {domain_name}. Available: {list(_DOMAIN_REGISTRY.keys())}")
    return _DOMAIN_REGISTRY[domain_name]


def get_all_domains() -> list[str]:
    """Get all registered domain names."""
    return list(_DOMAIN_REGISTRY.keys())


def get_domains_by_category(category: str) -> list[str]:
    """Get all domains in a category."""
    return _CATEGORY_REGISTRY.get(category, [])


def get_all_categories() -> list[str]:
    """Get all registered categories."""
    return list(_CATEGORY_REGISTRY.keys())


def print_registry_stats():
    """Print registry statistics."""
    print(f"Registered Domains: {len(_DOMAIN_REGISTRY)}")
    print(f"Categories: {len(_CATEGORY_REGISTRY)}")
    for cat, domains in _CATEGORY_REGISTRY.items():
        print(f"  {cat}: {len(domains)} domains")


# --- Domain Category Constants ---

CATEGORY_SOCIAL = "Social & Family"
CATEGORY_ENTERPRISE = "Enterprise & Business"
CATEGORY_HEALTHCARE = "Healthcare & Medicine"
CATEGORY_LEGAL = "Legal & Compliance"
CATEGORY_LOGISTICS = "Logistics & Transport"
CATEGORY_ROBOTICS = "Robotics & Manufacturing"
CATEGORY_COMPUTING = "Computing & Networks"
CATEGORY_ACADEMIC = "Academic & Education"
CATEGORY_FINANCE = "Finance & Economics"
CATEGORY_GAMING = "Gaming & Entertainment"
CATEGORY_GEOGRAPHIC = "Geographic & Environment"
CATEGORY_MISC = "Miscellaneous"

ALL_CATEGORIES = [
    CATEGORY_SOCIAL,
    CATEGORY_ENTERPRISE,
    CATEGORY_HEALTHCARE,
    CATEGORY_LEGAL,
    CATEGORY_LOGISTICS,
    CATEGORY_ROBOTICS,
    CATEGORY_COMPUTING,
    CATEGORY_ACADEMIC,
    CATEGORY_FINANCE,
    CATEGORY_GAMING,
    CATEGORY_GEOGRAPHIC,
    CATEGORY_MISC,
]
