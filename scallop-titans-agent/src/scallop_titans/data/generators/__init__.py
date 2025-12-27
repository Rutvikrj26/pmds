"""
Domain generators package initialization.

Imports all domain generators to register them.
"""
from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator,
    DomainConfig,
    Relation,
    EntityType,
    RelationProperty,
    GeneratedSample,
)
from scallop_titans.data.generators.domain_registry import (
    register_domain,
    get_domain_generator,
    get_all_domains,
    get_domains_by_category,
    get_all_categories,
    print_registry_stats,
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
)

# Import domain modules to trigger registration
from scallop_titans.data.generators.domains import social
from scallop_titans.data.generators.domains import enterprise
from scallop_titans.data.generators.domains import computing
from scallop_titans.data.generators.domains import healthcare
from scallop_titans.data.generators.domains import legal
from scallop_titans.data.generators.domains import logistics
from scallop_titans.data.generators.domains import robotics
from scallop_titans.data.generators.domains import academic
from scallop_titans.data.generators.domains import finance
from scallop_titans.data.generators.domains import gaming
from scallop_titans.data.generators.domains import geographic
from scallop_titans.data.generators.domains import misc

__all__ = [
    "BaseDomainGenerator",
    "DomainConfig", 
    "Relation",
    "EntityType",
    "RelationProperty",
    "GeneratedSample",
    "register_domain",
    "get_domain_generator",
    "get_all_domains",
    "get_domains_by_category",
    "get_all_categories",
    "print_registry_stats",
]
