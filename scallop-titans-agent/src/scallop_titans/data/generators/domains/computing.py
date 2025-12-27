"""
Domain Generators: Computing & Networks Category.

Domains:
- Network Routing
- Service Dependencies
- File Permissions
"""
from __future__ import annotations

from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, EntityType, RelationProperty
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_COMPUTING
from scallop_titans.data.generators.entity_pools import SERVER_NAMES, PERSON_NAMES


# ============================================================================
# D33: Network Routing
# ============================================================================
@register_domain(CATEGORY_COMPUTING)
class NetworkRoutingGenerator(BaseDomainGenerator):
    """Generator for network routing relations."""
    
    DOMAIN_NAME = "network_routing"
    CATEGORY = CATEGORY_COMPUTING
    
    def _default_config(self) -> DomainConfig:
        server = EntityType(name="server", pool=SERVER_NAMES)
        
        relations = [
            Relation(
                name="network_reachable", head_type="server", tail_type="server",
                properties={RelationProperty.TRANSITIVE},
                templates=[
                    "{head} can reach {tail}.",
                    "{head} has network connectivity to {tail}.",
                ]
            ),
            Relation(
                name="routes_to", head_type="server", tail_type="server",
                templates=[
                    "{head} routes to {tail}.",
                    "Traffic from {head} goes to {tail}.",
                ]
            ),
            Relation(
                name="next_hop_of", head_type="server", tail_type="server",
                templates=[
                    "{head} is the next hop to {tail}.",
                    "To reach {tail}, go through {head}.",
                ]
            ),
            Relation(
                name="blocked_at_firewall", head_type="server", tail_type="server",
                templates=[
                    "{head} is blocked from {tail} by firewall.",
                    "Firewall blocks {head} from reaching {tail}.",
                ]
            ),
            Relation(
                name="load_balanced_to", head_type="server", tail_type="server",
                templates=[
                    "{head} is load balanced to {tail}.",
                    "Load balancer sends {head} traffic to {tail}.",
                ]
            ),
            Relation(
                name="failover_to", head_type="server", tail_type="server",
                templates=[
                    "{head} fails over to {tail}.",
                    "If {head} fails, traffic goes to {tail}.",
                ]
            ),
            Relation(
                name="primary_route_for", head_type="server", tail_type="server",
                templates=[
                    "{head} is the primary route for {tail}.",
                    "{head} handles primary traffic to {tail}.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="network_routing",
            category=CATEGORY_COMPUTING,
            relations=relations,
            entity_types={"server": server},
            scallop_rules=[
                "rel network_reachable(a, c) = network_reachable(a, b), network_reachable(b, c)",
                "rel blocked(a, c) = blocked_at_firewall(a, b), routes_to(b, c)",
            ]
        )


# ============================================================================
# D34: Service Dependencies
# ============================================================================
@register_domain(CATEGORY_COMPUTING)
class ServiceDependencyGenerator(BaseDomainGenerator):
    """Generator for service dependency relations."""
    
    DOMAIN_NAME = "service_dependencies"
    CATEGORY = CATEGORY_COMPUTING
    
    def _default_config(self) -> DomainConfig:
        service = EntityType(name="service", prefix="svc-")
        
        relations = [
            Relation(
                name="service_calls", head_type="service", tail_type="service",
                templates=[
                    "{head} calls {tail}.",
                    "{head} makes API requests to {tail}.",
                ]
            ),
            Relation(
                name="depends_on_service", head_type="service", tail_type="service",
                properties={RelationProperty.TRANSITIVE},
                templates=[
                    "{head} depends on {tail}.",
                    "{head} requires {tail} to function.",
                ]
            ),
            Relation(
                name="circuit_breaks_if", head_type="service", tail_type="service",
                templates=[
                    "{head} circuit breaks if {tail} fails.",
                    "{head} stops calling {tail} after failures.",
                ]
            ),
            Relation(
                name="fallback_to", head_type="service", tail_type="service",
                templates=[
                    "{head} falls back to {tail}.",
                    "If primary fails, {head} uses {tail}.",
                ]
            ),
            Relation(
                name="publishes_to_queue", head_type="service", tail_type="service",
                templates=[
                    "{head} publishes to {tail} queue.",
                    "{head} sends messages to {tail}.",
                ]
            ),
            Relation(
                name="subscribes_to_topic", head_type="service", tail_type="service",
                templates=[
                    "{head} subscribes to {tail} topic.",
                    "{head} receives events from {tail}.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="service_dependencies",
            category=CATEGORY_COMPUTING,
            relations=relations,
            entity_types={"service": service},
        )


# ============================================================================
# D35: File Permissions
# ============================================================================
@register_domain(CATEGORY_COMPUTING)
class FilePermissionsGenerator(BaseDomainGenerator):
    """Generator for file permission relations."""
    
    DOMAIN_NAME = "file_permissions"
    CATEGORY = CATEGORY_COMPUTING
    
    def _default_config(self) -> DomainConfig:
        user = EntityType(name="user", pool=PERSON_NAMES)
        file = EntityType(name="file", prefix="file_")
        group = EntityType(name="group", prefix="group_")
        
        relations = [
            Relation(
                name="file_owner_of", head_type="user", tail_type="file",
                templates=[
                    "{head} owns {tail}.",
                    "{head} is the owner of {tail}.",
                ]
            ),
            Relation(
                name="file_group_of", head_type="group", tail_type="file",
                templates=[
                    "{head} is the group for {tail}.",
                    "{tail} belongs to group {head}.",
                ]
            ),
            Relation(
                name="readable_by_user", head_type="file", tail_type="user",
                templates=[
                    "{head} is readable by {tail}.",
                    "{tail} can read {head}.",
                ]
            ),
            Relation(
                name="writable_by_user", head_type="file", tail_type="user",
                templates=[
                    "{head} is writable by {tail}.",
                    "{tail} can write to {head}.",
                ]
            ),
            Relation(
                name="executable_by_user", head_type="file", tail_type="user",
                templates=[
                    "{head} is executable by {tail}.",
                    "{tail} can execute {head}.",
                ]
            ),
            Relation(
                name="inherits_permissions", head_type="file", tail_type="file",
                properties={RelationProperty.TRANSITIVE},
                templates=[
                    "{head} inherits permissions from {tail}.",
                    "{head} gets permissions from parent {tail}.",
                ]
            ),
            Relation(
                name="member_of_group", head_type="user", tail_type="group",
                templates=[
                    "{head} is a member of {tail}.",
                    "{head} belongs to group {tail}.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="file_permissions",
            category=CATEGORY_COMPUTING,
            relations=relations,
            entity_types={"user": user, "file": file, "group": group},
            scallop_rules=[
                "rel readable_by_user(f, u) = file_group_of(g, f), member_of_group(u, g)",
            ]
        )

# ============================================================================
# D41: Containers & Orchestration
# ============================================================================
@register_domain(CATEGORY_COMPUTING)
class ContainerGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "containers"
    CATEGORY = CATEGORY_COMPUTING
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("deployed_to", "Container", "Node", templates=["{head} is deployed to {tail}."]),
            Relation("contains_service", "Pod", "Service", templates=["{head} contains {tail}."]),
            Relation("scaled_by", "ReplicaSet", "Policy", templates=["{head} scaled by {tail}."]),
            Relation("exposed_via", "Service", "Ingress", templates=["{head} exposed via {tail}."]),
            Relation("mounts_volume", "Container", "Volume", templates=["{head} mounts {tail}."]),
        ], entity_types={"Container": EntityType("Cont", ["c1"]), "Node": EntityType("Node", ["n1"]), "Pod": EntityType("Pod", ["p1"]), "Service": EntityType("Svc", ["s1"]), "ReplicaSet": EntityType("RS", ["rs1"]), "Policy": EntityType("Pol", ["HPA"]), "Ingress": EntityType("Ing", ["ing1"]), "Volume": EntityType("Vol", ["vol1"])})

# ============================================================================
# D42: Databases
# ============================================================================
@register_domain(CATEGORY_COMPUTING)
class DatabaseGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "databases"
    CATEGORY = CATEGORY_COMPUTING
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("indexed_on", "Table", "Column", templates=["{head} indexed on {tail}."]),
            Relation("foreign_key_to", "Column", "Table", templates=["{head} is FK to {tail}."]),
            Relation("stored_in", "Table", "Database", templates=["{head} stored in {tail}."]),
            Relation("replicated_to", "Database", "Database", templates=["{head} replicated to {tail}."]),
            Relation("sharded_by", "Table", "Key", templates=["{head} sharded by {tail}."]),
        ], entity_types={"Table": EntityType("Tab", ["Users"]), "Column": EntityType("Col", ["id"]), "Database": EntityType("DB", ["MainDB"]), "Key": EntityType("Key", ["user_id"])})

# ============================================================================
# D43: Version Control
# ============================================================================
@register_domain(CATEGORY_COMPUTING)
class VersionControlGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "version_control"
    CATEGORY = CATEGORY_COMPUTING
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("committed_by", "Commit", "Developer", templates=["{head} committed by {tail}."]),
            Relation("merged_into", "Branch", "Branch", templates=["{head} merged into {tail}."]),
            Relation("tagged_as", "Commit", "Tag", templates=["{head} tagged as {tail}."]),
            Relation("resolves_issue", "PR", "Issue", templates=["{head} resolves {tail}."]),
            Relation("approved_by", "PR", "Reviewer", templates=["{head} approved by {tail}."]),
        ], entity_types={"Commit": EntityType("Commit", ["c123"]), "Developer": EntityType("Dev", ["Alice"]), "Branch": EntityType("Br", ["main"]), "Tag": EntityType("Tag", ["v1.0"]), "PR": EntityType("PR", ["#101"]), "Issue": EntityType("Iss", ["#99"]), "Reviewer": EntityType("Rev", ["Bob"])})
