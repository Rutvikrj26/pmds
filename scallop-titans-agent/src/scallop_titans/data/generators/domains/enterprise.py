"""
Domain Generators: Enterprise & Business Category.

Domains:
- RBAC/Access Control
- Org Hierarchy  
- Corporate Structure
- Projects
"""
from __future__ import annotations

from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, EntityType, RelationProperty
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_ENTERPRISE
from scallop_titans.data.generators.entity_pools import (
    PERSON_NAMES, ROLE_NAMES, RESOURCE_NAMES, DEPARTMENT_NAMES, 
    ORG_NAMES, PROJECT_NAMES
)


# ============================================================================
# D7: RBAC/Access Control
# ============================================================================
@register_domain(CATEGORY_ENTERPRISE)
class RBACGenerator(BaseDomainGenerator):
    """Generator for role-based access control relations."""
    
    DOMAIN_NAME = "rbac"
    CATEGORY = CATEGORY_ENTERPRISE
    
    def _default_config(self) -> DomainConfig:
        user = EntityType(name="user", pool=PERSON_NAMES)
        role = EntityType(name="role", pool=ROLE_NAMES)
        resource = EntityType(name="resource", pool=RESOURCE_NAMES)
        
        relations = [
            Relation(
                name="has_role", head_type="user", tail_type="role",
                templates=[
                    "{head} has the role {tail}.",
                    "{head} is assigned the {tail} role.",
                ]
            ),
            Relation(
                name="role_inherits_from", head_type="role", tail_type="role",
                properties={RelationProperty.TRANSITIVE},
                templates=[
                    "{head} role inherits from {tail}.",
                    "The {head} role extends {tail}.",
                ]
            ),
            Relation(
                name="can_access", head_type="user", tail_type="resource",
                templates=[
                    "{head} can access {tail}.",
                    "{head} has access to {tail}.",
                ]
            ),
            Relation(
                name="denied_access_to", head_type="user", tail_type="resource",
                templates=[
                    "{head} is denied access to {tail}.",
                    "{head} cannot access {tail}.",
                ]
            ),
            Relation(
                name="grants_permission", head_type="role", tail_type="resource",
                templates=[
                    "{head} role grants permission to {tail}.",
                    "The {head} role allows access to {tail}.",
                ]
            ),
            Relation(
                name="admin_of", head_type="user", tail_type="resource",
                templates=[
                    "{head} is admin of {tail}.",
                    "{head} administers {tail}.",
                ]
            ),
            Relation(
                name="owner_of", head_type="user", tail_type="resource",
                templates=[
                    "{head} owns {tail}.",
                    "{head} is the owner of {tail}.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="rbac",
            category=CATEGORY_ENTERPRISE,
            relations=relations,
            entity_types={"user": user, "role": role, "resource": resource},
            scallop_rules=[
                "rel can_access(u, r) = has_role(u, role), grants_permission(role, r)",
            ]
        )


# ============================================================================
# D8: Org Hierarchy
# ============================================================================
@register_domain(CATEGORY_ENTERPRISE)
class OrgHierarchyGenerator(BaseDomainGenerator):
    """Generator for organizational hierarchy relations."""
    
    DOMAIN_NAME = "org_hierarchy"
    CATEGORY = CATEGORY_ENTERPRISE
    
    def _default_config(self) -> DomainConfig:
        person = EntityType(name="person", pool=PERSON_NAMES)
        department = EntityType(name="department", pool=DEPARTMENT_NAMES)
        
        relations = [
            Relation(
                name="manages", head_type="person", tail_type="person",
                inverse="managed_by",
                templates=[
                    "{head} manages {tail}.",
                    "{head} is {tail}'s manager.",
                ]
            ),
            Relation(
                name="managed_by", head_type="person", tail_type="person",
                inverse="manages",
                templates=[
                    "{head} is managed by {tail}.",
                    "{tail} manages {head}.",
                ]
            ),
            Relation(
                name="reports_to", head_type="person", tail_type="person",
                properties={RelationProperty.TRANSITIVE},
                templates=[
                    "{head} reports to {tail}.",
                    "{head}'s supervisor is {tail}.",
                ]
            ),
            Relation(
                name="department_of", head_type="person", tail_type="department",
                templates=[
                    "{head} works in {tail}.",
                    "{head} is part of the {tail} department.",
                ]
            ),
            Relation(
                name="head_of", head_type="person", tail_type="department",
                templates=[
                    "{head} is head of {tail}.",
                    "{head} leads the {tail} department.",
                ]
            ),
            Relation(
                name="peer_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is a peer of {tail}.",
                    "{head} and {tail} are at the same level.",
                ]
            ),
            Relation(
                name="delegates_to", head_type="person", tail_type="person",
                templates=[
                    "{head} delegates to {tail}.",
                    "{head} has delegated authority to {tail}.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="org_hierarchy",
            category=CATEGORY_ENTERPRISE,
            relations=relations,
            entity_types={"person": person, "department": department},
        )


# ============================================================================
# D10: Projects
# ============================================================================
@register_domain(CATEGORY_ENTERPRISE)
class ProjectGenerator(BaseDomainGenerator):
    """Generator for project dependency relations."""
    
    DOMAIN_NAME = "projects"
    CATEGORY = CATEGORY_ENTERPRISE
    
    def _default_config(self) -> DomainConfig:
        project = EntityType(name="project", pool=PROJECT_NAMES)
        person = EntityType(name="person", pool=PERSON_NAMES)
        
        relations = [
            Relation(
                name="depends_on", head_type="project", tail_type="project",
                properties={RelationProperty.TRANSITIVE},
                templates=[
                    "{head} depends on {tail}.",
                    "{head} requires {tail} to be completed first.",
                ]
            ),
            Relation(
                name="blocked_by", head_type="project", tail_type="project",
                templates=[
                    "{head} is blocked by {tail}.",
                    "{tail} is blocking {head}.",
                ]
            ),
            Relation(
                name="blocks", head_type="project", tail_type="project",
                inverse="blocked_by",
                templates=[
                    "{head} blocks {tail}.",
                    "{head} is blocking {tail}.",
                ]
            ),
            Relation(
                name="enables", head_type="project", tail_type="project",
                templates=[
                    "{head} enables {tail}.",
                    "Completing {head} allows {tail} to proceed.",
                ]
            ),
            Relation(
                name="predecessor_of", head_type="project", tail_type="project",
                inverse="successor_of",
                templates=[
                    "{head} is the predecessor of {tail}.",
                    "{head} comes before {tail}.",
                ]
            ),
            Relation(
                name="assigned_to", head_type="project", tail_type="person",
                templates=[
                    "{head} is assigned to {tail}.",
                    "{tail} is working on {head}.",
                ]
            ),
            Relation(
                name="owned_by", head_type="project", tail_type="person",
                templates=[
                    "{head} is owned by {tail}.",
                    "{tail} owns {head}.",
                ]
            ),
            Relation(
                name="critical_path_of", head_type="project", tail_type="project",
                templates=[
                    "{head} is on the critical path of {tail}.",
                    "{head} is critical to completing {tail}.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="projects",
            category=CATEGORY_ENTERPRISE,
            relations=relations,
            entity_types={"project": project, "person": person},
        )

# ============================================================================
# D9: Corporate Structure
# ============================================================================
@register_domain(CATEGORY_ENTERPRISE)
class CorpStructureGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "corp_structure"
    CATEGORY = CATEGORY_ENTERPRISE
    
    def _default_config(self) -> DomainConfig:
        org = EntityType("org", ORG_NAMES)
        
        relations = [
            Relation("subsidiary_of", "org", "org",
                     templates=["{head} is a subsidiary of {tail}."]),
            Relation("parent_company_of", "org", "org", inverse="subsidiary_of",
                     templates=["{head} is the parent company of {tail}."]),
            Relation("acquired_by", "org", "org",
                     templates=["{head} was acquired by {tail}."]),
            Relation("merged_with", "org", "org", properties={RelationProperty.SYMMETRIC},
                     templates=["{head} merged with {tail}."]),
            Relation("joint_venture_with", "org", "org", properties={RelationProperty.SYMMETRIC},
                     templates=["{head} is a joint venture with {tail}."]),
        ]
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=relations, entity_types={"org": org})

# ============================================================================
# D11: Procurement
# ============================================================================
@register_domain(CATEGORY_ENTERPRISE)
class ProcurementGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "procurement"
    CATEGORY = CATEGORY_ENTERPRISE
    
    def _default_config(self) -> DomainConfig:
        vendor = EntityType("vendor", ORG_NAMES)
        org = EntityType("org", ORG_NAMES)
        
        relations = [
            Relation("approved_vendor_of", "vendor", "org",
                     templates=["{head} is an approved vendor of {tail}."]),
            Relation("bid_on_contract_for", "vendor", "org",
                     templates=["{head} bid on a contract for {tail}."]),
            Relation("awarded_contract_by", "vendor", "org",
                     templates=["{head} was awarded a contract by {tail}."]),
            Relation("blacklisted_by", "vendor", "org",
                     templates=["{head} is blacklisted by {tail}."]),
        ]
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=relations, entity_types={"vendor": vendor, "org": org})

# ============================================================================
# D13: Audit Trail
# ============================================================================
@register_domain(CATEGORY_ENTERPRISE)
class AuditGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "audit_trail"
    CATEGORY = CATEGORY_ENTERPRISE
    
    def _default_config(self) -> DomainConfig:
        person = EntityType("person", PERSON_NAMES)
        doc = EntityType("doc", ["Report", "Invoice", "Policy", "Memo"])
        
        relations = [
            Relation("created_by", "doc", "person",
                     templates=["{head} was created by {tail}."]),
            Relation("modified_by", "doc", "person",
                     templates=["{head} was modified by {tail}."]),
            Relation("approved_by", "doc", "person",
                     templates=["{head} was approved by {tail}."]),
            Relation("rejected_by", "doc", "person",
                     templates=["{head} was rejected by {tail}."]),
        ]
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=relations, entity_types={"person": person, "doc": doc})

# ============================================================================
# D14: Assets
# ============================================================================
@register_domain(CATEGORY_ENTERPRISE)
class AssetsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "assets"
    CATEGORY = CATEGORY_ENTERPRISE
    
    def _default_config(self) -> DomainConfig:
        asset = EntityType("asset", ["Laptop", "Monitor", "Phone", "Server"])
        person = EntityType("person", PERSON_NAMES)
        
        relations = [
            Relation("assigned_to", "asset", "person",
                     templates=["{head} is assigned to {tail}."]),
            Relation("maintained_by", "asset", "person",
                     templates=["{head} is maintained by {tail}."]),
            Relation("leased_by", "asset", "person",
                     templates=["{head} is leased by {tail}."]),
            Relation("retired_by", "asset", "person",
                     templates=["{head} was retired by {tail}."]),
        ]
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=relations, entity_types={"asset": asset, "person": person})

# ============================================================================
# D15: Budgets
# ============================================================================
@register_domain(CATEGORY_ENTERPRISE)
class BudgetsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "budgets"
    CATEGORY = CATEGORY_ENTERPRISE
    
    def _default_config(self) -> DomainConfig:
        budget = EntityType("budget", ["Marketing Budget", "R&D Budget", "IT Budget"])
        dept = EntityType("dept", DEPARTMENT_NAMES)
        
        relations = [
            Relation("identifies_funding_for", "budget", "dept",
                     templates=["{head} identifies funding for {tail}."]),
            Relation("allocated_to", "budget", "dept",
                     templates=["{head} is allocated to {tail}."]),
            Relation("child_budget_of", "budget", "budget",
                     templates=["{head} is a sub-budget of {tail}."]),
            Relation("overran_by", "dept", "budget",
                     templates=["{head} overran the {tail}."]),
        ]
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=relations, entity_types={"budget": budget, "dept": dept})
