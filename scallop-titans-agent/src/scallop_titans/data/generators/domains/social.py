"""
Domain Generators: Social & Family Category.

Domains:
- Kinship
- Marriage & Partnership
- Extended Family
- Friendship
- Professional Networks
- Groups & Membership
"""
from __future__ import annotations

from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, EntityType, RelationProperty
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_SOCIAL
from scallop_titans.data.generators.entity_pools import PERSON_NAMES


# ============================================================================
# D1: Kinship
# ============================================================================
@register_domain(CATEGORY_SOCIAL)
class KinshipGenerator(BaseDomainGenerator):
    """Generator for kinship/family relations."""
    
    DOMAIN_NAME = "kinship"
    CATEGORY = CATEGORY_SOCIAL
    
    def _default_config(self) -> DomainConfig:
        person = EntityType(name="person", pool=PERSON_NAMES)
        
        relations = [
            Relation(
                name="parent_of", head_type="person", tail_type="person",
                inverse="child_of",
                templates=[
                    "{head} is the parent of {tail}.",
                    "{tail}'s parent is {head}.",
                    "{head} has a child named {tail}.",
                ]
            ),
            Relation(
                name="child_of", head_type="person", tail_type="person",
                inverse="parent_of",
                templates=[
                    "{head} is the child of {tail}.",
                    "{tail} has a child named {head}.",
                ]
            ),
            Relation(
                name="mother_of", head_type="person", tail_type="person",
                templates=[
                    "{head} is the mother of {tail}.",
                    "{tail}'s mother is {head}.",
                    "{tail}'s mom is {head}.",
                ]
            ),
            Relation(
                name="father_of", head_type="person", tail_type="person",
                templates=[
                    "{head} is the father of {tail}.",
                    "{tail}'s father is {head}.",
                    "{tail}'s dad is {head}.",
                ]
            ),
            Relation(
                name="sibling_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is the sibling of {tail}.",
                    "{head} and {tail} are siblings.",
                ]
            ),
            Relation(
                name="brother_of", head_type="person", tail_type="person",
                templates=[
                    "{head} is the brother of {tail}.",
                    "{tail}'s brother is {head}.",
                ]
            ),
            Relation(
                name="sister_of", head_type="person", tail_type="person",
                templates=[
                    "{head} is the sister of {tail}.",
                    "{tail}'s sister is {head}.",
                ]
            ),
            Relation(
                name="grandparent_of", head_type="person", tail_type="person",
                properties={RelationProperty.TRANSITIVE},
                templates=[
                    "{head} is the grandparent of {tail}.",
                    "{tail}'s grandparent is {head}.",
                ]
            ),
            Relation(
                name="uncle_of", head_type="person", tail_type="person",
                templates=[
                    "{head} is the uncle of {tail}.",
                    "{tail}'s uncle is {head}.",
                ]
            ),
            Relation(
                name="aunt_of", head_type="person", tail_type="person",
                templates=[
                    "{head} is the aunt of {tail}.",
                    "{tail}'s aunt is {head}.",
                ]
            ),
            Relation(
                name="cousin_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is the cousin of {tail}.",
                    "{head} and {tail} are cousins.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="kinship",
            category=CATEGORY_SOCIAL,
            relations=relations,
            entity_types={"person": person},
            scallop_rules=[
                "rel grandparent_of(a, c) = parent_of(a, b), parent_of(b, c)",
                "rel sibling_of(a, b) = parent_of(p, a), parent_of(p, b), a != b",
            ]
        )


# ============================================================================
# D2: Marriage & Partnership  
# ============================================================================
@register_domain(CATEGORY_SOCIAL)
class MarriageGenerator(BaseDomainGenerator):
    """Generator for marriage and partnership relations."""
    
    DOMAIN_NAME = "marriage"
    CATEGORY = CATEGORY_SOCIAL
    
    def _default_config(self) -> DomainConfig:
        person = EntityType(name="person", pool=PERSON_NAMES)
        
        relations = [
            Relation(
                name="spouse_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is the spouse of {tail}.",
                    "{head} and {tail} are married.",
                ]
            ),
            Relation(
                name="husband_of", head_type="person", tail_type="person",
                inverse="wife_of",
                templates=[
                    "{head} is the husband of {tail}.",
                    "{tail}'s husband is {head}.",
                ]
            ),
            Relation(
                name="wife_of", head_type="person", tail_type="person",
                inverse="husband_of",
                templates=[
                    "{head} is the wife of {tail}.",
                    "{tail}'s wife is {head}.",
                ]
            ),
            Relation(
                name="engaged_to", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is engaged to {tail}.",
                    "{head} and {tail} are engaged.",
                ]
            ),
            Relation(
                name="divorced_from", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is divorced from {tail}.",
                    "{head} and {tail} got divorced.",
                ]
            ),
            Relation(
                name="domestic_partner_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is the domestic partner of {tail}.",
                    "{head} and {tail} are domestic partners.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="marriage",
            category=CATEGORY_SOCIAL,
            relations=relations,
            entity_types={"person": person},
        )


# ============================================================================
# D4: Friendship
# ============================================================================
@register_domain(CATEGORY_SOCIAL)
class FriendshipGenerator(BaseDomainGenerator):
    """Generator for friendship relations."""
    
    DOMAIN_NAME = "friendship"
    CATEGORY = CATEGORY_SOCIAL
    
    def _default_config(self) -> DomainConfig:
        person = EntityType(name="person", pool=PERSON_NAMES)
        
        relations = [
            Relation(
                name="friend_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is a friend of {tail}.",
                    "{head} and {tail} are friends.",
                ]
            ),
            Relation(
                name="best_friend_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is the best friend of {tail}.",
                    "{head} and {tail} are best friends.",
                ]
            ),
            Relation(
                name="acquaintance_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is an acquaintance of {tail}.",
                    "{head} knows {tail}.",
                ]
            ),
            Relation(
                name="introduced_by", head_type="person", tail_type="person",
                templates=[
                    "{head} was introduced by {tail}.",
                    "{tail} introduced {head} to the group.",
                ]
            ),
            Relation(
                name="enemy_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is an enemy of {tail}.",
                    "{head} and {tail} are enemies.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="friendship",
            category=CATEGORY_SOCIAL,
            relations=relations,
            entity_types={"person": person},
        )


# ============================================================================
# D5: Professional Networks
# ============================================================================
@register_domain(CATEGORY_SOCIAL)
class ProfessionalNetworkGenerator(BaseDomainGenerator):
    """Generator for professional network relations."""
    
    DOMAIN_NAME = "professional_network"
    CATEGORY = CATEGORY_SOCIAL
    
    def _default_config(self) -> DomainConfig:
        person = EntityType(name="person", pool=PERSON_NAMES)
        
        relations = [
            Relation(
                name="colleague_of", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} is a colleague of {tail}.",
                    "{head} and {tail} work together.",
                ]
            ),
            Relation(
                name="mentors", head_type="person", tail_type="person",
                inverse="mentee_of",
                templates=[
                    "{head} mentors {tail}.",
                    "{head} is {tail}'s mentor.",
                ]
            ),
            Relation(
                name="mentee_of", head_type="person", tail_type="person",
                inverse="mentors",
                templates=[
                    "{head} is a mentee of {tail}.",
                    "{tail} is mentoring {head}.",
                ]
            ),
            Relation(
                name="referred_by", head_type="person", tail_type="person",
                templates=[
                    "{head} was referred by {tail}.",
                    "{tail} referred {head} for the job.",
                ]
            ),
            Relation(
                name="collaborated_with", head_type="person", tail_type="person",
                properties={RelationProperty.SYMMETRIC},
                templates=[
                    "{head} collaborated with {tail}.",
                    "{head} and {tail} worked on a project together.",
                ]
            ),
        ]
        
        return DomainConfig(
            name="professional_network",
            category=CATEGORY_SOCIAL,
            relations=relations,
            entity_types={"person": person},
        )

# ============================================================================
# D3: Extended Family
# ============================================================================
@register_domain(CATEGORY_SOCIAL)
class ExtendedFamilyGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "extended_family"
    CATEGORY = CATEGORY_SOCIAL
    
    def _default_config(self) -> DomainConfig:
        person = EntityType(name="person", pool=PERSON_NAMES)
        
        relations = [
            Relation("in_law_of", "person", "person", properties={RelationProperty.SYMMETRIC},
                     templates=["{head} is an in-law of {tail}."]),
            Relation("stepparent_of", "person", "person",
                     templates=["{head} is the stepparent of {tail}."]),
            Relation("stepchild_of", "person", "person",
                     templates=["{head} is the stepchild of {tail}."]),
            Relation("stepsibling_of", "person", "person", properties={RelationProperty.SYMMETRIC},
                     templates=["{head} is a stepsibling of {tail}."]),
            Relation("adopted_by", "person", "person",
                     templates=["{head} was adopted by {tail}."]),
            Relation("godparent_of", "person", "person",
                     templates=["{head} is the godparent of {tail}."]),
        ]
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=relations, entity_types={"person": person})

# ============================================================================
# D6: Groups & Membership
# ============================================================================
@register_domain(CATEGORY_SOCIAL)
class GroupsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "groups"
    CATEGORY = CATEGORY_SOCIAL
    
    def _default_config(self) -> DomainConfig:
        person = EntityType(name="person", pool=PERSON_NAMES)
        group = EntityType(name="group", pool=["Book Club", "Chess Club", "Rotary Club", "Debate Team", "Choir", "Union"])
        
        relations = [
            Relation("member_of", "person", "group",
                     templates=["{head} is a member of {tail}."]),
            Relation("leader_of", "person", "group",
                     templates=["{head} is the leader of {tail}."]),
            Relation("founded_by", "group", "person",
                     templates=["{head} was founded by {tail}."]),
        ]
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=relations, entity_types={"person": person, "group": group})
