from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_ACADEMIC


# ==========================================
# Domain: Prerequisites
# ==========================================
@register_domain(CATEGORY_ACADEMIC)
class PrerequisitesGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "prerequisites"
    CATEGORY = CATEGORY_ACADEMIC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("course_requires", "Course", "Course", properties={RelationProperty.TRANSITIVE},
                         templates=["{head} requires {tail}.", "{tail} is a prerequisite for {head}."]),
                Relation("corequisite_of", "Course", "Course", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} is a corequisite of {tail}."]),
                Relation("satisfies_req", "Course", "Requirement",
                         templates=["{head} satisfies {tail}."]),
                Relation("equivalent_to", "Course", "Course", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} is equivalent to {tail}."]),
                Relation("waived_by", "Requirement", "Exam",
                         templates=["{tail} waives {head}."]),
            ],
            entity_types={
                "Course": EntityType("Course", ["CS101", "CS102", "MATH200", "PHYS100"]),
                "Requirement": EntityType("Req", ["QuantitativeReasoning", "Writing", "MajorCore"]),
                "Exam": EntityType("Exam", ["AP-Calc", "PlacementTest"]),
            }
        )

# ==========================================
# Domain: Research Citations
# ==========================================
@register_domain(CATEGORY_ACADEMIC)
class ResearchCitationsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "research_citations"
    CATEGORY = CATEGORY_ACADEMIC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("cites", "Paper", "Paper",
                         templates=["{head} cites {tail}.", "{tail} is cited by {head}."]),
                Relation("extends", "Paper", "Paper",
                         templates=["{head} extends {tail}."]),
                Relation("refutes", "Paper", "Paper",
                         templates=["{head} refutes {tail}."]),
                Relation("authored_by", "Paper", "Author",
                         templates=["{head} was authored by {tail}."]),
                Relation("published_in", "Paper", "Venue",
                         templates=["{head} was published in {tail}."]),
            ],
            entity_types={
                "Paper": EntityType("Paper", ["AttentionIsAllYouNeed", "ResNet", "BERT", "GPT-3"]),
                "Author": EntityType("Author", ["Vaswani", "He", "Devlin", "Brown"]),
                "Venue": EntityType("Venue", ["NeurIPS", "ICML", "CVPR", "Nature"]),
            }
        )

# ==========================================
# Domain: Academic Hierarchy
# ==========================================
@register_domain(CATEGORY_ACADEMIC)
class AcademicHierarchyGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "academic_hierarchy"
    CATEGORY = CATEGORY_ACADEMIC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("advised_by", "Student", "Professor",
                         templates=["{head} is advised by {tail}.", "{tail} advises {head}."]),
                Relation("committee_member", "Professor", "Student",
                         templates=["{head} is on {tail}'s committee."]),
                Relation("lab_member_of", "Student", "Lab",
                         templates=["{head} is a member of {tail}."]),
                Relation("pi_of", "Professor", "Lab",
                         templates=["{head} is the PI of {tail}."]),
                Relation("teaches", "Professor", "Course",
                         templates=["{head} teaches {tail}."]),
            ],
            entity_types={
                "Student": EntityType("Student", ["Alice", "Bob", "Charlie"]),
                "Professor": EntityType("Prof", ["Dr. Turing", "Dr. Hopper", "Dr. Shannon"]),
                "Lab": EntityType("Lab", ["AI-Lab", "Robo-Lab", "Theory-Group"]),
                "Course": EntityType("Course", ["CS229", "CS224N"]),
            }
        )

# ============================================================================
# D47: Degrees & Certifications
# ============================================================================
@register_domain(CATEGORY_ACADEMIC)
class DegreeGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "degrees"
    CATEGORY = CATEGORY_ACADEMIC
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("requires_credits", "Degree", "Credits", templates=["{head} requires {tail} credits."]),
            Relation("awarded_by", "Degree", "Institution", templates=["{head} is awarded by {tail}."]),
            Relation("earned_by", "Degree", "Student", templates=["{tail} earned {head}."]),
            Relation("program_includes", "Degree", "Course", templates=["{head} includes {tail}."]),
        ], entity_types={"Degree": EntityType("Deg", ["BS CS"]), "Credits": EntityType("Cred", ["120"]), "Institution": EntityType("Uni", ["MIT"]), "Student": EntityType("Stu", ["Alice"]), "Course": EntityType("Crs", ["CS101"])})

# ============================================================================
# D48: Institutions (Funding & Grants)
# ============================================================================
@register_domain(CATEGORY_ACADEMIC)
class InstitutionGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "institutions"
    CATEGORY = CATEGORY_ACADEMIC
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("affiliated_with", "Lab", "Institution", templates=["{head} is affiliated with {tail}."]),
            Relation("grants_funding_to", "Agency", "Lab", templates=["{head} grants funding to {tail}."]),
            Relation("employs", "Institution", "Professor", templates=["{head} employs {tail}."]),
            Relation("accredited_by", "Institution", "Body", templates=["{head} accredited by {tail}."]),
        ], entity_types={"Lab": EntityType("Lab", ["AI Lab"]), "Institution": EntityType("Uni", ["Stanford"]), "Agency": EntityType("Ag", ["NSF"]), "Professor": EntityType("Prof", ["Dr. X"]), "Body": EntityType("Body", ["ABET"])})
