from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_MISC


# ==========================================
# Domain: Food & Culinary
# ==========================================
@register_domain(CATEGORY_MISC)
class FoodGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "food_culinary"
    CATEGORY = CATEGORY_MISC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("ingredient_of", "Ingredient", "Dish",
                         templates=["{head} is an ingredient of {tail}."]),
                Relation("cuisine_of", "Dish", "Region",
                         templates=["{head} is a {tail} dish."]),
                Relation("pairs_with", "Wine", "Dish",
                         templates=["{head} pairs with {tail}."]),
                Relation("sourced_from", "Ingredient", "Farm",
                         templates=["{head} is sourced from {tail}."]),
                Relation("allergic_to_food", "Person", "Ingredient",
                         templates=["{head} is allergic to {tail}."]),
            ],
            entity_types={
                "Ingredient": EntityType("Ingdum", ["Tomato", "Basil", "Garlic"]),
                "Dish": EntityType("Dish", ["Pizza", "Pasta", "Soup"]),
                "Region": EntityType("Reg", ["Italian", "French", "Chinese"]),
                "Wine": EntityType("Wine", ["Merlot", "Chardonnay"]),
                "Farm": EntityType("Farm", ["LocalFarm", "FactoryFarm"]),
                "Person": EntityType("Person", ["Foodie", "Chef"]),
            }
        )

# ==========================================
# Domain: Sports
# ==========================================
@register_domain(CATEGORY_MISC)
class SportsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "sports"
    CATEGORY = CATEGORY_MISC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("plays_for", "Player", "Team",
                         templates=["{head} plays for {tail}."]),
                Relation("coached_by", "Team", "Coach",
                         templates=["{head} is coached by {tail}."]),
                Relation("defeated", "Team", "Team",
                         templates=["{head} defeated {tail}."]),
                Relation("home_stadium_of", "Stadium", "Team",
                         templates=["{head} is the home stadium of {tail}."]),
                Relation("captain_of", "Player", "Team",
                         templates=["{head} is the captain of {tail}."]),
            ],
            entity_types={
                "Player": EntityType("Player", ["Messi", "Ronaldo", "LeBron"]),
                "Team": EntityType("Team", ["Barca", "RealMadrid", "Lakers"]),
                "Coach": EntityType("Coach", ["Pep", "Zidane"]),
                "Stadium": EntityType("Stad", ["Camp Nou", "Bernabeu"]),
            }
        )

# ==========================================
# Domain: Military
# ==========================================
@register_domain(CATEGORY_MISC)
class MilitaryGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "military"
    CATEGORY = CATEGORY_MISC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("commands", "Officer", "Unit",
                         templates=["{head} commands {tail}."]),
                Relation("stationed_at", "Unit", "Base",
                         templates=["{head} is stationed at {tail}."]),
                Relation("allied_with_force", "Army", "Army", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} is allied with {tail}."]),
                Relation("deployed_to", "Unit", "Region",
                         templates=["{head} is deployed to {tail}."]),
                Relation("reports_to_officer", "Soldier", "Officer",
                         templates=["{head} reports to {tail}."]),
            ],
            entity_types={
                "Officer": EntityType("Officer", ["General X", "Captain Y"]),
                "Unit": EntityType("Unit", ["1st Division", "101st Airborne"]),
                "Base": EntityType("Base", ["Fort Knox", "Ramstein"]),
                "Army": EntityType("Army", ["Ally Forces", "Coalition"]),
                "Region": EntityType("Region", ["Frontier", "Border"]),
                "Soldier": EntityType("Soldier", ["Private Ryan", "Sgt. Pepper"]),
            }
        )

# ============================================================================
# D66: Construction
# ============================================================================
@register_domain(CATEGORY_MISC)
class ConstructionGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "construction"
    CATEGORY = CATEGORY_MISC
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("built_by", "Structure", "Contractor", templates=["{head} was built by {tail}."]),
            Relation("designed_by", "Structure", "Architect", templates=["{head} was designed by {tail}."]),
            Relation("located_at_site", "Structure", "Site", templates=["{head} is located at {tail}."]),
            Relation("requires_permit_from", "Project", "City", templates=["{head} requires permit from {tail}."]),
            Relation("material_supplied_by", "Project", "Supplier", templates=["{head} material supplied by {tail}."]),
        ], entity_types={"Structure": EntityType("Struct", ["Tower"]), "Contractor": EntityType("Cont", ["BuildCo"]), "Architect": EntityType("Arch", ["Ted"]), "Site": EntityType("Site", ["Lot 1"]), "Project": EntityType("Proj", ["Renovation"]), "City": EntityType("City", ["NYC"]), "Supplier": EntityType("Supp", ["LumberYard"])})
