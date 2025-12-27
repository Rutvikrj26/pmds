from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_GEOGRAPHIC


# ==========================================
# Domain: Political Geography
# ==========================================
@register_domain(CATEGORY_GEOGRAPHIC)
class PoliticalGeographyGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "political_geography"
    CATEGORY = CATEGORY_GEOGRAPHIC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("capital_of", "City", "Country",
                         templates=["{head} is the capital of {tail}."]),
                Relation("borders", "Country", "Country", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} borders {tail}."]),
                Relation("located_in", "City", "Country",
                         templates=["{head} is located in {tail}."]),
                Relation("part_of_continent", "Country", "Continent",
                         templates=["{head} is part of {tail}."]),
                Relation("member_of", "Country", "Alliance",
                         templates=["{head} is a member of {tail}."]),
            ],
            entity_types={
                "City": EntityType("City", ["Paris", "London", "Tokyo", "Berlin"]),
                "Country": EntityType("Country", ["France", "UK", "Japan", "Germany"]),
                "Continent": EntityType("Cont", ["Europe", "Asia", "Africa"]),
                "Alliance": EntityType("Alliance", ["NATO", "EU", "UN"]),
            }
        )

# ==========================================
# Domain: Ecology
# ==========================================
@register_domain(CATEGORY_GEOGRAPHIC)
class EcologyGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "ecology"
    CATEGORY = CATEGORY_GEOGRAPHIC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("eats", "Animal", "Animal",
                         templates=["{head} eats {tail}."]),
                Relation("pollinates", "Insect", "Plant",
                         templates=["{head} pollinates {tail}."]),
                Relation("habitat_of", "Biome", "Animal",
                         templates=["{head} is the habitat of {tail}."]),
                Relation("competes_with", "Species", "Species", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} competes with {tail}."]),
                Relation("symbiotic_with", "Species", "Species", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} is symbiotic with {tail}."]),
            ],
            entity_types={
                "Animal": EntityType("Animal", ["Lion", "Zebra", "Hawk"]),
                "Insect": EntityType("Insect", ["Bee", "Butterfly"]),
                "Plant": EntityType("Plant", ["Rose", "Grass", "Oak"]),
                "Biome": EntityType("Biome", ["Savanna", "Forest", "Desert"]),
                "Species": EntityType("Spec", ["Wolf", "Bear"]),
            }
        )

# ==========================================
# Domain: Hydrology
# ==========================================
@register_domain(CATEGORY_GEOGRAPHIC)
class HydrologyGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "hydrology"
    CATEGORY = CATEGORY_GEOGRAPHIC
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("flows_into", "River", "BodyOfWater",
                         templates=["{head} flows into {tail}."]),
                Relation("tributary_of", "River", "River",
                         templates=["{head} is a tributary of {tail}."]),
                Relation("source_of", "Location", "River",
                         templates=["{head} is the source of {tail}."]),
                Relation("banks_of", "City", "River",
                         templates=["{head} lies on the banks of {tail}."]),
                Relation("contains_water", "Aquifer", "Region",
                         templates=["{head} provides water to {tail}."]),
            ],
            entity_types={
                "River": EntityType("River", ["Nile", "Amazon", "Mississippi"]),
                "BodyOfWater": EntityType("Water", ["Mediterranean", "Atlantic", "Pacific"]),
                "Location": EntityType("Loc", ["Mountains", "Lake Victoria"]),
                "City": EntityType("City", ["Cairo", "New Orleans"]),
                "Aquifer": EntityType("Aquifer", ["Ogallala"]),
                "Region": EntityType("Region", ["Midwest"]),
            }
        )

# ============================================================================
# D62: Climate
# ============================================================================
@register_domain(CATEGORY_GEOGRAPHIC)
class ClimateGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "climate"
    CATEGORY = CATEGORY_GEOGRAPHIC
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("characterized_by", "ClimateZone", "WeatherPattern", templates=["{head} is characterized by {tail}."]),
            Relation("caused_by", "Phenomenon", "Factor", templates=["{head} is caused by {tail}."]),
            Relation("affects_yield_of", "Weather", "Crop", templates=["{head} affects yield of {tail}."]),
            Relation("occurs_in", "Phenomenon", "Region", templates=["{head} occurs in {tail}."]),
            Relation("measured_by", "Factor", "Instrument", templates=["{head} measured by {tail}."]),
        ], entity_types={"ClimateZone": EntityType("Zone", ["Tropical"]), "WeatherPattern": EntityType("Pat", ["Monsoon"]), "Phenomenon": EntityType("Phen", ["Drought"]), "Factor": EntityType("Fac", ["El Nino"]), "Weather": EntityType("Wthr", ["Rain"]), "Crop": EntityType("Crop", ["Wheat"]), "Region": EntityType("Reg", ["Asia"]), "Instrument": EntityType("Inst", ["Barometer"])})
