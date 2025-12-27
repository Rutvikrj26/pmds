from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_GAMING


# ==========================================
# Domain: Game Mechanics
# ==========================================
@register_domain(CATEGORY_GAMING)
class GameMechanicsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "game_mechanics"
    CATEGORY = CATEGORY_GAMING
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("dropped_by", "Item", "Monster",
                         templates=["{head} is dropped by {tail}."]),
                Relation("requires_skill", "Item", "Skill",
                         templates=["Equipping {head} requires {tail}."]),
                Relation("unlocks", "Level", "Ability",
                         templates=["Reaching {head} unlocks {tail}."]),
                Relation("crafted_from", "Item", "Material",
                         templates=["{head} is crafted from {tail}."]),
                Relation("buffs_stat", "Potion", "Stat",
                         templates=["{head} buffs {tail}."]),
            ],
            entity_types={
                "Item": EntityType("Item", ["Sword", "Shield", "Potion"]),
                "Monster": EntityType("Mob", ["Goblin", "Dragon", "Slime"]),
                "Skill": EntityType("Skill", ["Swordsmanship", "Magic"]),
                "Level": EntityType("Lvl", ["Level 10", "Level 50"]),
                "Ability": EntityType("Ability", ["Fireball", "Heal"]),
                "Material": EntityType("Mat", ["Iron", "Wood", "Gold"]),
                "Potion": EntityType("Pot", ["Health Pot", "Mana Pot"]),
                "Stat": EntityType("Stat", ["Strength", "Intelligence"]),
            }
        )

# ==========================================
# Domain: Multiplayer Social
# ==========================================
@register_domain(CATEGORY_GAMING)
class MultiplayerSocialGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "multiplayer_social"
    CATEGORY = CATEGORY_GAMING
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("guild_member", "Player", "Guild",
                         templates=["{head} is a member of {tail}."]),
                Relation("party_leader", "Player", "Party",
                         templates=["{head} leads the {tail}."]),
                Relation("friend_of", "Player", "Player", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} is friends with {tail}."]),
                Relation("blocked", "Player", "Player",
                         templates=["{head} blocked {tail}."]),
                Relation("invited", "Player", "Player",
                         templates=["{head} invited {tail}."]),
            ],
            entity_types={
                "Player": EntityType("Player", ["NoobMaster69", "TheLegend27", "Leeroy"]),
                "Guild": EntityType("Guild", ["Method", "Limit"]),
                "Party": EntityType("Party", ["RaidGroupA", "DungeonTeam"]),
            }
        )

# ==========================================
# Domain: Narrative Quests
# ==========================================
@register_domain(CATEGORY_GAMING)
class NarrativeQuestsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "narrative_quests"
    CATEGORY = CATEGORY_GAMING
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("starts_quest", "NPC", "Quest",
                         templates=["{head} starts the quest {tail}."]),
                Relation("quest_objective", "Quest", "Location",
                         templates=["{head} requires going to {tail}."]),
                Relation("completes_quest", "Player", "Quest",
                         templates=["{head} completed {tail}."]),
                Relation("rewards", "Quest", "Item",
                         templates=["{head} rewards {tail}."]),
                Relation("prerequisite_for", "Quest", "Quest", properties={RelationProperty.TRANSITIVE},
                         templates=["{head} is a prerequisite for {tail}."]),
            ],
            entity_types={
                "NPC": EntityType("NPC", ["King", "Merchant", "Guard"]),
                "Quest": EntityType("Quest", ["SavePrincess", "SlayDragon"]),
                "Location": EntityType("Loc", ["Castle", "Cave", "Forest"]),
                "Player": EntityType("Player", ["Hero", "Adventurer"]),
                "Item": EntityType("Item", ["Gold", "Sword"]),
            }
        )

# ============================================================================
# D57: Game Development
# ============================================================================
@register_domain(CATEGORY_GAMING)
class GameDevGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "game_dev"
    CATEGORY = CATEGORY_GAMING
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("developed_by", "Game", "Studio", templates=["{head} was developed by {tail}."]),
            Relation("published_by", "Game", "Publisher", templates=["{head} was published by {tail}."]),
            Relation("uses_engine", "Game", "Engine", templates=["{head} uses {tail} engine."]),
            Relation("released_on", "Game", "Date", templates=["{head} was released on {tail}."]),
            Relation("genre_is", "Game", "Genre", templates=["{head} is a {tail} game."]),
        ], entity_types={"Game": EntityType("Game", ["Half-Life 3"]), "Studio": EntityType("Studio", ["Valve"]), "Publisher": EntityType("Pub", ["EA"]), "Engine": EntityType("Eng", ["Unity"]), "Date": EntityType("Date", ["2024"]), "Genre": EntityType("Gen", ["FPS"])})

# ============================================================================
# D58: Esports
# ============================================================================
@register_domain(CATEGORY_GAMING)
class EsportsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "esports"
    CATEGORY = CATEGORY_GAMING
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("competes_in", "Team", "Tournament", templates=["{head} competes in {tail}."]),
            Relation("won_tournament", "Team", "Tournament", templates=["{head} won {tail}."]),
            Relation("sponsored_by", "Team", "Sponsor", templates=["{head} is sponsored by {tail}."]),
            Relation("streamed_on", "Tournament", "Platform", templates=["{head} is streamed on {tail}."]),
            Relation("mvp_of", "Player", "Tournament", templates=["{head} was MVP of {tail}."]),
        ], entity_types={"Team": EntityType("Team", ["T1"]), "Tournament": EntityType("Tourn", ["Worlds"]), "Sponsor": EntityType("Spon", ["Intel"]), "Platform": EntityType("Plat", ["Twitch"]), "Player": EntityType("Plyr", ["Faker"])})
