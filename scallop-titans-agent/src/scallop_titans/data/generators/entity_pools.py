"""
Entity pools for multi-domain generation.

Provides realistic entity names/IDs for each entity type across domains.
"""
from __future__ import annotations

# --- Person Names ---
PERSON_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", 
    "Ivan", "Judy", "Kevin", "Linda", "Mike", "Nancy", "Oscar", "Peggy",
    "Quentin", "Ruth", "Steve", "Trudy", "Ursula", "Victor", "Wendy", "Xavier",
    "Yvonne", "Zach", "Aaron", "Bella", "Carl", "Diana", "Edward", "Fiona",
    "George", "Helen", "Ian", "Julia", "Kyle", "Laura", "Mark", "Nora",
    "Oliver", "Paula", "Quinn", "Rachel", "Sam", "Tina", "Uma", "Vince",
    "Wanda", "Xander", "Yolanda", "Zane", "Adam", "Betty", "Chris", "Debbie",
    "Eric", "Felicia", "Gary", "Hannah", "Isaac", "Jenny", "Ken", "Lisa",
    "Matt", "Nicole", "Otto", "Pam", "Randy", "Sarah", "Tom", "Uma", 
    "Vince", "Wanda", "Xander", "Yolanda", "Zack", "Andrew", "Barbara", "Cody",
]

# --- Organization Names ---
ORG_NAMES = [
    "Acme Corp", "Globex", "Initech", "Umbrella Inc", "Stark Industries",
    "Wayne Enterprises", "Oscorp", "LexCorp", "Cyberdyne", "Weyland-Yutani",
    "Tyrell Corp", "Aperture Science", "Black Mesa", "Massive Dynamic",
    "Hooli", "Pied Piper", "Raviga", "Bachmanity", "Endframe", "Nucleus",
]

# --- Role/Title Names ---
ROLE_NAMES = [
    "Admin", "Editor", "Viewer", "Manager", "Developer", "Analyst", "Auditor",
    "Superuser", "Guest", "Moderator", "Owner", "Contributor", "Reviewer",
]

# --- Resource/File Names ---
RESOURCE_NAMES = [
    "ProjectPlan.docx", "FinancialReport.xlsx", "CustomerData.csv",
    "SourceCode.zip", "Credentials.key", "Backup.tar.gz", "Config.yaml",
    "Database.sql", "Logs.txt", "Secrets.env", "API_Keys.json",
]

# --- Server/Network Names ---
SERVER_NAMES = [
    "web-prod-1", "web-prod-2", "api-gateway", "db-master", "db-replica-1",
    "cache-redis", "worker-1", "worker-2", "lb-main", "monitor-1",
    "auth-service", "payment-api", "search-elastic", "queue-rabbitmq",
]

# --- Location Names ---
LOCATION_NAMES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "London", "Paris", "Tokyo", "Shanghai", "Mumbai", "Sydney", "Berlin",
]

# --- Warehouse/Zone Names ---
WAREHOUSE_NAMES = [
    "WH-East", "WH-West", "WH-Central", "WH-North", "WH-South",
    "Zone-A", "Zone-B", "Zone-C", "Zone-D", "Zone-E",
    "Dock-1", "Dock-2", "Dock-3", "Staging-1", "Staging-2",
]

# --- Product/Item Names ---
PRODUCT_NAMES = [
    "Widget-A", "Widget-B", "Gadget-X", "Gadget-Y", "Component-1",
    "Component-2", "Part-Alpha", "Part-Beta", "Module-M1", "Module-M2",
]

# --- Drug/Medicine Names ---
DRUG_NAMES = [
    "Aspirin", "Ibuprofen", "Acetaminophen", "Metformin", "Lisinopril",
    "Amlodipine", "Metoprolol", "Omeprazole", "Simvastatin", "Losartan",
    "Gabapentin", "Hydrochlorothiazide", "Sertraline", "Azithromycin",
]

# --- Medical Condition Names ---
CONDITION_NAMES = [
    "Hypertension", "Diabetes", "Arthritis", "Asthma", "Migraine",
    "Influenza", "Pneumonia", "Bronchitis", "Gastritis", "Dermatitis",
]

# --- Department Names ---
DEPARTMENT_NAMES = [
    "Engineering", "Sales", "Marketing", "Finance", "HR", "Operations",
    "Legal", "IT", "R&D", "Customer Support", "Product", "Design",
]

# --- Project Names ---
PROJECT_NAMES = [
    "Project-Alpha", "Project-Beta", "Project-Gamma", "Project-Delta",
    "Initiative-X", "Initiative-Y", "Phase-1", "Phase-2", "Sprint-A", "Sprint-B",
]

# --- Country Names ---
COUNTRY_NAMES = [
    "USA", "Canada", "Mexico", "UK", "France", "Germany", "Italy", "Spain",
    "Japan", "China", "India", "Australia", "Brazil", "Argentina", "Russia",
]

# --- Species Names ---
SPECIES_NAMES = [
    "Lion", "Tiger", "Bear", "Wolf", "Fox", "Eagle", "Hawk", "Salmon",
    "Deer", "Rabbit", "Snake", "Frog", "Butterfly", "Bee", "Ant",
]

# --- Game Item Names ---
GAME_ITEM_NAMES = [
    "Sword of Flames", "Shield of Light", "Potion of Healing", "Scroll of Wisdom",
    "Ring of Power", "Amulet of Protection", "Staff of Magic", "Bow of Precision",
    "Armor of Steel", "Boots of Speed", "Gloves of Dexterity", "Helm of Insight",
]

# --- Music/Artist Names ---
ARTIST_NAMES = [
    "The Echoes", "Midnight Sun", "Crystal Wave", "Electric Storm",
    "Velvet Thunder", "Neon Dreams", "Shadow Pulse", "Arctic Flames",
]

# --- Course Names ---
COURSE_NAMES = [
    "CS-101", "CS-201", "CS-301", "MATH-101", "MATH-201", "PHYS-101",
    "CHEM-101", "ENG-101", "HIST-101", "ECON-101", "BIO-101", "PSYCH-101",
]

# --- Team Names ---
TEAM_NAMES = [
    "Eagles", "Tigers", "Lions", "Bears", "Wolves", "Hawks", "Panthers",
    "Sharks", "Dragons", "Phoenix", "Giants", "Knights", "Warriors", "Titans",
]

# --- Building/Structure Names ---
BUILDING_NAMES = [
    "Tower-A", "Tower-B", "Building-1", "Building-2", "Annex-East", "Annex-West",
    "Main-Hall", "Conference-Center", "Lab-1", "Lab-2", "Warehouse-Main",
]
