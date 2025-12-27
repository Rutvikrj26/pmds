from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_LEGAL


# ==========================================
# Domain: Contract Clauses
# ==========================================
@register_domain(CATEGORY_LEGAL)
class ContractClausesGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "contract_clauses"
    CATEGORY = CATEGORY_LEGAL
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("binds", "Contract", "Party",
                         templates=["{head} binds {tail}.", "{tail} is bound by {head}."]),
                Relation("requires", "Clause", "Action",
                         templates=["{head} requires {tail}.", "{tail} is required by {head}."]),
                Relation("prohibits", "Clause", "Action",
                         templates=["{head} prohibits {tail}.", "{tail} is forbidden by {head}."]),
                Relation("supersedes", "Contract", "Contract",
                         templates=["{head} supersedes {tail}.", "{tail} is overridden by {head}."]),
                Relation("contains_clause", "Contract", "Clause",
                         templates=["{head} contains clause {tail}."]),
                Relation("breached_by", "Action", "Party",
                         templates=["{head} was breached by {tail}."]),
                Relation("triggers_penalty", "Action", "Penalty",
                         templates=["{head} triggers penalty {tail}."]),
            ],
            entity_types={
                "Contract": EntityType("Contract", ["MSA-2024", "NDA-X", "SLA-Basic", "Employment-Agmt"]),
                "Party": EntityType("Party", ["Acme Corp", "Beta LLC", "John Doe", "Jane Smith"]),
                "Clause": EntityType("Clause", ["Non-Compete", "Confidentiality", "Payment Terms", "Termination"]),
                "Action": EntityType("Action", ["Disclosure", "Late Payment", "Solicitation", "Disparagement"]),
                "Penalty": EntityType("Penalty", ["Fine $10k", "Termination", "Injunction"]),
            }
        )

# ==========================================
# Domain: Compliance Rules
# ==========================================
@register_domain(CATEGORY_LEGAL)
class ComplianceRulesGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "compliance_rules"
    CATEGORY = CATEGORY_LEGAL
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("governs", "Regulation", "Activity",
                         templates=["{head} governs {tail}.", "{tail} is regulated by {head}."]),
                Relation("compliant_with", "Process", "Regulation",
                         templates=["{head} is compliant with {tail}.", "{head} meets standards of {tail}."]),
                Relation("violates", "Action", "Regulation",
                         templates=["{head} violates {tail}.", "{tail} is breached by {head}."]),
                Relation("applies_to", "Regulation", "Region",
                         templates=["{head} applies to {tail}.", "{head} is effective in {tail}."]),
                Relation("audit_requires", "Regulation", "Document",
                         templates=["{head} audit requires {tail}."]),
                Relation("entity_in_region", "Entity", "Region",
                         templates=["{head} operates in {tail}."]),
            ],
            entity_types={
                "Regulation": EntityType("Reg", ["GDPR", "CCPA", "HIPAA", "SOX", "PCI-DSS"]),
                "Activity": EntityType("Activity", ["Data Processing", "Financial Reporting", "Patient Care"]),
                "Process": EntityType("Process", ["Encryption", "2FA", "Audit Logging"]),
                "Action": EntityType("Action", ["Data Leak", "Insider Trading", "Unsecured Storage"]),
                "Region": EntityType("Region", ["EU", "California", "USA", "Global"]),
                "Document": EntityType("Doc", ["Access Logs", "Financial Statements", "Consent Forms"]),
                "Entity": EntityType("Entity", ["Server-1", "Database-A", "Employee-X"]),
            }
        )

# ==========================================
# Domain: Case Precedents
# ==========================================
@register_domain(CATEGORY_LEGAL)
class CasePrecedentsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "case_precedents"
    CATEGORY = CATEGORY_LEGAL
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("cited_by", "Case", "Case",
                         templates=["{head} is cited by {tail}.", "{tail} cites {head}."]),
                Relation("overturned", "Case", "Case",
                         templates=["{head} overturned {tail}.", "{tail} was overturned by {head}."]),
                Relation("established_principle", "Case", "LegalPrinciple",
                         templates=["{head} established {tail}."]),
                Relation("presided_by", "Judge", "Case",
                         templates=["{head} presided over {tail}."]),
                Relation("represented", "Lawyer", "Client",
                         templates=["{head} represented {tail}."]),
                Relation("ruled_in_favor_of", "Case", "Party",
                         templates=["In {head}, court ruled in favor of {tail}."]),
            ],
            entity_types={
                "Case": EntityType("Case", ["Doe v. Roe", "State v. Smith", "Corp v. Corp", "In re Tech"]),
                "LegalPrinciple": EntityType("Principle", ["Fair Use", "Fruit of Poisonous Tree", "Miranda Rights"]),
                "Judge": EntityType("Judge", ["Judge Dredd", "Justic Warren", "Judge Judy"]),
                "Lawyer": EntityType("Lawyer", ["Saul Goodman", "Atticus Finch", "Daredevil"]),
                "Client": EntityType("Client", ["Plaintiff", "Defendant", "Apellant"]),
                "Party": EntityType("Party", ["Winner", "Loser"]),
            }
        )

# ============================================================================
# D24: Litigation
# ============================================================================
@register_domain(CATEGORY_LEGAL)
class LitigationGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "litigation"
    CATEGORY = CATEGORY_LEGAL
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("filed_suit_against", "Plaintiff", "Defendant", templates=["{head} filed suit against {tail}."]),
            Relation("represented_by", "Party", "Lawyer", templates=["{head} is represented by {tail}."]),
            Relation("witness_in", "Witness", "Case", templates=["{head} is a witness in {tail}."]),
            Relation("deposed_by", "Witness", "Lawyer", templates=["{head} was deposed by {tail}."]),
            Relation("subpoenaed_by", "Witness", "Lawyer", templates=["{head} was subpoenaed by {tail}."]),
        ], entity_types={"Plaintiff": EntityType("P", ["Company A"]), "Defendant": EntityType("D", ["Company B"]), "Party": EntityType("Party", ["Party A"]), "Lawyer": EntityType("L", ["Lawyer X"]), "Witness": EntityType("W", ["John"]), "Case": EntityType("C", ["Case 1"])})

# ============================================================================
# D25: IP
# ============================================================================
@register_domain(CATEGORY_LEGAL)
class IPGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "intellectual_property"
    CATEGORY = CATEGORY_LEGAL
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("patented_by", "Invention", "Inventor", templates=["{head} was patented by {tail}."]),
            Relation("licensed_to", "IP", "Company", templates=["{head} is licensed to {tail}."]),
            Relation("infringes_on", "Product", "IP", templates=["{head} infringes on {tail}."]),
            Relation("trademarked_by", "Brand", "Company", templates=["{head} is trademarked by {tail}."]),
            Relation("copyrighted_by", "Work", "Creator", templates=["{head} is copyrighted by {tail}."]),
        ], entity_types={"Invention": EntityType("Inv", ["Widget"]), "Inventor": EntityType("Person", ["Edison"]), "IP": EntityType("IP", ["Patent 123"]), "Company": EntityType("Co", ["Acme"]), "Product": EntityType("Prod", ["X1"]), "Brand": EntityType("Brand", ["Logo"]), "Work": EntityType("Work", ["Book"]), "Creator": EntityType("Creator", ["Author"])})

# ============================================================================
# D26: Privacy
# ============================================================================
@register_domain(CATEGORY_LEGAL)
class PrivacyGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "privacy_data"
    CATEGORY = CATEGORY_LEGAL
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("data_controller_of", "Entity", "Data", templates=["{head} is the data controller of {tail}."]),
            Relation("data_processor_for", "Processor", "Controller", templates=["{head} processes data for {tail}."]),
            Relation("consent_given_by", "Subject", "Entity", templates=["{head} gave consent to {tail}."]),
            Relation("breach_notified_to", "Subject", "Authority", templates=["{head} breach notified to {tail}."]),
            Relation("right_to_erasure_requested_by", "Subject", "Entity", templates=["{head} requested erasure from {tail}."]),
        ], entity_types={"Entity": EntityType("Ent", ["Corp"]), "Data": EntityType("Data", ["UserDb"]), "Processor": EntityType("Proc", ["CloudProvider"]), "Controller": EntityType("Cont", ["AppOwner"]), "Subject": EntityType("Sub", ["User"]), "Authority": EntityType("Auth", ["DPA"])})
