from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_HEALTHCARE


# ==========================================
# Domain: Medical Records (Access & Privacy)
# ==========================================
@register_domain(CATEGORY_HEALTHCARE)
class MedicalRecordsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "medical_records"
    CATEGORY = CATEGORY_HEALTHCARE
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("treating_physician", "Doctor", "Patient", 
                         templates=["{head} is the treating physician for {tail}.", "{tail} is being treated by {head}."]),
                Relation("has_access_to", "Doctor", "MedicalRecord",
                         templates=["{head} has access to {tail}.", "{tail} can be accessed by {head}."]),
                Relation("belongs_to", "MedicalRecord", "Patient",
                         templates=["{tail}'s record is {head}.", "{head} belongs to {tail}."]),
                Relation("authorized_viewer", "Person", "MedicalRecord",
                         templates=["{head} is an authorized viewer of {tail}.", "{tail} viewable by {head}."]),
                Relation("guardian_of", "Person", "Patient",
                         templates=["{head} is the guardian of {tail}.", "{tail} is the ward of {head}."]),
                Relation("nurse_for", "Nurse", "Patient",
                         templates=["{head} is the nurse for {tail}.", "{tail} is cared for by nurse {head}."]),
                Relation("department_access", "Department", "MedicalRecord",
                         templates=["{head} has department-level access to {tail}."]),
                Relation("works_in", "Doctor", "Department",
                         templates=["{head} works in {tail}.", "{head} is staff in {tail}."]),
            ],
            entity_types={
                "Doctor": EntityType("Doctor", ["Dr. Smith", "Dr. Jones", "Dr. House", "Dr. Strange", "Dr. Who"]),
                "Nurse": EntityType("Nurse", ["Nurse Betty", "Nurse Jackie", "Nurse Joy"]),
                "Patient": EntityType("Patient", ["John Doe", "Jane Doe", "Bob", "Alice", "Charlie"]),
                "Person": EntityType("Person", ["Alice", "Bob", "Charlie", "Dave", "Eve"]),
                "MedicalRecord": EntityType("Record", prefix="Rec-"),
                "Department": EntityType("Department", ["Oncology", "Pediatrics", "ER", "Cardiology"]),
            }
        )

# ==========================================
# Domain: Symptom Triage (Diagnosis)
# ==========================================
@register_domain(CATEGORY_HEALTHCARE)
class SymptomTriageGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "symptom_triage"
    CATEGORY = CATEGORY_HEALTHCARE
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("has_symptom", "Patient", "Symptom",
                         templates=["{head} has {tail}.", "{head} is experiencing {tail}."]),
                Relation("indicates", "Symptom", "Condition",
                         templates=["{head} indicates potential {tail}.", "{tail} is a sign of {head}."]),
                Relation("contraindicates", "Condition", "Treatment",
                         templates=["{head} contraindicates {tail}.", "{tail} should not be used for {head}."]),
                Relation("requires_urgent_care", "Symptom", "UrgencyLevel",
                         templates=["{head} requires {tail} attention."]),
                Relation("precursor_to", "Condition", "Condition",
                         templates=["{head} is a precursor to {tail}."]),
                Relation("treated_with", "Condition", "Treatment",
                         templates=["{head} is treated with {tail}.", "{tail} prescribes for {head}."]),
            ],
            entity_types={
                "Patient": EntityType("Patient", ["Patient A", "Patient B", "Patient C"]),
                "Symptom": EntityType("Symptom", ["Fever", "Cough", "Chest Pain", "Rash", "Nausea", "Dizziness"]),
                "Condition": EntityType("Condition", ["Flu", "Pneumonia", "Heart Attack", "Allergy", "Concussion"]),
                "Treatment": EntityType("Treatment", ["Antibiotics", "Rest", "Surgery", "Antihistamines"]),
                "UrgencyLevel": EntityType("Urgency", ["Immediate", "Urgent", "Routine"]),
            }
        )

# ==========================================
# Domain: Drug Interactions
# ==========================================
@register_domain(CATEGORY_HEALTHCARE)
class DrugInteractionsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "drug_interactions"
    CATEGORY = CATEGORY_HEALTHCARE
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("prescribed", "Patient", "Drug",
                         templates=["{head} is prescribed {tail}.", "{head} takes {tail}."]),
                Relation("interacts_with", "Drug", "Drug", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} interacts with {tail}.", "{head} and {tail} have an interaction."]),
                Relation("side_effect", "Drug", "Symptom",
                         templates=["{head} causes {tail}.", "{tail} is a side effect of {head}."]),
                Relation("allergic_to", "Patient", "Ingredient",
                         templates=["{head} is allergic to {tail}."]),
                Relation("contains", "Drug", "Ingredient",
                         templates=["{head} contains {tail}."]),
                Relation("inhibits", "Drug", "Protein",
                         templates=["{head} inhibits {tail}."]),
            ],
            entity_types={
                "Patient": EntityType("Patient", ["Subject 1", "Subject 2", "Subject 3"]),
                "Drug": EntityType("Drug", ["Aspirin", "Warfarin", "Ibuprofen", "Tylenol", "Amoxicillin"]),
                "Ingredient": EntityType("Ingredient", ["Penicillin", "NSAID", "Acetaminophen"]),
                "Symptom": EntityType("Symptom", ["Bleeding", "Rash", "Liver Damage", "Stomach Pain"]),
                "Protein": EntityType("Protein", ["COX-1", "COX-2", "Thrombin"]),
            }
        )

# ============================================================================
# D18: Hospital Staff
# ============================================================================
@register_domain(CATEGORY_HEALTHCARE)
class HospitalStaffGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "hospital_staff"
    CATEGORY = CATEGORY_HEALTHCARE
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("attending_for", "Doctor", "Patient", templates=["{head} is the attending physician for {tail}."]),
            Relation("resident_under", "Doctor", "Doctor", templates=["{head} is a resident under {tail}."]),
            Relation("charge_nurse_of", "Nurse", "Department", templates=["{head} is the charge nurse of {tail}."]),
            Relation("on_call_for", "Doctor", "Department", templates=["{head} is on call for {tail}."]),
            Relation("covering_for", "Doctor", "Doctor", templates=["{head} is covering for {tail}."]),
        ], entity_types={"Doctor": EntityType("Dr", ["Dr. Smith", "Dr. Jones"]), "Nurse": EntityType("RN", ["Nurse Betty"]), "Patient": EntityType("Pt", ["John"]), "Department": EntityType("Dept", ["ER", "ICU"])})

# ============================================================================
# D19: Insurance
# ============================================================================
@register_domain(CATEGORY_HEALTHCARE)
class InsuranceGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "insurance"
    CATEGORY = CATEGORY_HEALTHCARE
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("covered_by", "Procedure", "Plan", templates=["{head} is covered by {tail}."]),
            Relation("denied_by", "Claim", "Insurer", templates=["{head} was denied by {tail}."]),
            Relation("in_network_with", "Doctor", "Plan", templates=["{head} is in-network with {tail}."]),
            Relation("requires_prior_auth", "Procedure", "Insurer", templates=["{head} requires prior auth from {tail}."]),
        ], entity_types={"Procedure": EntityType("Proc", ["MRI", "Surgery"]), "Plan": EntityType("Plan", ["Gold Plan", "Silver Plan"]), "Claim": EntityType("Claim", ["C123"]), "Insurer": EntityType("Ins", ["Aetna", "BlueCross"]), "Doctor": EntityType("Dr", ["Dr. Lee"])})

# ============================================================================
# D20: Medical Equipment
# ============================================================================
@register_domain(CATEGORY_HEALTHCARE)
class EquipmentGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "medical_equipment"
    CATEGORY = CATEGORY_HEALTHCARE
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("calibrated_by", "Device", "Technician", templates=["{head} was calibrated by {tail}."]),
            Relation("serviced_by", "Device", "Technician", templates=["{head} was serviced by {tail}."]),
            Relation("assigned_to_room", "Device", "Room", templates=["{head} is assigned to {tail}."]),
            Relation("manufactured_by", "Device", "Company", templates=["{head} was manufactured by {tail}."]),
        ], entity_types={"Device": EntityType("Dev", ["Ventilator", "X-Ray"]), "Technician": EntityType("Tech", ["Bob"]), "Room": EntityType("Rm", ["ICU-1", "OR-2"]), "Company": EntityType("Co", ["GE", "Siemens"])})

# ============================================================================
# D21: Anatomy
# ============================================================================
@register_domain(CATEGORY_HEALTHCARE)
class AnatomyGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "anatomy"
    CATEGORY = CATEGORY_HEALTHCARE
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("connected_to", "Part", "Part", properties={RelationProperty.SYMMETRIC}, templates=["{head} is connected to {tail}."]),
            Relation("innervated_by", "Part", "Nerve", templates=["{head} is innervated by {tail}."]),
            Relation("vascularized_by", "Part", "Artery", templates=["{head} is vascularized by {tail}."]),
            Relation("distal_to", "Part", "Part", templates=["{head} is distal to {tail}."]),
            Relation("proximal_to", "Part", "Part", inverse="distal_to", templates=["{head} is proximal to {tail}."]),
        ], entity_types={"Part": EntityType("Part", ["Hand", "Arm", "Heart"]), "Nerve": EntityType("Nerve", ["Vagus"]), "Artery": EntityType("Art", ["Aorta"])})
