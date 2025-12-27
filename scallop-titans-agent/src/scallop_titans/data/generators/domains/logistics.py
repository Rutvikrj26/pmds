from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_LOGISTICS


# ==========================================
# Domain: Supply Chain
# ==========================================
@register_domain(CATEGORY_LOGISTICS)
class SupplyChainGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "supply_chain"
    CATEGORY = CATEGORY_LOGISTICS
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("supplies", "Supplier", "Manufacturer",
                         templates=["{head} supplies {tail}.", "{tail} is supplied by {head}."]),
                Relation("manufactures", "Manufacturer", "Product",
                         templates=["{head} manufactures {tail}.", "{tail} is made by {head}."]),
                Relation("component_of", "Part", "Product",
                         templates=["{head} is a component of {tail}."]),
                Relation("distributed_by", "Product", "Distributor",
                         templates=["{head} is distributed by {tail}."]),
                Relation("retailer_sells", "Retailer", "Product",
                         templates=["{head} sells {tail}."]),
                Relation("contracted_carrier", "Carrier", "Distributor",
                         templates=["{head} is the contracted carrier for {tail}."]),
            ],
            entity_types={
                "Supplier": EntityType("Supplier", ["Supplier-A", "RawMat Corp", "Global Parts"]),
                "Manufacturer": EntityType("Manufacturer", ["Factory-X", "BuildIt Inc", "AutoMaker"]),
                "Product": EntityType("Product", ["Widget", "Gadget", "Vehicle-Z"]),
                "Part": EntityType("Part", ["Chip-X", "Steel-Sheet", "Battery"]),
                "Distributor": EntityType("Distributor", ["LogiCorp", "ShipFast"]),
                "Retailer": EntityType("Retailer", ["Store-Mart", "OnlineShop"]),
                "Carrier": EntityType("Carrier", ["FedEx", "Maersk", "DHL"]),
            }
        )

# ==========================================
# Domain: Inventory Management
# ==========================================
@register_domain(CATEGORY_LOGISTICS)
class InventoryMgmtGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "inventory_mgmt"
    CATEGORY = CATEGORY_LOGISTICS
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("stored_in", "Item", "Warehouse",
                         templates=["{head} is stored in {tail}."]),
                Relation("located_at", "Warehouse", "Location",
                         templates=["{head} is located at {tail}."]),
                Relation("low_stock_for", "Item", "Warehouse",
                         templates=["{head} is low stock in {tail}."]),
                Relation("replenished_by", "Warehouse", "Supplier",
                         templates=["{head} is replenished by {tail}."]),
                Relation("has_capacity", "Warehouse", "CapacityLevel",
                         templates=["{head} has {tail} capacity."]),
                Relation("batch_expires", "Batch", "Date",
                         templates=["{head} expires on {tail}."]),
            ],
            entity_types={
                "Item": EntityType("Item", ["SKU-101", "SKU-202", "SKU-303"]),
                "Warehouse": EntityType("Warehouse", ["WH-North", "WH-South", "WH-East"]),
                "Location": EntityType("Loc", ["New York", "London", "Tokyo"]),
                "Supplier": EntityType("Supplier", ["Vendor-A", "Vendor-B"]),
                "CapacityLevel": EntityType("Level", ["High", "Medium", "Full"]),
                "Batch": EntityType("Batch", ["Batch-001", "Batch-002"]),
                "Date": EntityType("Date", ["2024-12-31", "2025-01-30"]),
            }
        )

# ==========================================
# Domain: Shipping Routes
# ==========================================
@register_domain(CATEGORY_LOGISTICS)
class ShippingRoutesGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "shipping_routes"
    CATEGORY = CATEGORY_LOGISTICS
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("connected_to", "Port", "Port", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} is connected to {tail}.", "Route exists between {head} and {tail}."]),
                Relation("delivers_to", "Carrier", "Region",
                         templates=["{head} delivers to {tail}."]),
                Relation("has_hub", "Carrier", "Port",
                         templates=["{head} has a hub at {tail}."]),
                Relation("shipment_en_route", "Shipment", "Port",
                         templates=["{head} is currently at {tail}."]),
                Relation("destination_is", "Shipment", "Port",
                         templates=["{head} is destined for {tail}."]),
                Relation("customs_check_at", "Port", "Agency",
                         templates=["{head} requires customs check by {tail}."]),
            ],
            entity_types={
                "Port": EntityType("Port", ["Port of LA", "Port of Shanghai", "Rotterdam", "Singapore"]),
                "Carrier": EntityType("Carrier", ["Maersk", "MSC", "CMA CGM"]),
                "Region": EntityType("Region", ["West Coast", "Asia Pacific", "Europe"]),
                "Shipment": EntityType("Shipment", ["Cont-123", "Cont-456"]),
                "Agency": EntityType("Agency", ["CBP", "Customs Authority"]),
            }
        )

# ============================================================================
# D28: Fleet Management
# ============================================================================
@register_domain(CATEGORY_LOGISTICS)
class FleetGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "fleet_management"
    CATEGORY = CATEGORY_LOGISTICS
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("driven_by", "Vehicle", "Driver", templates=["{head} is driven by {tail}."]),
            Relation("assigned_route", "Vehicle", "Route", templates=["{head} is assigned to {tail}."]),
            Relation("scheduled_maintenance", "Vehicle", "Date", templates=["{head} is scheduled for maintenance on {tail}."]),
            Relation("fueled_at", "Vehicle", "Station", templates=["{head} was fueled at {tail}."]),
            Relation("inspected_by", "Vehicle", "Inspector", templates=["{head} was inspected by {tail}."]),
        ], entity_types={"Vehicle": EntityType("Veh", ["Truck 1"]), "Driver": EntityType("Driver", ["Driver A"]), "Route": EntityType("Rt", ["R1"]), "Date": EntityType("Date", ["2024-01-01"]), "Station": EntityType("Stn", ["Shell"]), "Inspector": EntityType("Insp", ["Bob"])})

# ============================================================================
# D29: Warehouse Operations
# ============================================================================
@register_domain(CATEGORY_LOGISTICS)
class WarehouseGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "warehouse_ops"
    CATEGORY = CATEGORY_LOGISTICS
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("moved_to", "Item", "Bin", templates=["{head} moved to {tail}."]),
            Relation("picked_from", "Item", "Bin", templates=["{head} picked from {tail}."]),
            Relation("packed_by", "Order", "Packer", templates=["{head} packed by {tail}."]),
            Relation("staged_at", "Order", "Dock", templates=["{head} staged at {tail}."]),
            Relation("zone_of", "Bin", "Zone", templates=["{head} is in zone {tail}."]),
        ], entity_types={"Item": EntityType("Item", ["Widget"]), "Bin": EntityType("Bin", ["A1-01"]), "Order": EntityType("Ord", ["O123"]), "Packer": EntityType("Packer", ["Alice"]), "Dock": EntityType("Dock", ["Dock 1"]), "Zone": EntityType("Zone", ["PickZone"])})

# ============================================================================
# D30: Delivery
# ============================================================================
@register_domain(CATEGORY_LOGISTICS)
class DeliveryGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "last_mile_delivery"
    CATEGORY = CATEGORY_LOGISTICS
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("scheduled_for_delivery", "Package", "Date", templates=["{head} scheduled for {tail}."]),
            Relation("delivered_by", "Package", "Driver", templates=["{head} delivered by {tail}."]),
            Relation("signed_for_by", "Package", "Recipient", templates=["{head} signed for by {tail}."]),
            Relation("refused_by", "Package", "Recipient", templates=["{head} refused by {tail}."]),
            Relation("attempted_delivery", "Driver", "Package", templates=["{tail} delivery attempted by {head}."]),
        ], entity_types={"Package": EntityType("Pkg", ["P1"]), "Date": EntityType("Date", ["Morning"]), "Driver": EntityType("Dr", ["Driver"]), "Recipient": EntityType("Rec", ["Customer"])})

# ============================================================================
# D31: Customs
# ============================================================================
@register_domain(CATEGORY_LOGISTICS)
class CustomsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "customs_brokerage"
    CATEGORY = CATEGORY_LOGISTICS
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("cleared_at", "Shipment", "Port", templates=["{head} cleared customs at {tail}."]),
            Relation("duty_paid_by", "Shipment", "Importer", templates=["Duty for {head} paid by {tail}."]),
            Relation("requires_permit", "Goods", "Agency", templates=["{head} requires permit from {tail}."]),
            Relation("quarantined_at", "Goods", "Facility", templates=["{head} quarantined at {tail}."]),
            Relation("declared_value", "Shipment", "Value", templates=["{head} declared value is {tail}."]),
        ], entity_types={"Shipment": EntityType("Shp", ["S1"]), "Port": EntityType("Port", ["LA"]), "Importer": EntityType("Imp", ["ImpCo"]), "Goods": EntityType("Gds", ["Fruit"]), "Agency": EntityType("Ag", ["FDA"]), "Facility": EntityType("Fac", ["Warehouse"]), "Value": EntityType("Val", ["$100"])})
