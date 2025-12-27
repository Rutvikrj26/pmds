from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_ROBOTICS


# ==========================================
# Domain: Safety Interlocks
# ==========================================
@register_domain(CATEGORY_ROBOTICS)
class SafetyInterlocksGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "safety_interlocks"
    CATEGORY = CATEGORY_ROBOTICS
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("powered_by", "Device", "PowerSource",
                         templates=["{head} is powered by {tail}."]),
                Relation("enabled_by", "System", "Switch",
                         templates=["{head} is enabled by {tail}."]),
                Relation("disabled_by", "System", "Sensor",
                         templates=["{head} is disabled by {tail}."]),
                Relation("blocks_if", "Action", "Condition",
                         templates=["{head} is blocked if {tail}."]),
                Relation("safe_when", "Action", "Condition",
                         templates=["{head} is safe when {tail}."]),
                Relation("e_stop_controlled_by", "Machine", "Button",
                         templates=["{head} e-stop is controlled by {tail}."]),
            ],
            entity_types={
                "Device": EntityType("Device", ["RobotArm", "Conveyor", "Press"]),
                "PowerSource": EntityType("Source", ["24V-Bus", "Main-AC", "Battery-Backup"]),
                "System": EntityType("System", ["DriveUnit", "LogicController"]),
                "Switch": EntityType("Switch", ["KeySwitch", "DeadmanSwitch"]),
                "Sensor": EntityType("Sensor", ["LightCurtain", "DoorSwitch"]),
                "Action": EntityType("Action", ["Move", "Rotate", "Clamp"]),
                "Condition": EntityType("Cond", ["DoorOpen", "PresenceDetected", "Overheated"]),
                "Machine": EntityType("Machine", ["CNC", "Lathe"]),
                "Button": EntityType("Button", ["Estop-1", "Estop-2"]),
            }
        )

# ==========================================
# Domain: Sensors
# ==========================================
@register_domain(CATEGORY_ROBOTICS)
class SensorsGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "sensors"
    CATEGORY = CATEGORY_ROBOTICS
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("reads_from", "Controller", "Sensor",
                         templates=["{head} reads form {tail}."]),
                Relation("triggers", "Sensor", "Event",
                         templates=["{head} triggers {tail}."]),
                Relation("calibrated_with", "Sensor", "Tool",
                         templates=["{head} is calibrated with {tail}."]),
                Relation("monitored_by", "Zone", "Sensor",
                         templates=["{head} is monitored by {tail}."]),
                Relation("feeds_into", "Sensor", "Algorithm",
                         templates=["{head} data feeds into {tail}."]),
            ],
            entity_types={
                "Controller": EntityType("PLC", ["PLC-1", "EdgeNode"]),
                "Sensor": EntityType("Sensor", ["Lidar-A", "Camera-B", "Temp-C"]),
                "Event": EntityType("Event", ["StopRequest", "Alarm", "LogEntry"]),
                "Tool": EntityType("Tool", ["CalibTarget", "Multimeter"]),
                "Zone": EntityType("Zone", ["SafetyZone", "LoadingBay"]),
                "Algorithm": EntityType("Algo", ["Fusion", "ObjectDet"]),
            }
        )

# ==========================================
# Domain: Task Planning
# ==========================================
@register_domain(CATEGORY_ROBOTICS)
class TaskPlanningGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "task_planning"
    CATEGORY = CATEGORY_ROBOTICS
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("task_requires", "Task", "Resource",
                         templates=["{head} requires {tail}."]),
                Relation("task_before", "Task", "Task", properties={RelationProperty.TRANSITIVE},
                         templates=["{head} must be done before {tail}."]),
                Relation("conflicts_with", "Task", "Task", properties={RelationProperty.SYMMETRIC},
                         templates=["{head} conflicts with {tail}."]),
                Relation("preempts", "Task", "Task",
                         templates=["{head} preempts {tail}."]),
                Relation("assigned_to", "Task", "Robot",
                         templates=["{head} is assigned to {tail}."]),
            ],
            entity_types={
                "Task": EntityType("Task", ["Weld-A", "Pick-B", "Place-C"]),
                "Resource": EntityType("Res", ["Gripper", "Welder", "PaintGun"]),
                "Robot": EntityType("Bot", ["Arm-1", "Arm-2", "AGV-1"]),
            }
        )

# ============================================================================
# D36: Physical Layout
# ============================================================================
@register_domain(CATEGORY_ROBOTICS)
class LayoutGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "robot_layout"
    CATEGORY = CATEGORY_ROBOTICS
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("adjacent_to", "Zone", "Zone", properties={RelationProperty.SYMMETRIC}, templates=["{head} is adjacent to {tail}."]),
            Relation("reachable_from", "Point", "Point", templates=["{head} is reachable from {tail}."]),
            Relation("blocked_by", "Path", "Obstacle", templates=["{head} is blocked by {tail}."]),
            Relation("line_of_sight", "Sensor", "Target", templates=["{head} has line of sight to {tail}."]),
            Relation("docked_at", "Robot", "Station", templates=["{head} is docked at {tail}."]),
        ], entity_types={"Zone": EntityType("Zn", ["Z1"]), "Point": EntityType("Pt", ["P1"]), "Path": EntityType("Path", ["Path A"]), "Obstacle": EntityType("Obs", ["Wall"]), "Sensor": EntityType("Sens", ["Cam"]), "Target": EntityType("Tgt", ["Marker"]), "Robot": EntityType("Bot", ["R1"]), "Station": EntityType("Stn", ["Charger"])})

# ============================================================================
# D37: Manufacturing
# ============================================================================
@register_domain(CATEGORY_ROBOTICS)
class ManufacturingGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "manufacturing"
    CATEGORY = CATEGORY_ROBOTICS
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("assembled_from", "Product", "Component", templates=["{head} is assembled from {tail}."]),
            Relation("work_order_for", "Order", "Product", templates=["{head} is a work order for {tail}."]),
            Relation("produced_on", "Product", "Line", templates=["{head} is produced on {tail}."]),
            Relation("quality_checked_by", "Product", "Inspector", templates=["{head} checked by {tail}."]),
            Relation("scrapped_by", "Product", "Reason", templates=["{head} scrapped due to {tail}."]),
        ], entity_types={"Product": EntityType("Prod", ["Car"]), "Component": EntityType("Comp", ["Engine"]), "Order": EntityType("WO", ["WO123"]), "Line": EntityType("Line", ["Line 1"]), "Inspector": EntityType("Insp", ["Dave"]), "Reason": EntityType("Rsn", ["Defect"])})
