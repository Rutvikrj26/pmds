from scallop_titans.data.generators.base_generator import (
    BaseDomainGenerator, DomainConfig, Relation, RelationProperty, EntityType
)
from scallop_titans.data.generators.domain_registry import register_domain, CATEGORY_FINANCE


# ==========================================
# Domain: Trading
# ==========================================
@register_domain(CATEGORY_FINANCE)
class TradingGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "trading"
    CATEGORY = CATEGORY_FINANCE
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("bought", "Trader", "Asset",
                         templates=["{head} bought {tail}.", "{tail} was bought by {head}."]),
                Relation("sold", "Trader", "Asset",
                         templates=["{head} sold {tail}."]),
                Relation("listed_on", "Asset", "Exchange",
                         templates=["{head} is listed on {tail}."]),
                Relation("market_maker_for", "Firm", "Asset",
                         templates=["{head} is a market maker for {tail}."]),
                Relation("cleared_by", "Trade", "ClearingHouse",
                         templates=["{head} was cleared by {tail}."]),
            ],
            entity_types={
                "Trader": EntityType("Trader", ["Desk-A", "Quant-Fund", "Retail-Bob"]),
                "Asset": EntityType("Asset", ["AAPL", "GOOG", "BTC", "Gold"]),
                "Exchange": EntityType("Exch", ["NYSE", "NASDAQ", "CME"]),
                "Firm": EntityType("Firm", ["Citadel", "Virtu"]),
                "Trade": EntityType("Trade", ["T123", "T456"]),
                "ClearingHouse": EntityType("CH", ["DTCC", "OCC"]),
            }
        )

# ==========================================
# Domain: Banking
# ==========================================
@register_domain(CATEGORY_FINANCE)
class BankingGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "banking"
    CATEGORY = CATEGORY_FINANCE
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("account_holder", "Person", "Account",
                         templates=["{head} holds {tail}."]),
                Relation("transferred_to", "Account", "Account",
                         templates=["Funding transferred from {head} to {tail}."]),
                Relation("guaranteed_by", "Loan", "Person",
                         templates=["{head} is guaranteed by {tail}."]),
                Relation("issued_by", "Card", "Bank",
                         templates=["{head} was issued by {tail}."]),
                Relation("liable_for", "Person", "Loan",
                         templates=["{head} is liable for {tail}."]),
            ],
            entity_types={
                "Person": EntityType("Person", ["Alice", "Bob"]),
                "Account": EntityType("Acct", ["Checking-1", "Savings-2"]),
                "Loan": EntityType("Loan", ["Mortgage-A", "AutoLoan-B"]),
                "Card": EntityType("Card", ["Visa-1234", "MasterCard-5678"]),
                "Bank": EntityType("Bank", ["Chase", "BofA"]),
            }
        )

# ==========================================
# Domain: Taxation
# ==========================================
@register_domain(CATEGORY_FINANCE)
class TaxationGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "taxation"
    CATEGORY = CATEGORY_FINANCE
    
    def _default_config(self) -> DomainConfig:
        return DomainConfig(
            name=self.DOMAIN_NAME,
            category=self.CATEGORY,
            relations=[
                Relation("taxable_by", "Asset", "Jurisdiction",
                         templates=["{head} is taxable by {tail}."]),
                Relation("filed_return_in", "Person", "Jurisdiction",
                         templates=["{head} filed return in {tail}."]),
                Relation("exempt_from", "Asset", "TaxType",
                         templates=["{head} is exempt from {tail}."]),
                Relation("deductible_for", "Expense", "TaxType",
                         templates=["{head} is deductible for {tail}."]),
                Relation("audited_by", "Person", "Agency",
                         templates=["{head} was audited by {tail}."]),
            ],
            entity_types={
                "Asset": EntityType("Asset", ["Property-X", "Income-Y"]),
                "Person": EntityType("Person", ["Taxpayer-A", "Taxpayer-B"]),
                "Jurisdiction": EntityType("Jur", ["USA", "UK", "NY-State"]),
                "TaxType": EntityType("Tax", ["IncomeTax", "CapitalGains", "VAT"]),
                "Expense": EntityType("Exp", ["Charity", "Interest"]),
                "Agency": EntityType("Agency", ["IRS", "HMRC"]),
            }
        )

# ============================================================================
# D50: Portfolio Management
# ============================================================================
@register_domain(CATEGORY_FINANCE)
class PortfolioGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "portfolio_mgmt"
    CATEGORY = CATEGORY_FINANCE
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("holds_position_in", "Portfolio", "Asset", templates=["{head} holds a position in {tail}."]),
            Relation("diversified_in", "Portfolio", "Sector", templates=["{head} is diversified in {tail}."]),
            Relation("benchmarked_against", "Portfolio", "Index", templates=["{head} is benchmarked against {tail}."]),
            Relation("managed_by", "Portfolio", "Manager", templates=["{head} is managed by {tail}."]),
            Relation("rebalanced_on", "Portfolio", "Date", templates=["{head} was rebalanced on {tail}."]),
        ], entity_types={"Portfolio": EntityType("Port", ["Fund A"]), "Asset": EntityType("Asset", ["AAPL"]), "Sector": EntityType("Sec", ["Tech"]), "Index": EntityType("Idx", ["S&P500"]), "Manager": EntityType("Mgr", ["Ray"]), "Date": EntityType("Date", ["Q1"])})

# ============================================================================
# D52: Derivatives
# ============================================================================
@register_domain(CATEGORY_FINANCE)
class DerivativesGenerator(BaseDomainGenerator):
    DOMAIN_NAME = "derivatives"
    CATEGORY = CATEGORY_FINANCE
    def _default_config(self) -> DomainConfig:
        return DomainConfig(name=self.DOMAIN_NAME, category=self.CATEGORY, relations=[
            Relation("underlying_asset_of", "Asset", "Option", templates=["{head} is the underlying asset for {tail}."]),
            Relation("expires_on", "Option", "Date", templates=["{head} expires on {tail}."]),
            Relation("strike_price_is", "Option", "Price", templates=["{head} strike price is {tail}."]),
            Relation("hedges_against", "Derivative", "Risk", templates=["{head} hedges against {tail}."]),
            Relation("issued_by", "Derivative", "Bank", templates=["{head} issued by {tail}."]),
        ], entity_types={"Asset": EntityType("Asset", ["Oil"]), "Option": EntityType("Opt", ["Call"]), "Date": EntityType("Date", ["Dec"]), "Price": EntityType("Price", ["$100"]), "Derivative": EntityType("Deriv", ["Swap"]), "Risk": EntityType("Risk", ["Inflation"]), "Bank": EntityType("Bank", ["GS"])})
