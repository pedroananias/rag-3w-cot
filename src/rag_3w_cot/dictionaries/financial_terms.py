from typing import Dict, List

from .base import BaseTermsDictionary


class FinancialTermsDictionary(BaseTermsDictionary):
    terms: Dict[str, List[str]] = {
        "net income": ["profit", "earnings"],
        "gross profit": ["margin", "income"],
        "operating profit": ["EBIT", "earnings"],
        "profit before tax": ["EBT"],
        "profit after tax": ["PAT"],
        "EBITDA": ["cashflow"],
        "revenue": ["sales", "income"],
        "gross revenue": ["sales"],
        "operating revenue": ["income"],
        "operating costs": ["Opex", "expenses"],
        "SG&A": ["expenses"],
        "R&D": ["innovation"],
        "deferred tax": ["liability", "asset"],
        "assets": ["holdings", "PP&E"],
        "liabilities": ["debts"],
        "equity": ["capital", "worth"],
        "EPS": ["earnings"],
        "free cash flow": ["FCF"],
        "investing cash flow": ["investments"],
        "financing cash flow": ["debt"],
        "comprehensive income": ["earnings"],
        "dividend": ["payout"],
        "capital expenditures": ["CapEx"],
        "ROA": ["return"],
        "ROE": ["return"],
        "ROI": ["return"],
        "current ratio": ["liquidity"],
        "quick ratio": ["liquidity"],
        "debt-to-equity ratio": ["leverage"],
        "P/E ratio": ["valuation"],
        "ROS": ["return"],
        "EBIT": ["earnings"],
        "D&A": ["depreciation"],
        "accounts payable": ["payables"],
        "accounts receivable": ["receivables"],
        "sustainability": ["ESG", "CSR", "impact", "eco-friendly", "green", "climate"],
    }
