import re
from typing import List, Dict, Any

class RuleEngine:
    def __init__(self):
        pass

    def extract_numbers(self, text: str) -> List[float]:
        return [float(x) for x in re.findall(r"\d+\.?\d*", text)]

    def extract_percentages(self, text: str) -> List[str]:
        return re.findall(r"\d+\s*%", text)

    def extract_dates(self, text: str) -> List[str]:
        return re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", text)

    def check_exclusion(self, chunk: str) -> str:
        exclusion_keywords = ["not covered", "excluded", "exclusion", "not payable", "not included"]
        for kw in exclusion_keywords:
            if kw in chunk.lower():
                return kw
        return ""

    def check_coverage(self, chunk: str) -> str:
        coverage_keywords = ["covered", "included", "payable", "reimbursed", "allowed"]
        for kw in coverage_keywords:
            if kw in chunk.lower():
                return kw
        return ""

    def check_waiting_period(self, chunk: str) -> int:
        match = re.search(r"(\d+)\s*month[s]?\s*waiting period", chunk.lower())
        if match:
            return int(match.group(1)) * 30
        match = re.search(r"(\d+)\s*day[s]?\s*waiting period", chunk.lower())
        if match:
            return int(match.group(1))
        return 0

    def evaluate(self, query: str, demographics: Dict[str, Any], duration_days: int, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        steps = []
        approved = False
        denied = False
        payout = None
        waiting_period = 0
        for clause in clauses:
            text = clause["text"]
            # Exclusion check
            excl_kw = self.check_exclusion(text)
            if excl_kw:
                steps.append({"clause": text, "result": "Denied due to exclusion.", "highlight": excl_kw})
                denied = True
                continue
            # Coverage check
            cov_kw = self.check_coverage(text)
            if cov_kw:
                steps.append({"clause": text, "result": "Clause indicates coverage.", "highlight": cov_kw})
                approved = True
                # Payout extraction
                percentages = self.extract_percentages(text)
                if percentages:
                    payout = percentages[0]
            # Waiting period check
            wp = self.check_waiting_period(text)
            if wp:
                steps.append({"clause": text, "result": f"Waiting period found: {wp} days.", "highlight": f"waiting period: {wp} days"})
                waiting_period = max(waiting_period, wp)
        # Multi-clause reasoning
        if denied:
            decision = "denied"
            justification = "One or more clauses explicitly exclude this scenario."
        elif approved:
            if waiting_period and (duration_days < waiting_period):
                decision = "pending"
                justification = f"Waiting period of {waiting_period} days not satisfied."
            else:
                decision = "approved"
                justification = "Relevant clauses indicate coverage."
        else:
            decision = "pending"
            justification = "No explicit coverage or exclusion found."
        return {
            "decision": decision,
            "payout": payout,
            "justification": justification,
            "steps": steps
        } 