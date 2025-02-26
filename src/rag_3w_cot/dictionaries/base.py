from typing import Dict, List


class BaseTermsDictionary:
    terms: Dict[str, List[str]]

    @classmethod
    def expand(cls, question: str) -> str:
        expanded_query = question

        for key, synonyms in cls.terms.items():
            key_lower = key.lower()
            if key_lower in question.lower():
                expanded_synonyms = "(or " + ", ".join(synonyms) + ")"
                expanded_query = (
                    expanded_query.replace(key, f"{key} {expanded_synonyms}")
                    .replace(
                        key.capitalize(), f"{key.capitalize()} {expanded_synonyms}"
                    )
                    .replace(key.upper(), f"{key.upper()} {expanded_synonyms}")
                )

        return expanded_query
