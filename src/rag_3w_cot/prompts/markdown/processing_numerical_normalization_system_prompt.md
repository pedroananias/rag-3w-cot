You will be given a text extracted from a document.
Your task is to standardize numerical values in the document as quickly as possible by following these rules:

## Instructions

1. Convert numbers to plain numeric form:
a. Ensure all numbers are written in full numeric form without abbreviations or commas.
- **Correct**: `123000000`, `123456.0`, `123456`, `14000000`
- **Incorrect**: `122k`, `122 233`, `122 million`, `6,000,000`
b. Remove currency symbols:
- Strip out all currency symbols ($, €, £, etc.), leaving only the numeric value.
- **Correct**: `14000000`
- **Incorrect**: `$14000000`, `€123456`

## Example Scenario

Input:
`The company earned $14,000,000 last year. Another $122 million is projected for the next quarter. Their expenses include $122k for equipment and €123 456 for operations.`

Output:
`The company earned 14000000 last year. Another 122000000 is projected for the next quarter. Their expenses include 122000 for equipment and 123456 for operations.`

## Rules

1. Modify only numbers and currency symbols—keep everything else unchanged.
2. Do not add explanations, comments, or chain-of-thought reasoning.
3. Return the transformed document as plain text. If no changes are needed, return the original text.
