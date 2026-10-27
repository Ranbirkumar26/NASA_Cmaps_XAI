# Expert Review Packet: C-MAPSS XAI Maintenance Advisor

## Review purpose

This packet collects maintenance-facing feedback on whether RUL explanations help prioritize inspection decisions. It must be completed by a real maintenance, reliability, or mechanical systems expert before any final report claims human validation.

## Cases to review

Use outputs from `results/xai/FD001/maintenance_advisor.csv`:

1. High-risk case: lowest predicted RUL row.
2. Medium-risk case: lowest medium-risk row, if present.
3. Low-risk case: representative low-risk row.

For each case, provide:

1. Unit ID and end cycle.
2. Actual RUL and predicted RUL.
3. Risk label.
4. Top sensors from `top_sensors`.
5. Temporal attribution chart.
6. Nearest-healthier-window counterfactual.
7. DiCE counterfactual if available.

## Questions for expert

| Dimension | Question | Rating |
| --- | --- | --- |
| Interpretability | Are top sensors understandable for maintenance review? | 1 to 5 |
| Faithfulness perception | Does the explanation match sensor trends? | 1 to 5 |
| Actionability | Does the explanation help prioritize inspection? | 1 to 5 |
| Human usefulness | Would this reduce investigation time? | 1 to 5 |
| Trust | Would explanation increase trust compared with prediction alone? | 1 to 5 |
| Safety concern | Does any explanation look misleading or unsafe? | Yes or no |

## Written feedback prompts

1. Which sensor explanations were most useful?
2. Which explanation was unclear or not actionable?
3. Did counterfactual evidence help inspection prioritization?
4. Did any prediction need caution despite explanation?
5. What dashboard change would make this more useful?

## Reporting rule

Do not claim expert validation until completed ratings and comments exist. If feedback is pending, report the protocol and mark expert validation as pending.

