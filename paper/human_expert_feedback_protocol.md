# Human Expert Feedback Protocol for C-MAPSS XAI Maintenance Advisor

## Purpose

Collect domain feedback on whether RUL explanations improve maintenance decision support. This protocol must be completed with a real maintenance, reliability, or mechanical systems expert before final submission if expert feedback is claimed in the report.

## Reviewer Profile

Eligible reviewer:

1. Aircraft maintenance engineer.
2. Reliability engineer.
3. Mechanical systems engineer.
4. Industrial maintenance expert with sensor-based monitoring experience.

Record reviewer role, years of experience, review date, and review mode in the final submission.

## Review Cases

Use three model outputs:

1. High-risk engine case with low predicted RUL.
2. Medium-risk engine case.
3. Low-risk engine case.

Each case must show:

1. Engine unit ID.
2. End cycle.
3. Predicted RUL.
4. Actual RUL when available.
5. Risk label.
6. Top sensors from attribution.
7. Temporal attribution summary.
8. Counterfactual healthier-window evidence.
9. Suggested maintenance action.

## Rating Rubric

| Dimension | Question | Rating |
| --- | --- | --- |
| Interpretability | Are top sensors understandable for maintenance review? | 1 to 5 |
| Faithfulness perception | Does the explanation appear consistent with sensor trends? | 1 to 5 |
| Actionability | Does the explanation help choose inspection priority? | 1 to 5 |
| Human usefulness | Would this reduce manual investigation time? | 1 to 5 |
| Trust | Would explanation increase trust compared with prediction alone? | 1 to 5 |
| Safety concern | Does any explanation appear misleading or unsafe? | Yes or no |

Rating meaning:

1. Strongly disagree.
2. Disagree.
3. Neutral.
4. Agree.
5. Strongly agree.

## Required Written Feedback

Ask reviewer to answer:

1. Which sensor explanations were most useful?
2. Which explanation was confusing or not actionable?
3. Did counterfactual evidence help inspection prioritization?
4. Did any prediction need caution despite explanation?
5. What dashboard change would make this useful in real maintenance workflow?

## Reporting Rule

Do not report expert feedback unless a real reviewer completed the review. If feedback is not yet collected, report the protocol and state that expert validation remains pending.

