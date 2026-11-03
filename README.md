# early-warning-audit

Tests an early-warning model built on learning-platform logs as a decision rather than a score. It asks when a warning can still help, and who it misses.

## Why

Learning platforms log every click and submission. A common proposal is to read self-regulated learning from those traces, such as time management from when work comes in and help-seeking from forum use, and to warn students who fall behind. Domino and Shaffer (2025) judge such traces on validity, reliability and equitability. They note that the measures "are approximations, and there is no one truth" to check them against. They also name the null-data problem. A student with no forum trace may have decided the forum wasn't useful, or never thought about it.

An accuracy score can't settle whether a warning system works. This project puts three narrower questions to a deliberately ordinary model.

1. **Timing.** When does a warning come early enough to act on?
2. **Misses.** Does it miss more of the at-risk students in some groups than in others?
3. **Silence.** Does a missing trace mean the same thing for every group?

## Data

The [Open University Learning Analytics Dataset](https://doi.org/10.24432/C5KK69) (Kuzilek et al., 2017, CC BY 4.0) has 32,593 enrolments on 7 online modules from 2013 to 2014. It includes daily clicks in the learning platform, assessment submissions, final results, and demographics: deprivation band of the home area, declared disability and prior education. It isn't a computing course. I chose it because it is the public dataset that has both traces and demographics. The CodeWorkout data from the CSEDM challenges doesn't include these demographics, so it can't answer questions 2 and 3.

## How it works

- **Features use only the past.** At the end of each week from 0 to 20, the model sees only data dated before that point. That covers clicks by activity type, how recently and regularly the student was active, forum and live-tutorial use, quiz use, when assessments came in relative to the deadline, and how early they registered. Marks, demographics and module identity are left out. A test changes everything after the cutoff and checks that no feature moves.
- **Only reachable students count.** Each week only students still enrolled are scored. Students who have already withdrawn are counted separately, because no warning can reach them.
- **Train on the past, test on the next year.** The model trains on the 2013B, 2013J and 2014B presentations and is tested on 2014J (11,260 students, 5,803 of whom end up failing or withdrawing).
- **Three ways to rank students.** Gradient boosting is the main model. Logistic regression is a simpler check, and an inactivity rule (quietest first) needs no model at all.
- **A warning is a contact list.** Each week, tutors contact the 20% of enrolled students ranked most at risk. Intervals are 95% bootstrap intervals over students.

## Results

### 1. Waiting for a better warning doesn't reach more students

![AUC, precision and reach by week](results/by_week.png)

| End of week | AUC | Precision of the contact list | At-risk students already withdrawn | At-risk students reached |
|---|---|---|---|---|
| 0 (before the start) | 0.67 | 70% | 15% | 25% |
| 4 | 0.73 | 73% | 35% | 23% |
| 8 | 0.78 | 79% | 41% | 24% |
| 20 | 0.85 | 84% | 54% | 24% |

The warning gets more accurate every week, but the students it would catch keep leaving. One round of contacts reaches about a quarter of the eventual fails and withdrawals in any week. The list made before the course starts, from registration timing and pre-start activity alone, reaches as many as the list at week 20. A third of the students who will fail or withdraw have already withdrawn by the end of week 2. With a 30% budget the earliest list reaches the most, 35% against 30% at week 20. Logistic regression is within 0.02 AUC of boosting every week. The inactivity rule trails by 0.03 to 0.09.

### 2. The same share of at-risk students is missed in every group, but the risk scores ignore real differences

At week 4 the list misses 64% of at-risk students who are still enrolled, and every group sits within 3 points of that. No gap is distinguishable from zero, at week 4 or week 8. Precision differs (80% for the most deprived fifth of areas, 63% for the least deprived), as it must when base rates differ and miss rates are equal (Kleinberg et al., 2017).

![Predicted against observed risk by group](results/calibration.png)

The model's scores barely move between groups, even though real risk does. For students from the most and least deprived fifths of areas it predicts 47% and 46%, but 48% and 34% actually fail or withdraw. The pattern is the same for prior education (46% and 46% predicted, 48% and 33% observed) and disability (49% and 46% predicted, 53% and 40% observed). Clicks and submission timing carry little of what separates these groups. As a contact list it works about equally well for everyone. Read as a probability, it understates the risk for students with a declared disability or without A levels, by 2 to 4 points, and overstates it by about 12 points for students from the least deprived areas or with a degree. (Overall it predicts 47% against 41% observed, a sign the test year differs from the training years.)

### 3. A quiet week means less for students who declared a disability

![How much a missing trace raises risk, by group](results/missing_traces.png)

For most students a week with no platform activity is a strong signal. It raises the at-risk rate by 28 points at week 4. For students who declared a disability it raises it by 16 points, a gap of −12 points [−19, −4]. The result holds at week 8 (27 against 38 points, gap −11 [−19, −4]). These students are at higher risk even when active, 49% against 34%, so their activity is weaker evidence that they're fine. This is the null-data problem in a concrete form: the same trace, present or absent, means different things for different students. Never using the forum raises risk by about 20 points in every group at week 4, with no gap distinguishable from zero.

### 4. Which traces carry the warning

When one family of traces is shuffled at week 4, AUC drops by 0.085 for assessment timing and 0.080 for recency and regularity. It drops by 0.014 for click volume, 0.011 for registration timing, 0.006 for quizzes and 0.003 for forum and live-tutorial use. Help-seeking traces add almost nothing once the other traces are known. That doesn't mean help-seeking is unrelated to outcomes, only that it overlaps with the other traces. At week 4 the same model's AUC ranges from 0.68 to 0.76 across the seven modules.

Every table, with intervals, is in [results/summary.md](results/summary.md).

## Run it

```bash
pip install -e ".[dev]"
pytest
python -m ewaudit download     # OULAD from UCI, about 45 MB
python -m ewaudit run          # a few minutes on one core; writes results/
```

## Limits

- **Not a computing course, and not anyone's own platform.** The project shows a way to evaluate a warning system. It makes no claim about any particular course.
- **Auditing at-risk models for equity is an active area.** This isn't the first such audit. What it adds is tying the audit to the questions about trace validity and null data.
- **One test year.** Every result comes from the 2014J presentations.
- **Many comparisons, uncorrected.** The missing-trace table has 44 gaps across two weeks. The disability result is the one that holds at both weeks.
- **The groups are coarse.** Deprivation is measured for the home area, not the student. Disability is self-declared. Small education categories are merged.
- **Withdrawal mixes reasons.** Some students leave for reasons no tutor contact could change.
- **The 20% budget is an assumption.** The summary also has 10% and 30%.
- **Secondary analysis of public, anonymised data.** There was no student contact and no ethics review.

## References

- Domino, M., & Shaffer, C. A. (2025). Using a wider digital ecosystem to improve self-regulated learning. *Frontiers in Education*, 10. https://doi.org/10.3389/feduc.2025.1487344
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk scores. *Innovations in Theoretical Computer Science (ITCS)*.
- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

MIT license.
