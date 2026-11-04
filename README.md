# early-warning-audit

[![tests](https://github.com/shahbanobaber/early-warning-audit/actions/workflows/tests.yml/badge.svg)](https://github.com/shahbanobaber/early-warning-audit/actions/workflows/tests.yml)

**[Try it live →](https://early-warning-audit.vercel.app)**

Universities flag struggling students from their online activity. This checks whether the warning reaches them in time, and who it misses.

![Accuracy, precision and reach by week](results/by_week.png)

## Findings

Tested on 11,260 students (October 2014), repeated on 7,804 (February 2014).

- **Waiting doesn't help.** Any one week's call list reaches about a quarter of struggling students. Later lists are more accurate, but by then many students have left.
- **Calling again does.** Calls at weeks 0, 4 and 8 reach 59%, against 41% at random.
- **It misses students who look fine.** 80% of the missed future fails were active the week before, and 1% had missed a deadline. Adding marks catches only 1 in 8 of them.
- **Equal misses, wrong risk numbers.** Every group is missed at about the same rate, but the warning predicts about 46% risk for both the richest and poorest areas. The real figures are 34% and 48%.
- **Going quiet means less for some.** A quiet week adds 16 points of risk for students who declared a disability, against 28 for everyone else.

## How it works

- **Data.** Open University Learning Analytics Dataset: 32,593 students, 7 courses, 2013–2014.
- **Weekly snapshots.** Each week, every student is described from past data only: clicks, regularity, forum, quizzes and deadlines. A test checks that nothing leaks in from the future.
- **Model.** Gradient boosting (scikit-learn), trained on earlier years and tested on the next.
- **Contact list.** Tutors call the top 20% each week. Students who have already left count as unreachable.
- **Checks.** Equal opportunity (Hardt et al., 2016), calibration (Kleinberg et al., 2017), missing traces (Domino and Shaffer, 2025), trace importance by shuffling (Breiman, 2001) and bootstrap intervals (Efron, 1979).

Full tables are in [results/summary.md](results/summary.md) and [results/replication-2014B/summary.md](results/replication-2014B/summary.md).

## Run it

```bash
pip install -e ".[dev]"
pytest
python -m ewaudit download          # about 45 MB
python -m ewaudit run               # about 6 minutes on one core
python -m ewaudit run --test 2014B --out results/replication-2014B
python -m ewaudit web               # data for the live page
```

## Limits

- One university, two test years, and not a computing course.
- Many comparisons, uncorrected.
- The 20% budget and the 14-day marking lag are assumptions.
- Public, anonymised data, with no student contact.

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324
- Domino, M., & Shaffer, C. A. (2025). Using a wider digital ecosystem to improve self-regulated learning. *Frontiers in Education*, 10. https://doi.org/10.3389/feduc.2025.1487344
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *The Annals of Statistics*, 7(1), 1-26. https://doi.org/10.1214/aos/1176344552
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*, 29. arXiv:1610.02413
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk scores. *ITCS*. https://doi.org/10.4230/LIPIcs.ITCS.2017.43
- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171 (copy used: https://doi.org/10.24432/C5KK69)
- Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

MIT license.
