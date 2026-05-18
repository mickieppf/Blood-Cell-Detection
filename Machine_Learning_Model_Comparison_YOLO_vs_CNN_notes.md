# Speaker Notes

## Slide 1: Title and thesis
- Frame the deck as an evidence-based model choice, not a general YOLO-vs-CNN debate.
- The thesis is that YOLO v26m at 150 epochs is the stronger deployable candidate on the held-out test split.

## Slide 2: The decision in one slide
- Use this as the executive summary if time is short.
- The key point is not that YOLO has higher raw mAP50-95 alone; it also beats CNN on all comparable shared metrics.

## Slide 3: The problem is imbalanced
- Explain that most errors will be driven by RBC because RBC is the dominant class.
- This is why precision and F1 are important alongside recall.

## Slide 4: Two detector strategies were tested
- Keep the setup simple: YOLO has multiple trained checkpoints, CNN has one checkpoint plus threshold tuning.
- The common comparison is held-out test behavior.

## Slide 5: YOLO wins the shared test scoreboard
- This is the central evidence slide.
- Call out precision first because the CNN baseline had high false positives in earlier tuning notes.

## Slide 6: The best YOLO was not the longest run
- Use this to show that the model decision is not simply more epochs equals better.
- The held-out test ranking selected a 150-epoch yolo26m checkpoint.

## Slide 7: Generalization separates the contenders
- This slide explains why the held-out test summary is more important than training validation alone.
- Mention that all rows were marked higher train in the evaluation CSV.

## Slide 8: CNN tuning helped, but not enough
- Do not dismiss CNN; explain that tuning was meaningful.
- Then transition to the fact that the tuned CNN still trails YOLO on F1 and mAP50.

## Slide 9: Class risk is concentrated in RBC and Platelets
- This is a limitation slide: aggregate wins are not enough for medical imaging workflows.
- State that YOLO aggregate metrics win, but per-class YOLO export should be added next.

## Slide 10: Qualitative check: what the models draw
- Use this as a human sanity check, not as a replacement for the metrics.
- The visual examples make the problem concrete for non-technical stakeholders.

## Slide 11: Recommendation and next steps
- End with a direct model recommendation.
- Make clear what evidence is still missing: per-class YOLO, latency, model size, and external validation.

## Slide 12: Reproducibility map
- Use this slide if someone asks where the numbers came from.
- Call out the main assumption: both comparisons are anchored to the same local test split.
