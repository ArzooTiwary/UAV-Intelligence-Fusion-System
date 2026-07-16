# UAV Intelligence Fusion System
**Live Demo:** https://uav-intelligence-fusion-system.streamlit.app

I started this project with one question that kept bothering me during the literature review phase of my BSERC Summer internship: why does every UAV security paper solve only one problem at a time?

You have excellent work on GPS spoofing detection. Separately, strong work on aerial object detection. Separately again, work on telemetry anomaly classification. But an actual drone in an actual surveillance scenario faces all of these simultaneously, and the operator on the ground needs one answer, not three separate alerts from three separate systems that don't know about each other.

That gap is what this project is trying to close.

## What the system actually does

UAV-IFS takes the three main data streams a surveillance drone produces; visual imagery, GPS telemetry, and flight state data processes each one through a dedicated AI module, and then fuses all three into a single threat assessment with an explicit confidence measure and a plain-English report that any operator can read and act on.

The fusion step is the part I spent the most time on. The naive approach is to assign point scores: +10 for a vehicle detection, +40 if GPS looks suspicious, +30 if you're in a restricted zone, add them up and threshold. That works for a demo but it breaks immediately when you think about it carefully. It treats every vehicle detection as equally threatening regardless of where the drone is or what else is happening. Five vehicles at a border checkpoint during a shift change is completely normal. Five vehicles in an exclusion zone at 2am with the GPS signal showing a discontinuous jump is a serious incident. A point-scoring system gives you the same number for both.

The fusion engine here uses Dempster-Shafer evidence theory instead. Each module doesn't output a label, it outputs a *belief mass* distributed across three hypotheses: THREAT, BENIGN, and UNKNOWN. The UNKNOWN mass is the part that makes DS theory useful. It lets a module say "I'm not sure" rather than forcing it to pick a side. When you combine evidence from multiple modules using Dempster's combination rule, uncertain evidence gets appropriately down-weighted rather than polluting the final score. The context module (restricted zone, nighttime) doesn't detect anything on its own, it shifts how much weight the system places on what the other two modules found. Same evidence, different context, different conclusion. That's the core idea.


## Architecture

The system is built in six layers, each feeding into the next.

Visual data flows through a YOLOv8 detector that outputs structured object
counts. GPS and IMU telemetry runs through an Extended Kalman Filter that
produces a state estimate and, more importantly, an innovation residual,
a measure of how surprised the filter was by each new GPS reading. That
residual feeds directly into the anomaly classifier alongside raw telemetry
features, because filter residuals are a far stronger spoofing signal than
raw GPS coordinates alone.

All three outputs; vision counts, cyber anomaly probability, and contextual
zone information feed into the fusion engine simultaneously. The fusion
engine combines them using Dempster-Shafer evidence theory and produces a
single threat score with a confidence interval. The report generator then
converts that structured output into a plain-English summary, and the
dashboard presents everything together in one operator-facing interface.

The key architectural decision was making the fusion engine sit *above* all
three sensing layers rather than being a downstream consumer of a single
primary detector. Most systems treat cybersecurity as an afterthought bolted
onto a vision pipeline. Here it's a co-equal input to the same decision.

## Module breakdown

### Module 1- Visual Intelligence (`detect_image.py`)

YOLOv8n running inference on aerial imagery, classifying detections into three operationally relevant categories: vehicles, persons, and boats. The model used here is the pretrained COCO-weight version. For real deployment on aerial imagery, fine-tuning on VisDrone or DOTA is the right next step; small-object recall on top-down UAV imagery drops noticeably with stock COCO weights, which is a well-documented limitation in the aerial detection literature.

Output is a structured JSON object:
```json
{"vehicles": 4, "people": 2, "boats": 0}
```

This is a deliberate design choice. The module outputs counts, not bounding boxes or class labels. The fusion engine downstream doesn't need pixel coordinates; it needs operationally meaningful quantities. Keeping the interface simple makes the modules genuinely composable.

### Module 2- State Estimation (`module2_ekf.py`)

An Extended Kalman Filter tracking UAV position [lat, lon, alt] using a constant-velocity motion model. On every update step, the filter computes the *innovation*; the difference between what the GPS reported and what the filter predicted from physics alone. This innovation vector, expressed as a Mahalanobis distance, is the key output.

The reason for doing this rather than feeding raw GPS directly to a classifier: a spoofing attack manifests as a discontinuous jump in position that the filter's motion model considers highly improbable. That improbability is quantified as a large Mahalanobis residual. During normal flight the residual sits between 0.4 and 2.0. During the simulated spoofing attack in testing it spikes above 100. This is a much stronger anomaly signal than the raw GPS values themselves, because it isolates the *surprise* in the measurement rather than its absolute value.

### Module 3- Cyber Anomaly Detection (`module3_classifier.py`)

A Random Forest classifier trained to distinguish clean flight telemetry from GPS spoofing attacks. The feature set deliberately includes the EKF innovation residual from Module 2, alongside raw telemetry features (speed, altitude change, heading change). This is the key design decision, using filter residuals as classifier features rather than raw GPS values significantly improves discriminative power, because the residual captures the dynamic inconsistency of a spoofed signal rather than just its absolute position.

The module includes an adversarial robustness test: bounded random perturbations are applied to the feature vectors of correctly-classified spoofed samples, and recall is re-measured. This matters because a classifier that performs well on clean test data but degrades sharply under small input perturbations is not robust enough for a real threat-detection application.

Output:
```json
{"status": "spoofed", "confidence": 0.91}
```

### Module 4- Fusion Engine (`module4_fusion.py`)

This is the core of the project. Three evidence sources, each expressed as a basic probability assignment over {THREAT, BENIGN, UNKNOWN}, combined pairwise using Dempster's combination rule.

The combination rule:
- Agreement (both sources pointing the same direction)= reinforces belief
- Conflict (sources pointing opposite directions)= generates conflict mass K
- Conflict mass is discarded and the remaining masses are renormalized by 1/(1-K)

The context module is where the system's behaviour becomes genuinely interesting. It outputs a strong BENIGN lean in normal zones during daytime, and a moderate THREAT lean in restricted zones at night. Combined with vision and cyber evidence, this means the same 5-vehicle detection produces a threat score of 0.02 in a normal patrol corridor and 0.95 in a restricted exclusion zone with active GPS anomaly. That context-sensitivity is the property that makes this system research-grade rather than a threshold alert.

Belief and plausibility bounds are both reported, giving the operator a proper confidence interval rather than a single point estimate.

### Module 5- Report Generator (`module5_report.py`)

Template-based natural language generation. Every sentence in the output report is constructed directly from fields in the fusion JSON. There is no free text generation. This is intentional; in a system where an operator may take real action based on a report, every claim must be traceable to a specific input. The module includes a consistency check function that verifies generated text doesn't contain numbers not present in the source data.

### Module 6- Operator Dashboard (`module6_dashboard.py`)

Streamlit application with Plotly visualisations. The dashboard includes:

- A threat gauge showing the fused Dempster-Shafer belief score
- Stacked horizontal bar chart showing the evidence breakdown across all three modules, this directly visualises the DS maths and is useful for explaining the system to someone unfamiliar with the theory
- EKF innovation residual timeline, showing the real filter output across 120 timesteps with the anomaly window highlighted, when the GPS spoofing slider is raised above 0.3, the timeline turns red and the attack window is marked with a vertical line
- Auto-generated intelligence report
- Raw DS belief masses for all three evidence sources


## Running it

```bash
pip install ultralytics opencv-python scikit-learn streamlit plotly numpy pandas
```

Test individual modules:
```bash
python code/detect_image.py your_image.jpg
python code/module2_ekf.py
python code/module3_classifier.py
python code/module4_fusion.py
python code/module5_report.py
```

Launch the dashboard:
```bash
python -m streamlit run code/module6_dashboard.py
```

## What's synthetic and what isn't

The GPS telemetry used to train Module 3 and drive the EKF timeline in the dashboard is synthetically generated, clean flight paths with simulated spoofing attacks injected at random timesteps. Real GPS spoofing datasets exist (the Mendeley dataset by Aissou, 2022, is the standard benchmark) and plugging one in is a straightforward swap of the data loading step. The synthetic data was used here to keep the project self-contained and reproducible without external downloads.

The vision module uses real YOLOv8 inference on real images. The fusion maths is real. The EKF is a real filter running real equations, not a simulated output.


## What's next

The obvious extensions, roughly in order of research value:

1. Fine-tune Module 1 on VisDrone, this is the single highest-impact change for real aerial deployment
2. Replace synthetic telemetry with the Mendeley GPS spoofing dataset and re-evaluate Module 3 with proper cross-validation
3. Add multi-object tracking (ByteTrack or OC-SORT) to Module 1 so loitering is detected from actual track history rather than a checkbox
4. Extend Module 3 to a CNN+gradient-boosting stacked ensemble for improved robustness
5. Connect the dashboard to a live video feed and real telemetry stream

## Model Weights

The fine-tuned weights are not in this repository — a 6MB binary file does not 
belong in version control. To reproduce the fine-tuning from scratch, the 
Colab notebook is at `code/visdrone_finetune.ipynb`. It takes about two hours 
on a free T4 GPU. Download `best.pt` when it finishes and place it at 
`models/yolov8n_visdrone.pt`. If you just want to run the system quickly, 
remove the model path override in `detect_image.py` and it will fall back to 
stock `yolov8n.pt` which downloads automatically.

On the VisDrone 2019 detection dataset — 6,471 training images, 548 validation 
images, 10 aerial-specific classes — fine-tuning for 50 epochs on a Tesla T4 
brought mAP50 from 0.110 at epoch 1 to 0.298 at convergence, a 171% improvement. 
Car detection reached mAP50 of 0.723, which makes sense given cars are the most 
represented and visually consistent class in aerial imagery. The weakest classes 
were bicycle (0.061) and awning-tricycle (0.103) — both rare in the dataset and 
visually ambiguous at altitude. This is consistent with findings across the aerial 
detection literature and points to data augmentation or class-weighted sampling 
as a natural next step.

## Important note
This is a non-weaponized situational awareness and decision support platform. Every output is advisory. The system does not recommend, initiate, or score any offensive or kinetic action. It tells an operator what it sees and how confident it is. The operator decides what to do.

By:
Arzoo Tiwary,
Automation and Robotics,
SRM Institute of Science and Technology, 
*BSERC Summer Internship, 2026*.
