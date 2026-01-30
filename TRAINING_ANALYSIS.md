# Training Results Analysis

## Executive Summary

**Date:** January 30, 2026  
**Models Trained:** 
1. Baseline CNN (Custom Architecture)
2. MobileNetV2 (Transfer Learning)

**Overall Performance:**
- **Baseline CNN:** 69.16% accuracy ✅
- **MobileNetV2:** ~65% accuracy (still converging)

---

## 1. Baseline CNN Performance

### Overall Metrics
- **Accuracy:** 69.16%
- **Macro Avg F1:** 0.6613
- **Weighted Avg F1:** 0.6863

### Per-Emotion Performance

| Emotion  | Precision | Recall | F1-Score | Support | Performance |
|----------|-----------|--------|----------|---------|-------------|
| Happy    | 0.8772    | 0.9019 | 0.8894   | 1774    | ⭐ Excellent |
| Surprise | 0.7822    | 0.8123 | 0.7969   | 831     | ⭐ Good      |
| Disgust  | 0.7160    | 0.5225 | 0.6042   | 111     | ⚠️ Low Recall |
| Angry    | 0.6156    | 0.6336 | 0.6245   | 958     | ✅ Decent    |
| Neutral  | 0.5887    | 0.7242 | 0.6495   | 1233    | ✅ Decent    |
| Fear     | 0.6068    | 0.4023 | 0.4839   | 1024    | ❌ Poor      |
| Sad      | 0.5855    | 0.5766 | 0.5810   | 1247    | ⚠️ Mediocre  |

### Key Observations

**✅ Strengths:**
1. **Happy** emotion: Near 90% F1-score - model excels at detecting positive expressions
2. **Surprise** emotion: ~80% F1-score - distinctive facial features help
3. **Good convergence:** Training curves show smooth learning without severe overfitting

**❌ Weaknesses:**
1. **Fear** emotion: Only 40% recall - frequently misclassified as other emotions
2. **Disgust** emotion: Low recall (52%) despite good precision - likely due to class imbalance (only 111 samples)
3. **Sad/Neutral confusion:** These emotions share subtle facial features

---

## 2. Confusion Matrix Analysis

### Most Common Misclassifications

From the confusion matrix, the biggest confusion patterns are:

1. **Fear → Sad** (204 instances)
   - *Why:* Both involve downturned mouth, similar eye expressions
   
2. **Fear → Neutral** (137 instances)
   - *Why:* Subtle fear expressions can appear neutral
   
3. **Neutral → Sad** (151 instances)
   - *Why:* Neutral faces with slight frowns get misread
   
4. **Sad → Neutral** (264 instances)
   - *Why:* Bidirectional confusion - these emotions are inherently similar

5. **Angry → Neutral** (115 instances)
   - *Why:* Resting faces can appear slightly angry

### Implications
- **Fear** is the most problematic emotion - needs targeted improvement
- **Neutral/Sad/Fear** form a "confusion triangle" - hard to distinguish
- **Happy/Surprise** are well-separated from others (good!)

---

## 3. Training Curves Analysis

### Baseline CNN
- **Convergence:** Reached plateau around epoch 30-40
- **Overfitting:** Minimal gap between train/val accuracy (good generalization)
- **Loss:** Smooth decrease, stabilized around 0.8-0.9

### MobileNetV2
- **Convergence:** Still improving at epoch 35 (could benefit from more epochs)
- **Overfitting:** Slight gap emerging but not severe
- **Loss:** Decreasing steadily, not yet plateaued

**Recommendation:** MobileNetV2 could be trained longer (50-60 epochs) for better results.

---

## 4. Class Imbalance Impact

### Dataset Distribution (from support column)

| Emotion  | Samples | % of Dataset |
|----------|---------|--------------|
| Happy    | 1774    | 24.7%        |
| Neutral  | 1233    | 17.2%        |
| Sad      | 1247    | 17.4%        |
| Fear     | 1024    | 14.3%        |
| Angry    | 958     | 13.3%        |
| Surprise | 831     | 11.6%        |
| Disgust  | 111     | 1.5% ⚠️      |

**Critical Issue:** Disgust has only 111 samples (1.5% of dataset)
- This explains the low recall (52.25%)
- Model rarely predicts "disgust" to avoid false positives

---

## 5. Comparison: Baseline vs MobileNetV2

| Metric              | Baseline CNN | MobileNetV2 | Winner       |
|---------------------|--------------|-------------|--------------|
| Accuracy            | 69.16%       | ~65%        | Baseline     |
| Model Size          | 86 MB        | 31 MB       | MobileNetV2  |
| Training Speed      | Slower       | Faster      | MobileNetV2  |
| Inference Speed     | Slower       | Faster      | MobileNetV2  |
| Generalization      | Good         | TBD         | TBD          |

**Verdict:** 
- Baseline CNN is currently better for accuracy
- MobileNetV2 is better for deployment (smaller, faster)
- MobileNetV2 needs more training epochs to catch up

---

## 6. Next Steps (Phase 2 Recommendations)

### Immediate Actions
1. ✅ **Document these findings** (this file)
2. ✅ **Commit .gitignore update** (exclude dataset/)
3. 🔄 **Address Class Imbalance:**
   - Implement weighted loss function
   - Try oversampling for Disgust class
   - Consider SMOTE or data augmentation specifically for minority classes

### Phase 2 Implementation Plan
1. **Modify training scripts to handle class imbalance**
   - Add `class_weight` parameter to model.fit()
   - Calculate weights based on inverse frequency
   
2. **Retrain with balanced approach**
   - Focus on improving Fear and Disgust performance
   
3. **Document bias and limitations**
   - Lighting sensitivity
   - Dataset bias (FER-2013 is grayscale webcam images)
   - Cultural differences in emotion expression

---

## 7. Interview Talking Points

When discussing this project in interviews, highlight:

### Technical Depth
- "Trained two architectures: custom CNN and MobileNetV2 transfer learning"
- "Achieved 69% accuracy, with Happy emotion at 89% F1-score"
- "Identified class imbalance as key issue - Disgust only 1.5% of data"

### ML Maturity
- "Analyzed confusion matrix to understand failure modes"
- "Fear/Sad/Neutral form a confusion triangle due to subtle facial differences"
- "Implemented proper train/val split with callbacks to prevent overfitting"

### Real-World Awareness
- "Acknowledged dataset limitations - FER-2013 has known biases"
- "Planned Phase 2 to address class imbalance with weighted loss"
- "Considered deployment tradeoffs - MobileNetV2 is 3x smaller for edge devices"

---

## Conclusion

**Phase 1 Status:** ✅ **COMPLETE**

We have successfully:
- Trained our own models (not just using pre-trained black boxes)
- Evaluated with proper metrics (Precision, Recall, F1, Confusion Matrix)
- Analyzed failure modes and identified root causes
- Compared multiple architectures

**This project is now at ~7.5/10 for ML interviews.**

To reach 9/10, we need:
- Phase 2: Class imbalance handling ✅ Ready to implement
- Phase 3: Probabilistic emotion output (use softmax probabilities)
- Phase 4: User feedback loop
- Phase 5: Documentation and MLOps practices

**Ready to proceed with Phase 2!**
