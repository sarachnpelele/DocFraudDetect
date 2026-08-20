"""
Complete comparative visualizations of the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

#Plot 1: F1 comparison, before vs after Stage 2, both datasets 
categories = ['DocTamper\n(3-stream)', 'DocTamper\n(4-stream)', 'AIForge-Doc\n(3-stream)', 'AIForge-Doc\n(4-stream)']
f1_scores = [0.4838, 0.9343, 0.0519, 0.9150]
colors = ['#ff9999', '#66b3ff', '#ff9999', '#66b3ff']

plt.figure(figsize=(9, 6))
bars = plt.bar(categories, f1_scores, color=colors)
plt.ylabel('Pixel-F1 Score')
plt.title('Real Performance Comparison: 3-Stream Baseline vs 4-Stream Complete Model')
plt.ylim(0, 1)
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{score:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/plot_f1_comparison.png', dpi=200)
plt.close()
print("Saved plot_f1_comparison.png")

#Plot 2: IoU comparison, before vs after, both datasets 
iou_scores = [0.2141, 0.8755, 0.0238, 0.8434]

plt.figure(figsize=(9, 6))
bars = plt.bar(categories, iou_scores, color=colors)
plt.ylabel('Pixel-IoU Score')
plt.title('Real Performance Comparison (IoU): 3-Stream Baseline vs 4-Stream Complete Model')
plt.ylim(0, 1)
for bar, score in zip(bars, iou_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{score:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('results/plot_iou_comparison.png', dpi=200)
plt.close()
print("Saved plot_iou_comparison.png")

#Plot 3: ROC curves, real, all three 4-stream evaluations 
plt.figure(figsize=(8, 8))
for label, path in [
    ('DocTamper (main test)', 'results/raw_predictions_4stream.npz'),
    ('AIForge-Doc (AI-inpainting)', 'results/raw_predictions_4stream_aiforge.npz'),
    ('FCD (cross-domain)', 'results/raw_predictions_4stream_fcd.npz'),
]:
    data = np.load(path)
    preds = data['preds']
    labels = data['labels']
    rng = np.random.default_rng(seed=42)
    sample_size = min(2_000_000, len(preds))
    idx = rng.choice(len(preds), size=sample_size, replace=False)
    fpr, tpr, _ = roc_curve(labels[idx], preds[idx])
    plt.plot(fpr, tpr, label=label, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random chance', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: 4-Stream Model Across Real Evaluation Sets')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('results/plot_roc_curves.png', dpi=200)
plt.close()
print("Saved plot_roc_curves.png")

#Plot 4: AIForge-Doc specific, 3-stream vs 4-stream ROC
plt.figure(figsize=(8, 8))
data_4s = np.load('results/raw_predictions_4stream_aiforge.npz')
rng = np.random.default_rng(seed=42)
sample_size = min(2_000_000, len(data_4s['preds']))
idx = rng.choice(len(data_4s['preds']), size=sample_size, replace=False)
fpr_4s, tpr_4s, _ = roc_curve(data_4s['labels'][idx], data_4s['preds'][idx])
plt.plot(fpr_4s, tpr_4s, label='4-stream (AUC=0.995)', linewidth=2, color='#66b3ff')

try:
    data_3s = np.load('results/raw_predictions_aiforge.npz')
    sample_size = min(2_000_000, len(data_3s['preds']))
    idx = rng.choice(len(data_3s['preds']), size=sample_size, replace=False)
    fpr_3s, tpr_3s, _ = roc_curve(data_3s['labels'][idx], data_3s['preds'][idx])
    plt.plot(fpr_3s, tpr_3s, label='3-stream (AUC=0.738)', linewidth=2, color='#ff9999')
except FileNotFoundError:
    print("Note: 3-stream AIForge-Doc raw predictions not found, plotting 4-stream only.")

plt.plot([0, 1], [0, 1], 'k--', label='Random chance', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('The Key Result: Noise-Fingerprint Stream Fixes AI-Inpainting Blindness')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('results/plot_aiforge_before_after.png', dpi=200)
plt.close()
print("Saved plot_aiforge_before_after.png")

print("\nAll plots saved to results/")