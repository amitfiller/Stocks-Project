# =============================================================================
# 🔍 STAGE 4: MODEL EXPLAINABILITY AND FAIRNESS ANALYSIS
# =============================================================================

print("="*80)
print("🔍 STAGE 4: MODEL EXPLAINABILITY AND FAIRNESS ANALYSIS")
print("="*80)

# Display classification results
print("\n📊 MODEL PERFORMANCE OVERVIEW")
print("-" * 40)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate per-class metrics for bias analysis
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
classes = model.classes_

print("\n🎯 PER-CLASS PERFORMANCE ANALYSIS")
print("-" * 40)
for i, class_name in enumerate(classes):
    print(f"{class_name:12}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")

# Identify potential bias issues
print("\n⚖️ BIAS ANALYSIS")
print("-" * 40)

# Check for class imbalance
print("Class Distribution in Test Set:")
class_counts = pd.Series(y_test).value_counts().sort_index()
for class_name, count in class_counts.items():
    percentage = (count / len(y_test)) * 100
    print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

# Identify performance disparities
min_recall = min(recall)
max_recall = max(recall)
recall_disparity = max_recall - min_recall

print(f"\n🚨 FAIRNESS METRICS:")
print(f"  Recall Disparity: {recall_disparity:.3f} (Max: {max_recall:.3f}, Min: {min_recall:.3f})")
print(f"  Worst Performing Class: {classes[np.argmin(recall)]} (Recall: {min_recall:.3f})")

if recall_disparity > 0.1:
    print("  ⚠️  WARNING: Significant recall disparity detected (>0.1)")
    print("     This suggests potential systematic bias in the model.")

# SHAP Analysis Summary
print("\n🧠 SHAP FEATURE IMPORTANCE SUMMARY")
print("-" * 40)
print("Top 5 Most Important Features (based on SHAP analysis):")

# Get feature importance from SHAP values (assuming shap_values_for_class exists)
if 'shap_values_for_class' in locals():
    feature_importance = np.abs(shap_values_for_class).mean(0)
    feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(X_test.shape[1])]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.head().iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
else:
    print("  (SHAP analysis not available - run SHAP code first)")

# Bias Mitigation Recommendations
print("\n🛡️ BIAS MITIGATION RECOMMENDATIONS")
print("-" * 40)
print("1. Data-Level Solutions:")
print("   - Apply SMOTE or class balancing techniques")
print("   - Collect sector-specific features")
print("   - Implement stratified sampling")

print("\n2. Model-Level Adjustments:")
print("   - Use class weights in model training")
print("   - Implement threshold optimization per class")
print("   - Consider ensemble methods")

print("\n3. Monitoring & Validation:")
print("   - Set minimum performance thresholds per class")
print("   - Regular fairness audits")
print("   - Feature importance monitoring")

print("\n" + "="*80)
print("✅ STAGE 4 ANALYSIS COMPLETE")
print("="*80)