"""
Create a visual flow diagram showing the improved prediction logic
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
NAVY = "#1C2B3A"
GOLD = "#C8A96E"
GREEN = "#2D6A4F"
RED = "#8B2525"
AMBER = "#9B6A1A"

# Title
ax.text(5, 11.5, 'Improved Churn Prediction Flow', 
        ha='center', fontsize=18, fontweight='bold', color=NAVY)

# Step 1: Input
box1 = FancyBboxPatch((0.5, 9.5), 9, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor=NAVY, facecolor='#F0ECE4', linewidth=2)
ax.add_patch(box1)
ax.text(5, 10.3, 'STEP 1: Student Input', ha='center', fontsize=12, fontweight='bold', color=NAVY)
ax.text(5, 9.8, '32 Admission Features + College + Semester', ha='center', fontsize=10, color='#5C6B7A')

# Arrow
ax.arrow(5, 9.4, 0, -0.5, head_width=0.3, head_length=0.2, fc=GOLD, ec=GOLD, linewidth=2)

# Step 2: College Check
box2 = FancyBboxPatch((0.5, 7.5), 4, 1.5, boxstyle="round,pad=0.1", 
                       edgecolor=NAVY, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(box2)
ax.text(2.5, 8.6, 'BPCCS', ha='center', fontsize=11, fontweight='bold', color=NAVY)
ax.text(2.5, 8.2, 'Semesters: 1-6', ha='center', fontsize=9, color='#5C6B7A')
ax.text(2.5, 7.9, '588 students', ha='center', fontsize=9, color='#5C6B7A')

box3 = FancyBboxPatch((5.5, 7.5), 4, 1.5, boxstyle="round,pad=0.1", 
                       edgecolor=GOLD, facecolor='#FFF8E1', linewidth=2)
ax.add_patch(box3)
ax.text(7.5, 8.6, 'SVICS-G', ha='center', fontsize=11, fontweight='bold', color=GOLD)
ax.text(7.5, 8.2, 'Semesters: 1-3 only', ha='center', fontsize=9, color='#5C6B7A')
ax.text(7.5, 7.9, '254 students', ha='center', fontsize=9, color='#5C6B7A')

# Arrows to admission model
ax.arrow(2.5, 7.4, 0, -0.5, head_width=0.3, head_length=0.2, fc=NAVY, ec=NAVY, linewidth=2)
ax.arrow(7.5, 7.4, 0, -0.5, head_width=0.3, head_length=0.2, fc=GOLD, ec=GOLD, linewidth=2)

# Step 3: Admission Model
box4 = FancyBboxPatch((1.5, 5.5), 7, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor=NAVY, facecolor='#E3F2FD', linewidth=3)
ax.add_patch(box4)
ax.text(5, 6.3, 'Random Forest Model (32 Features)', ha='center', fontsize=12, fontweight='bold', color=NAVY)
ax.text(5, 5.8, 'Base Admission Risk Probability', ha='center', fontsize=10, color='#5C6B7A')

# Arrow
ax.arrow(5, 5.4, 0, -0.5, head_width=0.3, head_length=0.2, fc=GOLD, ec=GOLD, linewidth=2)

# Step 4: Semester Check
box5 = FancyBboxPatch((1.5, 3.5), 7, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor=AMBER, facecolor='#FFF8E1', linewidth=2)
ax.add_patch(box5)
ax.text(5, 4.3, 'Semester-Based Adjustment', ha='center', fontsize=12, fontweight='bold', color=AMBER)
ax.text(5, 3.8, 'Apply Adaptive Weighting', ha='center', fontsize=10, color='#5C6B7A')

# Semester branches
ax.arrow(3, 3.4, -1.2, -0.8, head_width=0.2, head_length=0.15, fc=RED, ec=RED, linewidth=2)
ax.arrow(4, 3.4, -0.5, -0.8, head_width=0.2, head_length=0.15, fc=AMBER, ec=AMBER, linewidth=2)
ax.arrow(5, 3.4, 0, -0.8, head_width=0.2, head_length=0.15, fc=GREEN, ec=GREEN, linewidth=2)
ax.arrow(6.5, 3.4, 0.8, -0.8, head_width=0.2, head_length=0.15, fc=NAVY, ec=NAVY, linewidth=2)

# Semester boxes
box_s1 = FancyBboxPatch((0.8, 1.3), 1.8, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor=RED, facecolor='#FFEBEE', linewidth=2)
ax.add_patch(box_s1)
ax.text(1.7, 1.9, 'Sem 1', ha='center', fontsize=10, fontweight='bold', color=RED)
ax.text(1.7, 1.5, '20% weight', ha='center', fontsize=8, color=RED)

box_s2 = FancyBboxPatch((2.8, 1.3), 1.8, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor=AMBER, facecolor='#FFF8E1', linewidth=2)
ax.add_patch(box_s2)
ax.text(3.7, 1.9, 'Sem 2', ha='center', fontsize=10, fontweight='bold', color=AMBER)
ax.text(3.7, 1.5, '15% weight', ha='center', fontsize=8, color=AMBER)

box_s3 = FancyBboxPatch((4.8, 1.3), 1.8, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor=GREEN, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(box_s3)
ax.text(5.7, 1.9, 'Sem 3', ha='center', fontsize=10, fontweight='bold', color=GREEN)
ax.text(5.7, 1.5, '5% weight', ha='center', fontsize=8, color=GREEN)

box_s4 = FancyBboxPatch((6.8, 1.3), 2.2, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor=NAVY, facecolor='#E3F2FD', linewidth=2)
ax.add_patch(box_s4)
ax.text(7.9, 1.9, 'Sem 4-6', ha='center', fontsize=10, fontweight='bold', color=NAVY)
ax.text(7.9, 1.5, '0% weight', ha='center', fontsize=8, color=NAVY)

# Arrows to final
ax.arrow(1.7, 1.2, 0.8, -0.4, head_width=0.15, head_length=0.1, fc=GOLD, ec=GOLD, linewidth=1.5)
ax.arrow(3.7, 1.2, 0.3, -0.4, head_width=0.15, head_length=0.1, fc=GOLD, ec=GOLD, linewidth=1.5)
ax.arrow(5.7, 1.2, -0.2, -0.4, head_width=0.15, head_length=0.1, fc=GOLD, ec=GOLD, linewidth=1.5)
ax.arrow(7.9, 1.2, -1.4, -0.4, head_width=0.15, head_length=0.1, fc=GOLD, ec=GOLD, linewidth=1.5)

# Final prediction
box_final = FancyBboxPatch((2, 0.1), 6, 0.6, boxstyle="round,pad=0.1", 
                            edgecolor=GOLD, facecolor=GOLD, linewidth=3)
ax.add_patch(box_final)
ax.text(5, 0.4, 'FINAL CHURN PROBABILITY', ha='center', fontsize=12, fontweight='bold', color='white')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#FFEBEE', edgecolor=RED, label='Critical Risk (Sem 1: 100% churn)'),
    mpatches.Patch(facecolor='#FFF8E1', edgecolor=AMBER, label='High Risk (Sem 2: 75-93% churn)'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor=GREEN, label='Low Risk (Sem 3: <3% churn)'),
    mpatches.Patch(facecolor='#E3F2FD', edgecolor=NAVY, label='Stable (Sem 4-6: 0% churn)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
          title='Risk Levels', title_fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig('prediction_flow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Flow diagram saved as 'prediction_flow_diagram.png'")

# Create second diagram: College comparison
fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

# BPCCS
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, 10)
ax_left.axis('off')
ax_left.text(5, 9.5, 'BPCCS College', ha='center', fontsize=16, fontweight='bold', color=NAVY)
ax_left.text(5, 9, '588 Students | Rs 18,000 Fees', ha='center', fontsize=10, color='#5C6B7A')

y_pos = 8
for sem, count, churned, rate in [
    (1, 43, 43, 100), (2, 32, 24, 75), (3, 28, 7, 25),
    (4, 5, 0, 0), (5, 13, 0, 0), (6, 467, 0, 0)
]:
    color = RED if rate == 100 else AMBER if rate >= 50 else GREEN if rate > 0 else NAVY
    box = FancyBboxPatch((1, y_pos-0.6), 8, 0.5, boxstyle="round,pad=0.05",
                          edgecolor=color, facecolor=f'{color}22', linewidth=2)
    ax_left.add_patch(box)
    ax_left.text(2, y_pos-0.35, f'Sem {sem}', fontsize=10, fontweight='bold', color=color)
    ax_left.text(4.5, y_pos-0.35, f'{count} students', fontsize=9, color='#5C6B7A')
    ax_left.text(7, y_pos-0.35, f'{churned} churned ({rate}%)', fontsize=9, 
                 fontweight='bold', color=color)
    y_pos -= 1

# SVICS-G
ax_right.set_xlim(0, 10)
ax_right.set_ylim(0, 10)
ax_right.axis('off')
ax_right.text(5, 9.5, 'SVICS-G College', ha='center', fontsize=16, fontweight='bold', color=GOLD)
ax_right.text(5, 9, '254 Students | Rs 27,000 Fees', ha='center', fontsize=10, color='#5C6B7A')

y_pos = 8
for sem, count, churned, rate in [
    (1, 18, 18, 100), (2, 15, 14, 93.3), (3, 221, 1, 0.45)
]:
    color = RED if rate == 100 else AMBER if rate >= 50 else GREEN
    box = FancyBboxPatch((1, y_pos-0.6), 8, 0.5, boxstyle="round,pad=0.05",
                          edgecolor=color, facecolor=f'{color}22', linewidth=2)
    ax_right.add_patch(box)
    ax_right.text(2, y_pos-0.35, f'Sem {sem}', fontsize=10, fontweight='bold', color=color)
    ax_right.text(4.5, y_pos-0.35, f'{count} students', fontsize=9, color='#5C6B7A')
    ax_right.text(7, y_pos-0.35, f'{churned} churned ({rate:.1f}%)', fontsize=9, 
                 fontweight='bold', color=color)
    y_pos -= 1

# Add note
box_note = FancyBboxPatch((1, y_pos-0.8), 8, 0.6, boxstyle="round,pad=0.05",
                           edgecolor='#999', facecolor='#F5F5F5', linewidth=1, linestyle='--')
ax_right.add_patch(box_note)
ax_right.text(5, y_pos-0.5, 'Sem 4-6: No data available', ha='center', 
              fontsize=9, style='italic', color='#666')

plt.tight_layout()
plt.savefig('college_comparison_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ College comparison saved as 'college_comparison_diagram.png'")

print("\n" + "="*60)
print("VISUAL DIAGRAMS CREATED")
print("="*60)
print("\n1. prediction_flow_diagram.png")
print("   - Shows complete prediction flow")
print("   - Illustrates adaptive weighting logic")
print("\n2. college_comparison_diagram.png")
print("   - Side-by-side college comparison")
print("   - Shows semester distribution and churn rates")
