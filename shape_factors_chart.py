import matplotlib.pyplot as plt
import numpy as np

hours = list(range(1, 25))
shape_factors = [
    0.8440, 0.8212, 0.8060, 0.7964, 0.7995, 0.8384,
    0.9146, 0.9855, 1.0000, 0.9934, 0.9855, 0.9681,
    0.9496, 0.9569, 0.9527, 0.9338, 0.9177, 0.9000,
    0.9456, 0.9705, 0.9634, 0.9462, 0.9030, 0.8600
]

colors = ['#2E74B5' if sf == 1.0 else '#A8C4E0' for sf in shape_factors]

fig, ax = plt.subplots(figsize=(12, 5))

bars = ax.bar(hours, shape_factors, color=colors, edgecolor='white', linewidth=0.5)

# Annotate peak bar
peak_hour = 9
ax.annotate(
    'Peak (sh = 1.00)',
    xy=(peak_hour, 1.0),
    xytext=(peak_hour + 2, 1.015),
    arrowprops=dict(arrowstyle='->', color='#1F3864', lw=1.2),
    fontsize=9, color='#1F3864'
)

ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('Shape Factor  $s_h = L_h \\ / \\ L^*$', fontsize=11)
ax.set_title(r'Forecast Hourly Shape Factors for $d^*$ = March 13, 2026  ($L^*$ = 28,054 MW)',
             fontsize=12, pad=12)

ax.set_xticks(hours)
ax.set_xlim(0.3, 24.7)
ax.set_ylim(0.75, 1.06)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.axhline(y=1.0, color='#1F3864', linewidth=0.8, linestyle='--', alpha=0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('shape_factors_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to shape_factors_chart.png")
