"""
make_eda_figs.py — generate the three EDA charts with Georgian labels.

Run this ON YOUR WINDOWS MACHINE (it uses a Georgian-capable font that
ships with Windows, e.g. Sylfaen).  Requires:  pip install matplotlib

    python scripts/make_eda_figs.py

Outputs PDF (use in LaTeX) + PNG to  eval_out/eda_figs/.
"""
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

# --- pick a Georgian-capable font available on the machine ---
for cand in ['Sylfaen', 'Noto Sans Georgian', 'BPG Glaho', 'DejaVu Sans']:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        matplotlib.rcParams['font.family'] = cand
        print(f'using font: {cand}')
        break
    except Exception:
        continue

plt.rcParams.update({'font.size': 11, 'figure.dpi': 130, 'axes.unicode_minus': False})

OUT = os.path.join(os.path.dirname(__file__), '..', 'eval_out', 'eda_figs')
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'))

# ---- 1. Class frequency (DeepScores val), log scale ----
classes = [
    ('noteheadBlack', 33827), ('beam', 9026), ('noteheadHalf', 3119),
    ('augmentationDot', 2624), ('restWhole', 1670), ('keySharp', 1539),
    ('slur', 1280), ('flag8thDown', 1148), ('restQuarter', 1093),
    ('noteheadWhole', 933), ('accidentalNatural', 921),
    ('ornamentTrill', 27), ('flag64thUp', 20), ('flag32ndUp', 3),
    ('accidentalDoubleFlat', 1),
]
names = [c[0] for c in classes][::-1]
vals = [c[1] for c in classes][::-1]
fig, ax = plt.subplots(figsize=(8, 6.5))
bars = ax.barh(names, vals, color='#3b6ea5')
ax.set_xscale('log')
ax.set_xlabel('ინსტანციათა რაოდენობა (ლოგარითმული მასშტაბი)')
ax.set_title('სიმბოლოთა კლასების სიხშირე (DeepScores V2)')
for b, v in zip(bars, vals):
    ax.text(v * 1.15, b.get_y() + b.get_height() / 2, str(v), va='center', fontsize=8)
ax.set_xlim(0.7, 80000)
fig.tight_layout()
save(fig, 'class_frequency')

# ---- 2. Voice-count distribution, log y ----
vc = [(1, 11), (2, 42), (3, 2562), (4, 37), (5, 14), (6, 17), (8, 5), (10, 1)]
labels = [str(k) for k, _ in vc]
counts = [v for _, v in vc]
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(labels, counts, color='#3b6ea5')
ax.set_yscale('log')
ax.set_xlabel('ხმათა რაოდენობა')
ax.set_ylabel('ნაწარმოები (ლოგ. მასშტაბი)')
ax.set_title('ხმათა რაოდენობის განაწილება (2 689 ნაწარმოები)')
for b, v in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.15, str(v), ha='center', fontsize=9)
ax.set_ylim(0.7, 5000)
fig.tight_layout()
save(fig, 'voice_distribution')

# ---- 3. Rhythm comparison liturgical vs folk ----
cats = ['მერვედი', 'მეოთხედი', 'ნახევარი', 'მეთექვსმეტ.', 'წერტილიანი']
lit = [40.1, 38.2, 13.6, 3.6, 2.7]
folk = [53.8, 14.0, 5.4, 15.4, 10.2]
x = np.arange(len(cats))
w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 4.2))
b1 = ax.bar(x - w / 2, lit, w, label='საეკლესიო (Antologia)', color='#3b6ea5')
b2 = ax.bar(x + w / 2, folk, w, label='ხალხური (აბესაძეები)', color='#d08a3e')
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel('ნოტების %')
ax.set_title('ხანგრძლივობის პროფილი: საეკლესიო vs. ხალხური')
ax.legend()
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f'{b.get_height():.1f}', ha='center', fontsize=8)
fig.tight_layout()
save(fig, 'rhythm_comparison')

print('done ->', OUT)
