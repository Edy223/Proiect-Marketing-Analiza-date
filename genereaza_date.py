"""Generează 3 CSV-uri de test: produse, campanii, vânzări zilnice."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------- PRODUSE ----------------
catalog = {
    "Electronice": [("Căști Bluetooth", 249), ("Smartwatch Fit", 599), ("Boxă portabilă", 199), ("Powerbank 20000mAh", 129), ("Cameră Web HD", 159)],
    "Modă": [("Geacă de iarnă", 399), ("Sneakers Urban", 329), ("Rucsac City", 179), ("Ochelari de soare", 149), ("Ceas clasic", 449)],
    "Casă & Grădină": [("Aspirator robot", 899), ("Set oale inox", 349), ("Lampă LED smart", 119), ("Difuzor aromaterapie", 89), ("Grătar electric", 279)],
    "Frumusețe": [("Set ser facial", 189), ("Parfum Noir 50ml", 259), ("Uscător de păr Pro", 229), ("Trusă machiaj", 139)],
    "Sport": [("Saltea yoga", 99), ("Gantere reglabile", 299), ("Bicicletă fitness", 1299), ("Sticlă termică", 69), ("Bandă elastică set", 49)],
}
rows = []
i = 1
for cat, items in catalog.items():
    for nume, pret in items:
        rows.append({"produs_id": f"P{i:02d}", "produs": nume, "categorie": cat,
                     "pret": pret, "cost": round(pret * rng.uniform(0.40, 0.70), 2)})
        i += 1
produse = pd.DataFrame(rows)

# ---------------- CAMPANII ----------------
canale = {  # cpm (RON/1000 afișări), ctr, cvr medii
    "Facebook Ads": (30, 0.012, 0.014),
    "Google Ads": (45, 0.030, 0.030),
    "Instagram": (34, 0.010, 0.010),
    "TikTok": (20, 0.014, 0.006),
    "Email": (22, 0.030, 0.015),
    "Influenceri": (60, 0.018, 0.020),
}
camp = []
n = 130
for k in range(1, n + 1):
    p = produse.sample(1, random_state=int(rng.integers(1e6))).iloc[0]
    canal = rng.choice(list(canale))
    cpm, ctr, cvr = canale[canal]
    start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 250)))
    durata = int(rng.integers(7, 31))
    buget = float(np.round(np.clip(rng.lognormal(7.9, 0.7), 400, 20000), 2))
    afisari = int(buget / cpm * 1000 * rng.uniform(0.8, 1.2))
    clickuri = int(afisari * ctr * rng.uniform(0.6, 1.4))
    conversii = int(clickuri * cvr * rng.uniform(0.5, 1.5))
    venituri = float(np.round(conversii * p.pret * rng.uniform(0.9, 1.1), 2))
    camp.append({"campanie_id": f"C{k:03d}", "produs_id": p.produs_id, "canal": canal,
                 "data_start": start.date(), "data_sfarsit": (start + pd.Timedelta(days=durata - 1)).date(),
                 "buget": buget, "afisari": afisari, "click_uri": clickuri,
                 "conversii": conversii, "venituri": venituri})
campanii = pd.DataFrame(camp)

# "murdărie" realistă: valori lipsă, un outlier, duplicate
campanii.loc[rng.choice(n, 5, replace=False), "click_uri"] = np.nan
campanii.loc[rng.choice(n, 4, replace=False), "conversii"] = np.nan
campanii.loc[7, "buget"] = 95000.0          # outlier de buget
campanii = pd.concat([campanii, campanii.sample(2, random_state=1)], ignore_index=True)

# ---------------- VÂNZĂRI ZILNICE ----------------
zile = pd.date_range("2026-01-01", "2026-09-30")
promo = set()
for _, c in pd.DataFrame(camp).iterrows():
    for d in pd.date_range(c["data_start"], c["data_sfarsit"]):
        promo.add((c["produs_id"], d))
baza = {pid: rng.uniform(1.5, 9) * (1.3 if pr < 150 else 0.6 if pr > 800 else 1.0)
        for pid, pr in zip(produse.produs_id, produse.pret)}
v = []
for pid, pret in zip(produse.produs_id, produse.pret):
    for d in zile:
        sezon = 1.25 if d.dayofweek >= 5 else 1.0
        sezon *= 1 + 0.15 * np.sin(2 * np.pi * d.dayofyear / 365)
        p_on = int((pid, d) in promo)
        lam = baza[pid] * sezon * (rng.uniform(1.4, 1.9) if p_on else 1.0)
        unit = int(rng.poisson(lam))
        v.append({"data": d.date(), "produs_id": pid, "unitati_vandute": unit,
                  "venit": round(unit * pret * rng.uniform(0.85, 1.0), 2), "promovat": p_on})
vanzari = pd.DataFrame(v)

produse.to_csv("date_test/produse.csv", index=False)
campanii.to_csv("date_test/campanii.csv", index=False)
vanzari.to_csv("date_test/vanzari_zilnice.csv", index=False)
print(produse.shape, campanii.shape, vanzari.shape)
