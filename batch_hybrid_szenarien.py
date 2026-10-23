"""Batch-Lauf aller Hybridpark-Szenarien (2.b Hybrider Netzzugang).

Szenariomatrix:
  - Datenordner data/2.b_Hybrider_Netzzugang_<Jahr>[_Speicher]_<Zins>%   (Jahr, Speicher, Zinssatz)
  - PV-Variante  S / OW / Track   -> LEGO.py --pv-invest (EnableInvest nur für diesen PV-Typ)
  - Regime       Spitze / Operativ -> LEGO.py --curtailment-pv / --curtailment-wind

Jeder Lauf ist ein eigener LEGO.py-Prozess (Abstürze/Speicherprobleme betreffen nur diesen Lauf). Das Ergebnis wird
lokal in TEMP_ORDNER geschrieben und erst nach erfolgreichem Lauf nach ZIEL_ORDNER/<Jahr>/ verschoben, Dateiname:
  <Jahr>_<Variante>_<Regime>[_Speicher]_<Zins>%.sqlite   (z. B. 2025_S_Operativ_Speicher_3%.sqlite)

Bereits vorhandene Ergebnisse werden übersprungen (Wiederaufnahme nach Abbruch), außer mit --neu.

Aufruf (im Ordner LEGO-Pyomo, in der LEGO-Umgebung):
  python batch_hybrid_szenarien.py --dry-run                                   # nur Liste der Läufe
  python batch_hybrid_szenarien.py --jahr 2025 --variante S --regime Spitze --zins 3 --speicher nein   # Testlauf
  python batch_hybrid_szenarien.py                                             # alle 48 Läufe
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# ===================== PARAMETER =====================
LEGO_ORDNER = Path(__file__).resolve().parent
DATEN_ORDNER = LEGO_ORDNER / "data"
ORDNER_MUSTER = re.compile(r"^2\.b_Hybrider_Netzzugang_(?P<jahr>\d{4})(?P<speicher>_Speicher)?_(?P<zins>\d+(?:[.,]\d+)?)%$")
ZIEL_ORDNER = Path(r"R:\01_Modelle\2025-02-24_DyLeR-20\Scenarios_2.a\Hybrider Netzzugang\Ergebnisse_Zeitkorrektur")
TEMP_ORDNER = Path(r"C:\Temp\LEGO_Batch")  # lokal schreiben, danach verschieben (SQLite auf Netzlaufwerk ist langsam)

PV_VARIANTEN = {"S": ["PV_S_35"], "OW": ["PV_OW_15"], "Track": ["PV_Track"]}
REGIME = {  # (pCurtailmentPV, pCurtailmentWind)
    "Spitze": (0.4, 0.15),  # nur Spitzenkappung erlaubt
    "Operativ": (1.0, 0.0),  # PV voll abregelbar, Wind nicht
}
# =====================================================


@dataclass
class Lauf:
    jahr: int
    speicher: bool
    zins: str
    variante: str
    regime: str
    datenordner: Path

    @property
    def name(self) -> str:
        return f"{self.jahr}_{self.variante}_{self.regime}{'_Speicher' if self.speicher else ''}_{self.zins}%"

    @property
    def ziel(self) -> Path:
        return ZIEL_ORDNER / str(self.jahr) / f"{self.name}.sqlite"


def finde_laeufe(args) -> list[Lauf]:
    laeufe = []
    for ordner in sorted(DATEN_ORDNER.iterdir()):
        treffer = ORDNER_MUSTER.match(ordner.name)
        if not ordner.is_dir() or not treffer:
            continue
        for variante in PV_VARIANTEN:
            for regime in REGIME:
                laeufe.append(Lauf(int(treffer["jahr"]), bool(treffer["speicher"]), treffer["zins"].replace(",", "."),
                                   variante, regime, ordner))
    # Filter
    laeufe = [l for l in laeufe
              if (not args.jahr or l.jahr in args.jahr)
              and (not args.variante or l.variante in args.variante)
              and (not args.regime or l.regime in args.regime)
              and (not args.zins or l.zins in args.zins)
              and (args.speicher is None or l.speicher == (args.speicher == "ja"))]
    return sorted(laeufe, key=lambda l: (l.jahr, l.speicher, float(l.zins), list(PV_VARIANTEN).index(l.variante),
                                         list(REGIME).index(l.regime)))


def zielfunktion(pfad: Path) -> float | None:
    try:
        with sqlite3.connect(f"file:{quote(pfad.absolute().as_posix())}?mode=ro", uri=True) as conn:
            return float(conn.execute('SELECT "values" FROM objective').fetchone()[0])
    except Exception:
        return None


def protokolliere(datei: Path, zeile: dict):
    neu = not datei.exists()
    with datei.open("a", newline="", encoding="utf-8-sig") as f:
        schreiber = csv.DictWriter(f, fieldnames=list(zeile), delimiter=";")
        if neu:
            schreiber.writeheader()
        schreiber.writerow(zeile)


def fuehre_aus(lauf: Lauf, log_ordner: Path) -> tuple[str, str]:
    """Startet LEGO.py für einen Lauf; gibt (Status, Meldung) zurück."""
    TEMP_ORDNER.mkdir(parents=True, exist_ok=True)
    temp = TEMP_ORDNER / f"{lauf.name}.sqlite"
    if temp.exists():
        temp.unlink()
    curt_pv, curt_wind = REGIME[lauf.regime]
    befehl = [sys.executable, str(LEGO_ORDNER / "LEGO.py"), str(lauf.datenordner),
              "--pv-invest", *PV_VARIANTEN[lauf.variante],
              "--curtailment-pv", str(curt_pv), "--curtailment-wind", str(curt_wind),
              "--output", str(temp), "--no-mps"]
    log = log_ordner / f"{lauf.name}.log"
    with log.open("w", encoding="utf-8") as f:
        f.write(" ".join(befehl) + "\n\n")
        f.flush()
        ergebnis = subprocess.run(befehl, cwd=LEGO_ORDNER, stdout=f, stderr=subprocess.STDOUT)
    if ergebnis.returncode != 0:
        return "Fehler", f"LEGO.py Exitcode {ergebnis.returncode} (siehe {log.name})"
    if not temp.exists():
        return "Fehler", f"Keine Ergebnisdatei geschrieben (siehe {log.name})"
    lauf.ziel.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp), str(lauf.ziel))  # erst nach erfolgreichem Lauf in den Ergebnisordner
    return "ok", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jahr", type=int, nargs="+", help="nur diese Jahre, z. B. 2025")
    parser.add_argument("--variante", nargs="+", choices=list(PV_VARIANTEN), help="nur diese PV-Varianten")
    parser.add_argument("--regime", nargs="+", choices=list(REGIME), help="nur diese Regime")
    parser.add_argument("--zins", nargs="+", help="nur diese Zinssätze, z. B. 3 5")
    parser.add_argument("--speicher", choices=["ja", "nein"], help="nur Läufe mit / ohne Speicheroption")
    parser.add_argument("--neu", action="store_true", help="vorhandene Ergebnisse neu rechnen statt überspringen")
    parser.add_argument("--dry-run", action="store_true", help="nur die geplanten Läufe auflisten")
    args = parser.parse_args()

    laeufe = finde_laeufe(args)
    if not laeufe:
        raise SystemExit("Keine passenden Läufe gefunden.")
    print(f"{len(laeufe)} Läufe, Ziel: {ZIEL_ORDNER}")
    for lauf in laeufe:
        status = "vorhanden" if lauf.ziel.exists() else "offen"
        print(f"  {lauf.name:<34} {lauf.datenordner.name:<42} {status}")
    if args.dry_run:
        return

    log_ordner = ZIEL_ORDNER / "logs"
    log_ordner.mkdir(parents=True, exist_ok=True)
    protokoll = ZIEL_ORDNER / "laufprotokoll.csv"
    start_gesamt = time.time()
    zaehler = {"ok": 0, "Fehler": 0, "übersprungen": 0}

    for i, lauf in enumerate(laeufe, start=1):
        if lauf.ziel.exists() and not args.neu:
            print(f"[{i}/{len(laeufe)}] {lauf.name}: vorhanden - übersprungen")
            zaehler["übersprungen"] += 1
            continue
        print(f"[{i}/{len(laeufe)}] {lauf.name}: läuft ({datetime.now():%H:%M}) ...", flush=True)
        start = time.time()
        try:
            status, meldung = fuehre_aus(lauf, log_ordner)
        except Exception as fehler:  # z. B. Verschieben fehlgeschlagen - Batch läuft weiter
            status, meldung = "Fehler", repr(fehler)
        dauer = (time.time() - start) / 60
        zaehler[status] += 1
        print(f"      -> {status} nach {dauer:.1f} min {meldung}", flush=True)
        protokolliere(protokoll, {
            "Zeitpunkt": f"{datetime.now():%Y-%m-%d %H:%M}", "Szenario": lauf.name, "Datenordner": lauf.datenordner.name,
            "Status": status, "Dauer [min]": f"{dauer:.1f}".replace(".", ","),
            "Zielfunktion": f"{zielfunktion(lauf.ziel):.2f}".replace(".", ",") if status == "ok" else "",
            "Meldung": meldung,
        })

    print(f"Fertig nach {(time.time() - start_gesamt) / 3600:.1f} h: {zaehler['ok']} ok, {zaehler['Fehler']} Fehler, "
          f"{zaehler['übersprungen']} übersprungen. Protokoll: {protokoll}")


if __name__ == "__main__":
    main()
