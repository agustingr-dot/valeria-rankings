#!/usr/bin/env python3
"""ValerIA: rankings mensuales de momentum, metodologia Portfolio Visualizer.

    score(t) = [cierre(t-1) / cierre(t-12) - 1] / volatilidad 3m que acaba en el cierre de t-1

Lo ejecuta GitHub Actions tras cada cierre de mes. Descarga cotizaciones de
Yahoo Finance, las convierte a EUR y escribe rankings.json, que una tarea
programada de Claude copia a la base de datos de la app "Ranking ValerIA".
"""
import json, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

DIAS_ANIO, MIN_OBS_VOL, DECISIONES_GUARDADAS = 252, 40, 37   # 3 años + la próxima


def mes_completo(px):
    """El ultimo mes solo cuenta si su ultimo dato esta en los 4 dias finales habiles."""
    ult = px.index.max()
    fin = (ult + pd.offsets.MonthEnd(0)).normalize()
    while fin.weekday() >= 5:
        fin -= pd.Timedelta(days=1)
    return (fin - ult.normalize()).days <= 4


def calcular(px, lookback=12, skip=1, vol_meses=3, incluir_incompleto=False):
    """px: precios diarios ya en la moneda base, una columna por activo."""
    px = px.sort_index().dropna(how="all")
    meses = px.index.to_period("M")
    mensual = px.groupby(meses).last()          # ultimo cierre de cada activo
    cierre = px.index.to_series().groupby(meses).max()
    span = lookback - skip
    # el mes en curso solo entra en el avance provisional, nunca en las decisiones
    tope = len(mensual) - 1 if (mes_completo(px) or incluir_incompleto) else len(mensual) - 2
    out = []
    for k in range(span, tope + 1):
        p = mensual.index[k]
        f = cierre[p]
        mom = mensual.iloc[k] / mensual.iloc[k - span] - 1
        desde = f - pd.DateOffset(months=vol_meses)   # recorta a fin de mes como EDATE
        vol = {}
        for c in px.columns:
            r = px[c].loc[desde:f].dropna().pct_change().dropna()
            vol[c] = r.std(ddof=1) * np.sqrt(DIAS_ANIO) if len(r) >= MIN_OBS_VOL else np.nan
        vol = pd.Series(vol)
        score = (mom / vol).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        if score.empty:
            continue
        out.append({
            "mesSenal": str(p),
            "fechaSenal": f.strftime("%Y-%m-%d"),
            "mesDecision": str(p + skip),
            "mesCartera": str(p + skip + 1),
            "ranking": [{"senal": c, "pos": i + 1, "score": round(float(s), 6),
                         "mom": round(float(mom[c]), 6), "vol": round(float(vol[c]), 6)}
                        for i, (c, s) in enumerate(score.items())],
        })
    return out


def descargar(tickers, desde):
    import yfinance as yf
    d = yf.download(tickers, start=desde, auto_adjust=True, progress=False, group_by="column")
    px = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]].rename(columns={"Close": tickers[0]})
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px.reindex(columns=tickers).dropna(how="all")


def divisa_de(ticker):
    """Divisa real del ticker segun Yahoo; 'GBp' son peniques de Londres."""
    import yfinance as yf
    try:
        return str(yf.Ticker(ticker).fast_info["currency"])
    except Exception:
        return "USD"


def serie_fx(origen, base, desde):
    """Igual que el motor del Mac: par directo USDEUR=X y, si falla, 1/EURUSD=X."""
    par = f"{origen}{base}=X"
    try:
        s = descargar([par], desde).iloc[:, 0]
    except Exception:
        s = pd.Series(dtype=float)
    if s.notna().sum() < 50:
        s = 1.0 / descargar([f"{base}{origen}=X"], desde).iloc[:, 0]
    return s.dropna()


def a_moneda(px, base, desde):
    """Convierte cada columna a la moneda base segun la divisa real de su ticker."""
    if base == "NONE":
        return px, []
    notas, cache = [], {}
    for c in px.columns:
        cur = divisa_de(c)
        if cur == "GBp":
            px[c] = px[c] / 100.0
            cur = "GBP"
        cur = cur.upper()
        if cur == base:
            continue
        if cur not in cache:
            cache[cur] = serie_fx(cur, base, desde)
        px[c] = px[c] * cache[cur].reindex(px.index).ffill().bfill()
    if cache:
        notas.append(f"Convertido a {base} desde " + ", ".join(sorted(cache)))
    return px, notas


def main(cfg_path="universos.json", out_path="rankings.json"):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    base = cfg.get("moneda", "EUR")
    par = cfg.get("parametros", {})
    desde = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    salida = {"generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "moneda": base, "metodologia": "Portfolio Visualizer: rentabilidad t-12 a t-1, vol 3m hasta t-1",
              "estrategias": {}}
    for clave, est in cfg["estrategias"].items():
        activos = est["activos"]
        tickers = [a["senal"] for a in activos]
        px = descargar(tickers, desde)
        avisos = [f"Sin datos de {t}" for t in tickers if t not in px or px[t].notna().sum() < 200]
        px = px[[t for t in tickers if t in px and px[t].notna().sum() >= 200]]
        px, notas = a_moneda(px, base, desde)
        args = (par.get("lookback", 12), par.get("skip", 1), par.get("vol_meses", 3))
        dec = calcular(px, *args)
        prov = None
        if not mes_completo(px):                       # avance con el mes en curso
            todas = calcular(px, *args, incluir_incompleto=True)
            if todas and todas[-1]["mesSenal"] == str(px.index.max().to_period("M")):
                prov = todas[-1]
        meta = {a["senal"]: a for a in activos}
        for d in dec + ([prov] if prov else []):
            for r in d["ranking"]:
                r["nombre"] = meta[r["senal"]].get("nombre", r["senal"])
                if meta[r["senal"]].get("compra"):
                    r["compra"] = meta[r["senal"]]["compra"]
        salida["estrategias"][clave] = {
            "titulo": est.get("titulo", clave.title()),
            "esperados": est.get("esperados", len(activos)),
            "activos": len(px.columns),
            "ultimoDato": px.index.max().strftime("%Y-%m-%d"),
            "mesIncompleto": not mes_completo(px),
            "avisos": avisos,
            "notas": notas,
            "decisiones": dec[-DECISIONES_GUARDADAS:],
            "provisional": prov,
        }
        print(f"{clave}: {len(px.columns)} activos, ultima señal {dec[-1]['mesSenal']}, "
              f"1º {dec[-1]['ranking'][0]['senal']} ({dec[-1]['ranking'][0]['score']:.2f})")
    json.dump(salida, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
