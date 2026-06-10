"""
Cloud build: Nifty 500 stock screen + MF screen + encrypted personal-holdings panel.
Designed to run in GitHub Actions. All secrets come from environment variables:
  DASH_PASSWORD  - password that decrypts the dashboard (required)
  HOLDINGS_JSON  - {"stocks":[[sym,sector,qty,avg],...],"mf":[[name,isin,cat,units,avg],...]}
                   (optional; falls back to the embedded defaults for local runs)
Output: ./public/index.html  (single encrypted file — safe to publish)
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
import json, os, io, time, base64, hashlib, urllib.request, datetime as dt
import numpy as np, pandas as pd, yfinance as yf
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "public"); os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "index.html")
TPL_PATH = os.path.join(HERE, "template.html")
N500_CSV = os.path.join(HERE, "nifty500.csv")
H = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Accept': '*/*'}
def get(url, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout).read()

# ---- holdings ----
# Real holdings NEVER live in this file (the repo is public). They come ONLY from:
#   1. the HOLDINGS_JSON secret (cloud / GitHub Actions), or
#   2. holdings.local.json in this folder (gitignored) for local test runs.
# Format: {"stocks":[[sym,sector,qty,avg],...], "mf":[[name,isin,cat,units,avg],...]}
_h = os.environ.get("HOLDINGS_JSON")
if not _h:
    _local = os.path.join(HERE, "holdings.local.json")
    if os.path.exists(_local): _h = open(_local, encoding="utf-8").read()
if not _h:
    raise SystemExit("No holdings: set the HOLDINGS_JSON env var or create holdings.local.json")
_j = json.loads(_h)
STOCK_HOLDINGS = [tuple(x) for x in _j["stocks"]]
MF_HOLDINGS    = [tuple(x) for x in _j["mf"]]
HELD_SYMS = {s for s,_,_,_ in STOCK_HOLDINGS}

PASSWORD = os.environ.get("DASH_PASSWORD")
if not PASSWORD:
    p = os.path.join(HERE, "dash_secret.txt")
    if os.path.exists(p): PASSWORD = open(p,encoding="utf-8").read().strip()
if not PASSWORD: raise SystemExit("DASH_PASSWORD not set")

# ----------------------------------------------------------------------------- 1. universe (bundled)
n500 = pd.read_csv(N500_CSV)
sym2sector = dict(zip(n500["Symbol"], n500["Industry"]))
universe = sorted(set(n500["Symbol"]) | HELD_SYMS)
for s,sec,_,_ in STOCK_HOLDINGS: sym2sector.setdefault(s, sec)
print(f"universe: {len(universe)} symbols")

# ----------------------------------------------------------------------------- 2. prices (with retry)
def download(tickers):
    for attempt in range(3):
        try:
            d = yf.download(tickers, period="15mo", interval="1d", auto_adjust=True,
                            progress=False, group_by="ticker", threads=True)
            if d is not None and len(d): return d
        except Exception as e:
            print(f"  download attempt {attempt+1} failed: {repr(e)[:80]}")
        time.sleep(10)
    raise SystemExit("price download failed after retries")
data = download([s+".NS" for s in universe])
last_bar = str(data.index[-1].date())

def adx(df, n=14):
    h,l,c = df["High"],df["Low"],df["Close"]
    pdm=h.diff(); mdm=-l.diff()
    pdm=pdm.where((pdm>mdm)&(pdm>0),0.0); mdm=mdm.where((mdm>pdm)&(mdm>0),0.0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pdm.ewm(alpha=1/n,adjust=False).mean()/atr
    mdi=100*mdm.ewm(alpha=1/n,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return float(dx.ewm(alpha=1/n,adjust=False).mean().iloc[-1])
def rr(c,n): return float(c.iloc[-1]/c.iloc[-n]-1) if len(c)>n else np.nan

rows=[]
for s in universe:
    try:
        df=data[s+".NS"].dropna()
        if len(df)<150: continue
        c=df["Close"]; px=float(c.iloc[-1]); ret=c.pct_change().dropna()
        ma20,ma50,ma200=c.rolling(20).mean().iloc[-1],c.rolling(50).mean().iloc[-1],c.rolling(200).mean().iloc[-1]
        vol=float(ret.std()*np.sqrt(252)); downside=float(ret[ret<0].std()*np.sqrt(252)) or np.nan
        sharpe=float(ret.mean()/ret.std()*np.sqrt(252)) if ret.std()>0 else np.nan
        sortino=float(ret.mean()*252/downside) if downside and downside==downside else np.nan
        cum=(1+ret).cumprod(); maxdd=float((cum/cum.cummax()-1).min())
        hi52=float(c.iloc[-252:].max())
        m=c.resample("ME").last().pct_change().dropna().iloc[-12:] if len(c)>260 else c.pct_change().dropna()
        consist=float((m>0).mean()) if len(m) else np.nan
        rows.append(dict(sym=s,sector=sym2sector.get(s,"—"),px=px,
            r1d=rr(c,2),r1w=rr(c,6),r1m=rr(c,21),r3m=rr(c,63),r6m=rr(c,126),r1y=rr(c,252),
            adx=adx(df),vol=vol,sharpe=sharpe,sortino=sortino,maxdd=maxdd,consist=consist,
            dist_hi=float(px/hi52-1),a20=bool(px>ma20),a50=bool(px>ma50),a200=bool(px>ma200),
            dist50=float(px/ma50-1)))
    except Exception: continue
d=pd.DataFrame(rows)
print(f"screened {len(d)} stocks, last bar {last_bar}")

def z(col):
    s=d[col]; return (s-s.mean())/s.std(ddof=0)
d["mom_z"]=(z("r1m")+z("r3m")+z("r6m"))/3; d["risk_z"]=z("sharpe"); d["trend_z"]=z("adx"); d["vol_z"]=z("vol")
d["score"]=(1.0*d["mom_z"]+0.6*d["risk_z"]+0.5*d["trend_z"]-0.25*d["vol_z"]
            +0.3*d["a200"].astype(int)+0.15*d["a50"].astype(int)).round(2)
d["stab"]=(-z("vol")-z("maxdd")+z("consist")+z("sharpe")*0.5+0.4*d["a200"].astype(int)).round(2)
d["pctile"]=(d["score"].rank(pct=True)*100).round()
def signal(x):
    up=x.a50 and x.a200 and x.adx>20
    if up and x.dist50>0.20: return "EXTENDED"
    if up: return "BUY-WATCH"
    if x.score<-1.2 or (not x.a200 and x.r3m<0): return "AVOID"
    return "NEUTRAL"
d["tag"]=d.apply(signal,axis=1)
d=d.sort_values("score",ascending=False).reset_index(drop=True)
def topn(df,by,n=10,need_up=True):
    x=df[df.a50&df.a200] if need_up else df
    return x.sort_values(by,ascending=False).head(n)
def pack(df):
    out=[]
    for _,x in df.iterrows():
        f=lambda v: round(v*100,1) if v==v else 0
        out.append(dict(sym=x.sym,sector=x.sector,px=round(x.px,1),r1d=f(x.r1d),r1w=f(x.r1w),
            r1m=f(x.r1m),r3m=f(x.r3m),r6m=f(x.r6m),r1y=f(x.r1y),adx=round(x.adx),vol=round(x.vol*100),
            sharpe=round(x.sharpe,2) if x.sharpe==x.sharpe else 0,maxdd=round(x.maxdd*100),
            dist50=round(x.dist50*100,1),score=x.score,stab=x.stab,pctile=int(x.pctile),
            tag=x.tag,held=x.sym in HELD_SYMS))
    return out
stocks_all=pack(d); top_day=pack(topn(d,"r1d")); top_week=pack(topn(d,"r1w")); top_month=pack(topn(d,"r1m"))
stable=pack(d[d.a200&(d.consist>=0.6)].sort_values("stab",ascending=False).head(12))

# ----------------------------------------------------------------------------- 3. mutual funds
amfi=get('https://www.amfiindia.com/spages/NAVAll.txt').decode('utf-8','ignore')
isin2code,code2name,name_index={},{},[]
for ln in amfi.splitlines():
    p=ln.split(';')
    if len(p)>=6 and p[0].strip().isdigit():
        code=p[0].strip(); code2name[code]=p[3].strip(); name_index.append((code,p[3].strip().lower()))
        for i in (p[1].strip(),p[2].strip()):
            if i and i!='-': isin2code[i]=code
def hist(code):
    j=json.loads(get(f'https://api.mfapi.in/mf/{code}',timeout=30))
    s=pd.DataFrame(j['data']); s['date']=pd.to_datetime(s['date'],format='%d-%m-%Y')
    s['nav']=pd.to_numeric(s['nav'],errors='coerce')
    return s.dropna().sort_values('date').set_index('date')['nav']
def cagr(s,days):
    if len(s)<2: return np.nan
    w=s[s.index<=s.index[-1]-pd.Timedelta(days=days)]
    if w.empty: return np.nan
    yrs=days/365; r=s.iloc[-1]/w.iloc[-1]
    return float(r**(1/yrs)-1) if yrs>=1 else float(r-1)
def fund_metrics(code,name,cat):
    s=hist(code); ret=s.pct_change().dropna()
    sharpe=float(ret.mean()/ret.std()*np.sqrt(252)) if ret.std()>0 else np.nan
    cum=(1+ret).cumprod(); maxdd=float((cum/cum.cummax()-1).min())
    mm=s.resample('ME').last().pct_change().dropna().iloc[-36:]
    return dict(code=code,name=name,cat=cat,nav=round(float(s.iloc[-1]),2),
        r1m=cagr(s,30),r3m=cagr(s,91),r6m=cagr(s,182),r1y=cagr(s,365),r3y=cagr(s,1095),r5y=cagr(s,1825),
        sharpe=sharpe,maxdd=maxdd,consist=float((mm>0).mean()) if len(mm) else np.nan)
def find_code(kw):
    kw=[k.lower() for k in kw]
    c=[(a,n) for a,n in name_index if all(k in n for k in kw) and 'direct' in n and 'growth' in n]
    if not c: c=[(a,n) for a,n in name_index if all(k in n for k in kw) and 'direct' in n]
    return min(c,key=lambda x:len(x[1]))[0] if c else None
CURATED={
 "Flexi Cap":[["parag","parikh","flexi"],["hdfc","flexi","cap"],["quant","flexi"],["jm","flexi"]],
 "Large Cap":[["icici","bluechip"],["nippon","large","cap"],["hdfc","top","100"]],
 "Large & Mid Cap":[["bajaj","finserv","large","mid"],["motilal","large","mid"],["kotak","equity","opportunities"],["sbi","large","midcap"],["navi","large"]],
 "Mid Cap":[["motilal","midcap"],["hdfc","mid-cap","opportunities"],["quant","mid","cap"],["edelweiss","mid","cap"],["nippon","growth"]],
 "Small Cap":[["nippon","small","cap"],["quant","small","cap"],["invesco","smallcap"],["hdfc","small","cap"],["bandhan","small","cap"],["tata","small","cap"],["sbi","small","cap"]],
 "ELSS":[["quant","elss"],["hdfc","elss"],["mirae","tax","saver"],["parag","parikh","elss"]],
 "Hybrid Aggressive":[["icici","equity","debt"],["hdfc","hybrid","equity"],["quant","absolute"],["sbi","equity","hybrid"]],
 "Index":[["nippon","nifty","smallcap","250","index"],["uti","nifty","50","index"],["motilal","nifty","midcap","150"]],
}
funds=[]; seen=set()
for name,isin,cat,units,avgn in MF_HOLDINGS:
    code=isin2code.get(isin)
    if not code: continue
    try: m=fund_metrics(code,name,cat); m['held']=True; funds.append(m); seen.add(code)
    except Exception: pass
for cat,lst in CURATED.items():
    for kw in lst:
        code=find_code(kw)
        if not code or code in seen: continue
        seen.add(code)
        try: m=fund_metrics(code,code2name[code],cat); m['held']=False; funds.append(m)
        except Exception: pass
print(f"funds: {len(funds)} ({sum(f['held'] for f in funds)} held)")
fd=pd.DataFrame(funds); fd['rank_metric']=fd['r1y'].fillna(fd['r6m'])
fd['cat_pctile']=fd.groupby('cat')['rank_metric'].rank(pct=True)*100
def packf(df):
    out=[]
    for _,x in df.sort_values(['cat','rank_metric'],ascending=[True,False]).iterrows():
        g=lambda v: round(v*100,1) if v==v else None
        out.append(dict(name=x['name'][:46],cat=x['cat'],nav=x['nav'],r1m=g(x['r1m']),r3m=g(x['r3m']),
            r6m=g(x['r6m']),r1y=g(x['r1y']),r3y=g(x['r3y']),r5y=g(x['r5y']),
            sharpe=round(x['sharpe'],2) if x['sharpe']==x['sharpe'] else None,
            maxdd=round(x['maxdd']*100) if x['maxdd']==x['maxdd'] else None,
            pctile=int(x['cat_pctile']) if x['cat_pctile']==x['cat_pctile'] else None,held=bool(x['held'])))
    return out
funds_all=packf(fd)

# ----------------------------------------------------------------------------- 4. my portfolio
dmap={r['sym']:r for r in stocks_all}
my_stocks=[]; invested_s=present_s=0.0
for sym,sec,qty,avg in STOCK_HOLDINGS:
    r=dmap.get(sym); px=r['px'] if r else avg
    inv=qty*avg; pres=qty*px; invested_s+=inv; present_s+=pres
    rep=None
    if r and (r['tag']=="AVOID" or r['pctile']<35):
        same=[x for x in stocks_all if x['sector']==sec and not x['held'] and x['tag'] in ("BUY-WATCH","EXTENDED")]
        pool=same or [x for x in stocks_all if not x['held'] and x['tag']=="BUY-WATCH"]
        if pool: rep=pool[0]['sym']
    if not r: read="NO DATA"
    elif r['pctile']>=60 and r['tag'] in("BUY-WATCH","EXTENDED"): read="KEEP"
    elif r['pctile']<35 or r['tag']=="AVOID": read="TRIM/REPLACE"
    else: read="HOLD"
    my_stocks.append(dict(sym=sym,sector=sec,qty=qty,avg=round(avg,2),px=px,inv=round(inv),pres=round(pres),
        pl=round(pres-inv),plpct=round((px/avg-1)*100,1),wt=0,score=r['score'] if r else None,
        pctile=r['pctile'] if r else None,tag=r['tag'] if r else "—",read=read,rep=rep))
for m in my_stocks: m['wt']=round(100*m['pres']/present_s,1)
my_stocks.sort(key=lambda x:x['pl'])
hmap={f['name']:f for f in funds_all}; my_mf=[]; invested_m=present_m=0.0
for name,isin,cat,units,avgn in MF_HOLDINGS:
    code=isin2code.get(isin); rec=next((f for f in funds if f['code']==code),None) if code else None
    nav=rec['nav'] if rec else avgn; inv=units*avgn; pres=units*nav; invested_m+=inv; present_m+=pres
    fa=hmap.get(name[:46]); pctile=fa['pctile'] if fa else None
    read="—" if pctile is None else ("KEEP" if pctile>=60 else ("REVIEW" if pctile<35 else "HOLD"))
    my_mf.append(dict(name=name,cat=cat,units=round(units,2),avg=round(avgn,2),nav=round(nav,2),inv=round(inv),
        pres=round(pres),pl=round(pres-inv),plpct=round((nav/avgn-1)*100,1),wt=0,
        r1y=fa['r1y'] if fa else None,pctile=pctile,read=read))
for m in my_mf: m['wt']=round(100*m['pres']/present_m,1)
my_mf.sort(key=lambda x:x['pl'])
sector_w={}
for m in my_stocks: sector_w[m['sector']]=sector_w.get(m['sector'],0)+m['pres']
sector_w={k:round(100*v/present_s,1) for k,v in sorted(sector_w.items(),key=lambda x:-x[1])}
mf_cat_w={}
for m in my_mf: mf_cat_w[m['cat']]=mf_cat_w.get(m['cat'],0)+m['pres']
mf_cat_w={k:round(100*v/present_m,1) for k,v in sorted(mf_cat_w.items(),key=lambda x:-x[1])}
summary=dict(
    s_inv=round(invested_s),s_pres=round(present_s),s_pl=round(present_s-invested_s),s_plpct=round(100*(present_s/invested_s-1),2),
    m_inv=round(invested_m),m_pres=round(present_m),m_pl=round(present_m-invested_m),m_plpct=round(100*(present_m/invested_m-1),2),
    t_inv=round(invested_s+invested_m),t_pres=round(present_s+present_m),t_pl=round(present_s+present_m-invested_s-invested_m),
    t_plpct=round(100*((present_s+present_m)/(invested_s+invested_m)-1),2),
    n_up=int((d.a50&d.a200&(d.adx>20)).sum()),n_screened=len(d),
    breadth=round(100*(d.a50&d.a200&(d.adx>20)).sum()/len(d)),top=stocks_all[0]['sym'],last_bar=last_bar,
    asof=str(dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=5,minutes=30))).strftime("%Y-%m-%d %H:%M IST")))
payload=dict(summary=summary,stocks_all=stocks_all,top_day=top_day,top_week=top_week,top_month=top_month,
    stable=stable,funds_all=funds_all,my_stocks=my_stocks,my_mf=my_mf,sector_w=sector_w,mf_cat_w=mf_cat_w)

# ----------------------------------------------------------------------------- 5. encrypt + write
ITER=200_000; salt,iv=os.urandom(16),os.urandom(12)
key=hashlib.pbkdf2_hmac("sha256",PASSWORD.encode(),salt,ITER,dklen=32)
ct=AESGCM(key).encrypt(iv,json.dumps(payload).encode(),None)
enc=dict(salt=base64.b64encode(salt).decode(),iv=base64.b64encode(iv).decode(),
         ct=base64.b64encode(ct).decode(),iter=ITER)
with open(TPL_PATH,encoding="utf-8") as f: tpl=f.read()
with open(OUT,"w",encoding="utf-8") as f: f.write(tpl.replace("/*__DATA__*/","const ENC="+json.dumps(enc)+";"))
print("WROTE",OUT)
print(f"Stocks {summary['s_plpct']}% | MF {summary['m_plpct']}% | Total {summary['t_plpct']}% | breadth {summary['breadth']}%")
