#!/usr/bin/env python3
"""
Trajectory-only RNN baseline for autonomous virtual seal trips.

The RNN uses only movement history (north/east displacement vectors).
Each virtual trip starts from a haulout location and then rolls out
recursively. Empirical constraints are learned from the real trip data:
step-length distribution, signed turning-angle distribution, and trip
length. Land endpoints and land-crossing segments are rejected.

Example:
python univariate_rnn_virtual_trips.py \
  --trip_csv social_cues_2010_13_haulout_trips.csv \
  --haulout_csv Clean_IRO_HAULOUT_2010-13_spatial.csv \
  --land_shp ./Land_polygons/highres_map.shp \
  --out_dir virtual_rnn_2010_13 \
  --window 20 --epochs 100 --n_virtual_trips 100
"""

import argparse, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import wasserstein_distance, ks_2samp

KM_LAT = 110.574
KM_LON = 111.320
TRIP_IDS = ["trip_id", "trip_number", "trip", "haulout_number"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ll_to_ne(lat1, lon1, lat2, lon2):
    mid = 0.5 * (lat1 + lat2)
    n = (lat2 - lat1) * KM_LAT
    e = (lon2 - lon1) * KM_LON * np.cos(np.radians(mid))
    return np.array([n, e], dtype=float)


def ne_to_ll(lat, lon, vec):
    n, e = float(vec[0]), float(vec[1])
    new_lat = lat + n / KM_LAT
    c = max(abs(np.cos(np.radians(lat))), 1e-6)
    new_lon = lon + e / (KM_LON * c)
    return float(new_lat), float(new_lon)


def signed_turn(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12: return 0.0
    aa = math.atan2(a[1], a[0]); bb = math.atan2(b[1], b[0])
    d = bb - aa
    while d > math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return float(d)


def rotate(v, angle):
    n, e = map(float, v)
    x, y = e, n
    xr = x * math.cos(angle) - y * math.sin(angle)
    yr = x * math.sin(angle) + y * math.cos(angle)
    return np.array([yr, xr], float)


def rescale(v, length):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12: return np.array([length, 0.0], float)
    return v * (float(length) / n)


class LandMask:
    """Robust spatial-index land checks; avoids global unary_union."""
    def __init__(self, shp):
        self.available = False
        if not shp: return
        import geopandas as gpd
        import shapely
        land = gpd.read_file(shp)
        if land.crs is None: land = land.set_crs("EPSG:4326")
        land = land.to_crs("EPSG:4326")
        land = land[land.geometry.notna() & ~land.geometry.is_empty].copy()
        fixed = []
        for g in land.geometry:
            try:
                if not g.is_valid:
                    g = shapely.make_valid(g) if hasattr(shapely, "make_valid") else g.buffer(0)
                if g is not None and not g.is_empty and not g.is_valid: g = g.buffer(0)
                fixed.append(g)
            except Exception:
                fixed.append(None)
        land["geometry"] = fixed
        land = land[land.geometry.notna() & ~land.geometry.is_empty].reset_index(drop=True)
        self.land = land
        self.sindex = land.sindex
        self.available = len(land) > 0
        print(f"Usable land polygons: {len(land):,}")

    def on_land(self, lat, lon):
        if not self.available: return False
        from shapely.geometry import Point
        p = Point(float(lon), float(lat))
        for i in self.sindex.intersection(p.bounds):
            g = self.land.geometry.iloc[i]
            try:
                if g.contains(p) or g.touches(p): return True
            except Exception: pass
        return False

    def crosses_land(self, lat1, lon1, lat2, lon2):
        if not self.available: return False
        from shapely.geometry import LineString
        line = LineString([(float(lon1), float(lat1)), (float(lon2), float(lat2))])
        for i in self.sindex.intersection(line.bounds):
            g = self.land.geometry.iloc[i]
            try:
                inter = line.intersection(g)
                if inter.is_empty: continue
                if inter.geom_type == "Point": continue
                return True
            except Exception: pass
        return False


def load_trips(path, gap_hours):
    df = pd.read_csv(path)
    need = ["seal", "d_date", "lat", "lon"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"Trip CSV missing {miss}")
    df["d_date"] = pd.to_datetime(df["d_date"], errors="coerce", utc=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=need).copy()
    trip_col = next((c for c in TRIP_IDS if c in df.columns), None)
    if trip_col:
        df["_trip"] = df["seal"].astype(str) + "_" + df[trip_col].astype(str)
        print("Trip ID column:", trip_col)
    else:
        parts = []
        for seal, g in df.sort_values(["seal","d_date"]).groupby("seal"):
            g = g.copy()
            gaps = g["d_date"].diff().dt.total_seconds().div(3600)
            block = gaps.gt(gap_hours).fillna(False).cumsum()
            g["_trip"] = str(seal) + "_" + block.astype(str)
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
        print(f"Trip IDs inferred from >{gap_hours} h gaps")
    return df.sort_values(["_trip","d_date"]).reset_index(drop=True)


def extract_trip_records(df, max_gap, min_points):
    records, all_steps, all_turns = [], [], []
    for tid, g in df.groupby("_trip"):
        g = g.sort_values("d_date")
        if len(g) < min_points: continue
        steps = []
        for i in range(1, len(g)):
            dt = (g["d_date"].iloc[i] - g["d_date"].iloc[i-1]).total_seconds()/3600
            if not np.isfinite(dt) or dt <= 0 or dt > max_gap: continue
            steps.append(ll_to_ne(g["lat"].iloc[i-1], g["lon"].iloc[i-1], g["lat"].iloc[i], g["lon"].iloc[i]))
        steps = np.asarray(steps, float)
        if len(steps) < 3: continue
        lengths = np.linalg.norm(steps, axis=1)
        turns = np.array([signed_turn(steps[i-1], steps[i]) for i in range(1, len(steps))], float)
        records.append({
            "trip_id": tid, "seal": g["seal"].iloc[0], "steps": steps,
            "n_steps": len(steps), "total_distance_km": float(lengths.sum())
        })
        all_steps.extend(steps.tolist()); all_turns.extend(turns.tolist())
    if not records: raise RuntimeError("No usable trips after preprocessing")
    return records, np.asarray(all_steps,float), np.asarray(all_turns,float)


class Scaler:
    def fit(self, x):
        x = np.asarray(x,float); self.mean=x.mean(0); self.std=x.std(0); self.std[self.std<1e-8]=1; return self
    def transform(self,x): return (np.asarray(x,float)-self.mean)/self.std
    def inverse(self,x): return np.asarray(x,float)*self.std+self.mean


class WinDataset(Dataset):
    def __init__(self, trips, window, scaler):
        self.data=[]
        for t in trips:
            s=scaler.transform(t["steps"]).astype(np.float32)
            for i in range(window,len(s)):
                self.data.append((s[i-window:i], s[i]))
        if not self.data: raise RuntimeError("No training windows; reduce --window")
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        x,y=self.data[i]
        return torch.tensor(x), torch.tensor(y)


class RNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.rnn=nn.RNN(2,hidden,batch_first=True,nonlinearity="tanh")
        self.fc=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,2))
    def forward(self,x):
        o,_=self.rnn(x)
        return self.fc(o[:,-1])


class Constraints:
    def __init__(self,trips,steps,turns,qlo,qhi):
        lens=np.linalg.norm(steps,axis=1)
        self.step_lo=float(np.quantile(lens,qlo)); self.step_hi=float(np.quantile(lens,qhi)); self.step_med=float(np.median(lens))
        self.turn_lo=float(np.quantile(turns,qlo)); self.turn_hi=float(np.quantile(turns,qhi))
        self.step_samples=lens; self.turn_samples=turns
        self.trip_steps=np.array([t["n_steps"] for t in trips],int)
        self.trip_dist=np.array([t["total_distance_km"] for t in trips],float)
    def constrain(self, prev, prop):
        v=np.asarray(prop,float); L=np.linalg.norm(v)
        if not np.isfinite(L) or L<1e-10: v=rescale(prev,self.step_med); L=self.step_med
        L=float(np.clip(L,self.step_lo,self.step_hi)); v=rescale(v,L)
        turn=signed_turn(prev,v); turn2=float(np.clip(turn,self.turn_lo,self.turn_hi))
        if abs(turn2-turn)>1e-12: v=rescale(rotate(prev,turn2),L)
        return v
    def sample_steps(self,rng,min_steps):
        x=self.trip_steps[self.trip_steps>=min_steps]
        return int(rng.choice(x if len(x) else self.trip_steps))


def valid_move(lat,lon,nlat,nlon,land,departure=False):
    if not (-90<=nlat<=90 and -180<=nlon<=180): return False
    if land and land.available:
        if land.on_land(nlat,nlon): return False
        if not departure and land.crosses_land(lat,lon,nlat,nlon): return False
    return True


def repair_move(lat,lon,prev,prop,C,land,departure=False):
    base=C.constrain(prev,prop)
    for factor in [1.0,.8,.6,.4,.25]:
        for deg in [0,10,-10,20,-20,35,-35,50,-50,70,-70,90,-90,120,-120,160,-160,180]:
            v=rescale(rotate(base,math.radians(deg)), max(C.step_lo,np.linalg.norm(base)*factor))
            v=C.constrain(prev,v)
            nlat,nlon=ne_to_ll(lat,lon,v)
            if valid_move(lat,lon,nlat,nlon,land,departure): return nlat,nlon,v,(deg!=0 or factor!=1)
    return lat,lon,np.zeros(2),True


def synthetic_departure(h_lat,h_lon,window,C,land,rng):
    coords=[(float(h_lat),float(h_lon))]; steps=[]
    # first marine step from haulout
    for _ in range(200):
        L=float(rng.choice(C.step_samples)); a=float(rng.uniform(-math.pi,math.pi))
        v=np.array([L*math.cos(a),L*math.sin(a)])
        nlat,nlon=ne_to_ll(h_lat,h_lon,v)
        if valid_move(h_lat,h_lon,nlat,nlon,land,departure=True):
            coords.append((nlat,nlon)); steps.append(v); break
    if not steps: raise RuntimeError("No valid departure direction from haulout")
    while len(steps)<window:
        prev=steps[-1]; L=float(rng.choice(C.step_samples)); ang=float(rng.choice(C.turn_samples))
        prop=rescale(rotate(prev,ang),L)
        nlat,nlon,v,_=repair_move(coords[-1][0],coords[-1][1],prev,prop,C,land)
        if np.linalg.norm(v)<1e-10: continue
        coords.append((nlat,nlon)); steps.append(v)
    return np.asarray(coords,float), np.asarray(steps,float)


@torch.no_grad()
def generate_trip(model,scaler,C,h_lat,h_lon,window,nsteps,device,land,rng):
    coords,steps=synthetic_departure(h_lat,h_lon,window,C,land,rng)
    steps=list(steps); interventions=0
    while len(steps)<nsteps:
        x=scaler.transform(np.asarray(steps[-window:])).astype(np.float32)
        pred=model(torch.tensor(x[None],device=device)).cpu().numpy()
        prop=scaler.inverse(pred)[0]
        lat,lon=coords[-1]
        nlat,nlon,v,changed=repair_move(lat,lon,steps[-1],prop,C,land)
        interventions += int(changed)
        if np.linalg.norm(v)<1e-10:
            # empirical fallback
            prop=rescale(rotate(steps[-1],float(rng.choice(C.turn_samples))),float(rng.choice(C.step_samples)))
            nlat,nlon,v,_=repair_move(lat,lon,steps[-1],prop,C,land)
        if np.linalg.norm(v)<1e-10: break
        coords=np.vstack([coords,[nlat,nlon]]); steps.append(v)
    return coords,np.asarray(steps,float),interventions


def train(model,loader,device,epochs,lr):
    opt=torch.optim.Adam(model.parameters(),lr=lr); lossfn=nn.MSELoss(); hist=[]
    for ep in range(1,epochs+1):
        model.train(); vals=[]
        for x,y in loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(); p=model(x); loss=lossfn(p,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); vals.append(loss.item())
        m=float(np.mean(vals)); hist.append({"epoch":ep,"train_mse_scaled":m})
        if ep==1 or ep%10==0 or ep==epochs: print(f"Epoch {ep:03d}/{epochs} MSE={m:.6f}")
    return pd.DataFrame(hist)


def load_haulouts(path):
    df=pd.read_csv(path)
    if not {"lat","lon"}.issubset(df.columns): raise ValueError("Haulout CSV needs lat, lon")
    df["lat"]=pd.to_numeric(df["lat"],errors="coerce"); df["lon"]=pd.to_numeric(df["lon"],errors="coerce")
    df=df.dropna(subset=["lat","lon"]).drop_duplicates(subset=["lat","lon"]).reset_index(drop=True)
    if len(df)==0: raise RuntimeError("No usable haulout locations")
    return df


def plot_map(vdf,land,out):
    fig,ax=plt.subplots(figsize=(10,8))
    if land and land.available: land.land.plot(ax=ax,facecolor="0.93",edgecolor="0.5",linewidth=.2,zorder=0)
    for _,g in vdf.groupby("virtual_trip_id"):
        g=g.sort_values("point_index"); ax.plot(g.lon,g.lat,lw=.9,alpha=.65); ax.scatter(g.lon.iloc[0],g.lat.iloc[0],s=12,marker="s")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("Autonomous RNN virtual trips from haulouts"); ax.grid(alpha=.2)
    if len(vdf):
        dx=max(.05,(vdf.lon.max()-vdf.lon.min())*.08); dy=max(.05,(vdf.lat.max()-vdf.lat.min())*.08)
        ax.set_xlim(vdf.lon.min()-dx,vdf.lon.max()+dx); ax.set_ylim(vdf.lat.min()-dy,vdf.lat.max()+dy)
    fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig)


def histplot(real,virt,xlabel,title,out):
    fig,ax=plt.subplots(figsize=(7,5)); ax.hist(real,bins=60,density=True,alpha=.45,label="Real"); ax.hist(virt,bins=60,density=True,alpha=.45,label="Virtual")
    ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.set_title(title); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig)


def main(a):
    set_seed(a.seed); out=Path(a.out_dir); plots=out/"plots"; out.mkdir(parents=True,exist_ok=True); plots.mkdir(exist_ok=True)
    device=torch.device(a.device if a.device!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")); print("Device:",device)
    df=load_trips(a.trip_csv,a.trip_gap_hours)
    trips,steps,turns=extract_trip_records(df,a.max_step_gap_hours,max(5,a.window+2))
    print(f"Usable trips: {len(trips):,}; steps: {len(steps):,}")
    scaler=Scaler().fit(steps); C=Constraints(trips,steps,turns,a.constraint_low_q,a.constraint_high_q)
    pd.DataFrame([{
        "step_low_km":C.step_lo,"step_high_km":C.step_hi,
        "turn_low_deg":math.degrees(C.turn_lo),"turn_high_deg":math.degrees(C.turn_hi),
        "trip_steps_median":float(np.median(C.trip_steps)),"trip_distance_median_km":float(np.median(C.trip_dist))
    }]).to_csv(out/"empirical_constraints.csv",index=False)
    print(f"Step support: {C.step_lo:.4f}-{C.step_hi:.4f} km")
    print(f"Turn support: {math.degrees(C.turn_lo):.1f}-{math.degrees(C.turn_hi):.1f} deg")

    ds=WinDataset(trips,a.window,scaler); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True)
    model=RNN(a.hidden_dim).to(device); history=train(model,loader,device,a.epochs,a.lr); history.to_csv(out/"training_history.csv",index=False)
    torch.save({"model":model.state_dict(),"window":a.window,"hidden_dim":a.hidden_dim,"mean":scaler.mean,"std":scaler.std},out/"rnn_model.pt")

    haul=load_haulouts(a.haulout_csv); land=LandMask(a.land_shp); rng=np.random.default_rng(a.seed+1000)
    rows=[]; summaries=[]; vsteps=[]; vturns=[]
    generated=0; attempts=0
    while generated<a.n_virtual_trips and attempts<a.n_virtual_trips*30:
        attempts+=1; h=haul.iloc[int(rng.integers(0,len(haul)))]; nsteps=C.sample_steps(rng,a.window)
        try: coords,st,inter=generate_trip(model,scaler,C,float(h.lat),float(h.lon),a.window,nsteps,device,land,rng)
        except RuntimeError: continue
        if len(st)<a.window: continue
        generated+=1; vid=f"VTRIP_{generated:05d}"
        sl=np.linalg.norm(st,axis=1); ta=np.array([signed_turn(st[i-1],st[i]) for i in range(1,len(st))])
        vsteps.extend(sl.tolist()); vturns.extend(ta.tolist())
        for i,(lat,lon) in enumerate(coords): rows.append({"virtual_trip_id":vid,"point_index":i,"lat":lat,"lon":lon,"haulout_lat":float(h.lat),"haulout_lon":float(h.lon)})
        summaries.append({"virtual_trip_id":vid,"requested_steps":nsteps,"n_steps":len(st),"total_distance_km":float(sl.sum()),"mean_step_km":float(sl.mean()),"mean_abs_turn_deg":float(np.degrees(np.abs(ta)).mean()) if len(ta) else np.nan,"constraint_interventions":inter,"constraint_intervention_fraction":inter/max(1,len(st)-a.window)})
        if generated==1 or generated%10==0 or generated==a.n_virtual_trips: print(f"Generated {generated}/{a.n_virtual_trips}")

    vdf=pd.DataFrame(rows); sdf=pd.DataFrame(summaries); vdf.to_csv(out/"virtual_trips.csv",index=False); sdf.to_csv(out/"virtual_trip_summary.csv",index=False)
    real_step=np.linalg.norm(steps,axis=1); real_trip=np.array([t["total_distance_km"] for t in trips],float)
    comps=[]
    for name,r,v in [
        ("step_length_km",real_step,np.asarray(vsteps)),
        ("signed_turn_angle_rad",turns,np.asarray(vturns)),
        ("trip_steps",C.trip_steps.astype(float),sdf.n_steps.to_numpy(float)),
        ("trip_total_distance_km",real_trip,sdf.total_distance_km.to_numpy(float))]:
        ks=ks_2samp(r,v); comps.append({"metric":name,"real_mean":float(np.mean(r)),"virtual_mean":float(np.mean(v)),"wasserstein":float(wasserstein_distance(r,v)),"ks_statistic":float(ks.statistic),"ks_pvalue":float(ks.pvalue)})
    pd.DataFrame(comps).to_csv(out/"real_vs_virtual_distribution.csv",index=False)
    plot_map(vdf,land,plots/"virtual_trajectories.png")
    histplot(real_step,np.asarray(vsteps),"Step length (km)","Real vs virtual step length",plots/"step_length_distribution.png")
    histplot(np.degrees(turns),np.degrees(np.asarray(vturns)),"Signed turning angle (degrees)","Real vs virtual turning angle",plots/"turning_angle_distribution.png")
    histplot(real_trip,sdf.total_distance_km.to_numpy(float),"Total trip distance (km)","Real vs virtual trip distance",plots/"trip_distance_distribution.png")
    print("Saved to",out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--trip_csv",required=True); p.add_argument("--haulout_csv",required=True); p.add_argument("--land_shp",default=None); p.add_argument("--out_dir",default="univariate_rnn_virtual_trips")
    p.add_argument("--window",type=int,default=20); p.add_argument("--hidden_dim",type=int,default=64); p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch_size",type=int,default=256); p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--n_virtual_trips",type=int,default=100); p.add_argument("--constraint_low_q",type=float,default=.005); p.add_argument("--constraint_high_q",type=float,default=.995)
    p.add_argument("--trip_gap_hours",type=float,default=24); p.add_argument("--max_step_gap_hours",type=float,default=24); p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto"); p.add_argument("--seed",type=int,default=42)
    main(p.parse_args())
