"""Render one frozen-source RetiPath architecture figure; no training or target access."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.support_partition import _BC_RADIUS, _AC_RADIUS
OUT=ROOT/'output/figures/model_schematics'
REG=ROOT/'output/evaluations/retipath_final_model_evidence_20260913/model_registry.json'
MOVIE=ROOT/'data/real/schottdorf_lee_2021_macaque/1x10_256.mpg'
W,H=1680,900
INK='#20252b'; GRAY='#56616b'; BLUE='#2867a5'; CYAN='#138598'; GREEN='#298052'; ORANGE='#c86b28'; PURPLE='#7653a4'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def frozen_data():
    prior=json.loads((OUT/'verified_architecture.json').read_text(encoding='utf-8'))
    relevant=[s for s in prior['source_files'] if s['path'].startswith('models/mechanistic_retina/') or 'MIGRATION_PROTOCOL' in s['path'] or 'model_registry' in s['path']]
    for src in relevant:
        assert sha(ROOT/src['path'])==src['sha256'],src['path']
    registry=json.loads(REG.read_text(encoding='utf-8'))
    assert registry['architecture_id']=='retipath_spatial_conductance_v1'
    ref=registry['cells']['67#14']['RetiPath']['2026091301']
    cp_path=ROOT/ref['path']; assert sha(cp_path)==ref['sha256']
    cp=torch.load(cp_path,map_location='cpu',weights_only=True)
    model=RetiPath(MechanisticRetinaConfig(**cp['model_config']),cp['cone_positions_degs'],cp['cell_positions_degs'],tuple(cp['cell_types']),tuple(cp['polarities']),rms_e=torch.tensor(cp['rms']['e']),rms_i=torch.tensor(cp['rms']['i']))
    model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False)
    with torch.inference_mode():
        spatial=model.feature_bank.path_spatial_basis[0].numpy().reshape(4,2,17,17)
        temporal=model.feature_bank.temporal_basis.numpy()
        weights=model.bipolar.positive_weights()[0].numpy()
        geometry=cp['cone_positions_degs'].numpy().reshape(17,17,2)
        center=cp['cell_positions_degs'][0].numpy()
        masks=np.stack((model.feature_bank.bc_support[0].numpy(),model.feature_bank.ac_support[0].numpy())).reshape(2,17,17)
        graph=model.h1.graph.dense_kernel().numpy()
        alpha=float(model.alpha)
    assert spatial.shape==(4,2,17,17) and temporal.shape==(2,3,16) and weights.shape==(2,2,3)
    assert cp['model_config']['lag_steps']==16 and cp['model_config']['dt_ms']==1000/150
    assert sha(MOVIE)==prior['movie_sha256']
    capture=cv2.VideoCapture(str(MOVIE)); frames=[]
    try:
        for i in range(808,813):
            capture.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=capture.read();assert ok
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    closest=int(np.argmin(np.sum((geometry.reshape(-1,2)-center)**2,axis=1)))
    cr,cc=divmod(closest,17)
    rs=max(0,min(cr-4,8));cs=max(0,min(cc-4,8))
    roi=(slice(rs,rs+9),slice(cs,cs+9))
    details={'checkpoint':ref,'cell':'67#14','seed':2026091301,'source_files':relevant,'movie_frames':[808,809,810,811,812],'movie_sha256':sha(MOVIE),'roi':[rs,rs+9,cs,cs+9],'source_shapes':prior['shape_probe']['RetiPath'],'no_training':True,'no_new_target_access':True,'no_model_forward':True,'no_prediction_arrays_read':True}
    return dict(spatial=spatial,temporal=temporal,geometry=geometry,center=center,masks=masks,graph=graph,closest=closest,roi=roi,frames=frames,details=details)


REFERENCE=Path('C:/Users/win11-pc/Downloads/ChatGPT Image 2026年9月17日 10_27_24.png')
FONT='C:/Windows/Fonts/YuGothM.ttc'
FONT_BOLD='C:/Windows/Fonts/YuGothB.ttc'
FONT_EN='C:/Windows/Fonts/segoeui.ttf'
FONT_EN_BOLD='C:/Windows/Fonts/segoeuib.ttf'
DIRECT='#4c86af'; BROAD='#4c9470'; AC='#9470ba'; EDGE='#737b83'

ENGLISH_LABELS={
    '自然動画入力':'Natural movie input',
    'T フレーム':'T frames',
    '錐体入力 L+M':'Cone input L+M',
    'H1 水平相互作用':'H1 lateral\ninteraction',
    '局所近傍の集約':'Local aggregation',
    '再帰的な状態更新':'Recurrent state update',
    'BC 広域経路':'BC broad pathway',
    'BC 直接経路':'BC direct pathway',
    '広域支持':'Broad support',
    '局所支持':'Local support',
    '（中心を含む）':'(includes center)',
    '空間モード':'Spatial modes',
    '時間処理':'Time filters',
    '持続':'Sustained',
    '瞬変':'Transient',
    '加重和':'Weighted sum',
    '非対称変換':'Asymmetric\ntransform',
    '分岐内で K 共通':'Shared over K',
    '基底・重み共有':'Shared bases / weights',
    'AC 経路':'AC pathway',
    '遅延・低域通過':'Delay / low-pass',
    '符号・利得':'Sign / gain',
    'コンダクタンス\n変換':'Conductance\nmapping',
    '膜電位\nダイナミクス':'Membrane\ndynamics',
    'RGC 読み出し':'RGC readout',
    '空間平均':'Mode average',
    '基線・尺度調整':'Baseline / scale',
    '閾値・利得':'Threshold / gain',
    '順応・偏置':'Adaptation / bias',
    '過去の実測発火':'Observed spikes',
    '1 bin遅延・τ=30 ms':'1-bin lag; τ=30 ms',
    '発火確率':'Spike\nprobability',
    '単一 RGC':'Single RGC',
}


class Figure:
    def __init__(self,language='ja'):
        self.language=language
        self.stem='retipath_architecture'+('_en' if language=='en' else '')
        self.fig,self.ax=plt.subplots(figsize=(W/100,H/100),dpi=100)
        self.fig.subplots_adjust(0,0,1,1)
        self.ax.set(xlim=(0,W),ylim=(H,0),aspect='equal');self.ax.axis('off')
        self.labels=[]
    def text(self,x,y,s,size=11.8,bold=False,color='#111119',ha='center'):
        if self.language=='en':
            if s=='遅延・低域通過':x+=26
            s=ENGLISH_LABELS.get(s,s)
            font=font_manager.FontProperties(fname=FONT_EN_BOLD if bold else FONT_EN)
        else:
            font=font_manager.FontProperties(fname=FONT_BOLD if bold else FONT)
        a=self.ax.text(x,y,s,fontsize=size,fontproperties=font,ha=ha,va='center',color=color,linespacing=1.20,zorder=40)
        self.labels.append(a)
    def line(self,pts,color='#202028',lw=1.0,dash=False,z=9):
        self.ax.plot(*zip(*pts),color=color,lw=lw,ls=(0,(3,2)) if dash else '-',zorder=z,solid_capstyle='round')
    def arrow(self,pts,color='#202028',dash=False,lw=1.1):
        if len(pts)>2:self.line(pts[:-1],color,lw,dash,z=30)
        self.ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=10,lw=lw,color=color,linestyle=(0,(3,2)) if dash else '-',shrinkA=0,shrinkB=0,zorder=30))
    def dot(self,x,y,color='#202028',r=2.0):
        self.ax.add_patch(Circle((x,y),r,facecolor=color,edgecolor='none',zorder=23))
    def rect(self,x,y,w,h,fill='white',edge='#646875',lw=.65,rounded=False,z=14):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=6',facecolor=fill,edgecolor=edge,lw=lw,zorder=z) if rounded else Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,lw=lw,zorder=z)
        self.ax.add_patch(patch);return patch
    def node(self,x,y,s,r=12,color='#373440',fill='white',size=12):
        self.ax.add_patch(Circle((x,y),r,facecolor=fill,edgecolor=color,lw=1.0,zorder=18))
        if s:self.text(x,y,s,size=size)
    def plane(self,x,y,w,h,color='#708bbd',n=17,z=15,front=False):
        transform=Affine2D.from_values(1,-.40,0,1,x,y)
        xy=lambda u,v:transform.transform((u,v))
        pts=[xy(0,0),xy(w,0),xy(w,h),xy(0,h)]
        base=np.array(to_rgb(color));fill=tuple(np.ones(3)*.86+base*.14)
        self.ax.add_patch(Polygon(pts,facecolor='#fafafa' if front else fill,edgecolor=color,lw=.70,zorder=z))
        for j in range(n+1):
            self.line([xy(w*j/n,0),xy(w*j/n,h)],color,.27,z=z+.1)
            self.line([xy(0,h*j/n),xy(w,h*j/n)],color,.27,z=z+.1)
        return xy
    def heatmap(self,x,y,w,h,values,color='#548aac'):
        values=np.asarray(values);nr,nc=values.shape
        scale=values/max(float(values.max()),1e-12)
        for i in range(nr):
            for j in range(nc):
                t=float(scale[i,j]);base=np.array(to_rgb(color))
                shade=tuple(np.ones(3)*(1-.82*t)+base*.82*t)
                self.rect(x+j*w/nc,y+i*h/nr,w/nc+.03,h/nr+.03,shade,'none',0,z=18)
        for j in range(nc+1):self.line([(x+j*w/nc,y),(x+j*w/nc,y+h)],'#b9b9bd',.20,z=20)
        for i in range(nr+1):self.line([(x,y+i*h/nr),(x+w,y+i*h/nr)],'#b9b9bd',.20,z=20)
        self.rect(x,y,w,h,'none',color,.7,z=22)
    def save(self):
        self.fig.canvas.draw();ren=self.fig.canvas.get_renderer();bounds=[]
        for a in self.labels:
            b=a.get_window_extent(ren)
            assert b.x0>=0 and b.y0>=0 and b.x1<=W and b.y1<=H,(a.get_text(),b)
            bounds.append({'text':a.get_text(),'bounds':[b.x0,b.y0,b.x1,b.y1]})
        for suffix in ('svg','pdf','png'):
            self.fig.savefig(OUT/f'{self.stem}.{suffix}',dpi=240,transparent=False,facecolor='white')
        (OUT/f'qa/{self.stem}_text_bounds.json').write_text(json.dumps(bounds,ensure_ascii=False,indent=2),encoding='utf-8')
        plt.close(self.fig)


def stack(f,x,y,w,h,color,layers=4,values=None,n=4):
    pale=tuple(.89+.11*np.array(to_rgb(color)))
    for j in range(layers-1,0,-1):
        f.rect(x+5*j,y-5*j,w,h,pale,color,.7,rounded=True,z=14)
    f.rect(x,y,w,h,pale,color,.85,rounded=True,z=15)
    if values is not None:
        f.heatmap(x+5,y+5,w-10,h-10,values,color)
    elif n:
        for j in range(1,n):
            f.line([(x+w*j/n,y+4),(x+w*j/n,y+h-4)],color,.22,z=20)
            f.line([(x+4,y+h*j/n),(x+w-4,y+h*j/n)],color,.22,z=20)


def cycle(f,x,y,r,color=INK):
    a=np.linspace(np.deg2rad(28),np.deg2rad(333),45)
    pts=list(zip(x+r*np.cos(a),y+r*np.sin(a)))
    f.line(pts[:-1],color,.95,z=25)
    f.arrow(pts[-3:],color,lw=.95)


def region(f,x,y,w,h,color,dash=False):
    p=f.rect(x,y,w,h,'white',color,.75,rounded=True,z=0)
    if dash:p.set_linestyle((0,(3,2)))


def inputs_and_h1(f,d):
    f.text(105,144,'自然動画入力',size=14,bold=True)
    f.text(105,169,'T フレーム',size=11.6)
    for j,im in enumerate(d['frames']):
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        x,y=43+7*j,207+7*j
        f.ax.imshow(gray,cmap='gray',vmin=0,vmax=255,extent=(x,x+94,y+107,y),zorder=15+j)
        f.rect(x,y,94,107,'none','#555d64',.65,z=21+j)
    f.text(26,249,r'$t-2$',size=11)
    f.text(26,279,r'$t-1$',size=11)
    f.text(26,311,r'$t$',size=11)
    f.text(54,363,r'$\vdots$',size=15)
    f.arrow([(112,352),(112,395)],lw=1.05)
    f.text(110,425,'錐体入力 L+M',size=13,bold=True)
    f.text(111,450,r'$T\times17\times17$',size=12.4)
    gray_values=np.full((7,7),.05);gray_values[3,3]=.60
    stack(f,51,504,100,111,'#8b9297',5,gray_values)
    f.text(109,644,r'$T\times289$',size=13)
    f.text(109,669,'T=150',size=11.6)
    f.text(294,293,'H1 水平相互作用',size=14,bold=True)
    f.text(294,323,r'$T\times17\times17$',size=12.4)
    f.arrow([(171,550),(197,550),(197,463),(238,463)],lw=1.05)
    f.dot(197,463)
    f.arrow([(197,463),(197,362),(377,362),(377,448)],lw=.9)
    local=np.full((7,7),.02);local[2:5,2:5]=.17;local[3,3]=.8
    stack(f,241,403,106,118,DIRECT,5,local)
    f.text(296,545,'局所近傍の集約',size=11.8)
    f.arrow([(296,554),(296,577)],lw=.9)
    cycle(f,296,604,23)
    f.text(295,652,'再帰的な状態更新',size=11.8)
    f.arrow([(322,604),(370,604),(370,482),(377,482),(377,474)],lw=.9)
    f.text(354,577,r'$G^{\mathsf{T}}$',size=11)
    f.node(377,462,'−',r=12,size=15)
    f.arrow([(390,462),(399,462),(399,219),(408,219)],lw=1.05)
    f.arrow([(399,462),(399,525),(408,525)],lw=1.05)


def bc_path(f,d,broad=False):
    off=306 if broad else 0;color=BROAD if broad else DIRECT
    region(f,412,80+off,532,248,color)
    f.text(428,106+off,'BC 広域経路' if broad else 'BC 直接経路',size=14.6,bold=True,ha='left')
    f.text(429,133+off,r'$T\times17\times17$',size=11.5,ha='left')
    stack(f,435,181+off,77,80,color,5,d['masks'][int(broad)][d['roi']])
    f.text(476,284+off,'広域支持' if broad else '局所支持',size=11.7)
    f.text(476,306+off,'（中心を含む）',size=10.5)
    f.arrow([(533,220+off),(552,220+off)],lw=.95)
    f.text(583,151+off,'空間モード',size=11)
    f.text(583,172+off,r'$K=2$',size=11.6)
    for k,y in enumerate((187,246)):
        f.heatmap(558,y+off,50,49,d['spatial'][2 if broad else 0,k][d['roi']],color)
    f.line([(614,188+off),(624,188+off),(624,295+off),(614,295+off)],color,.9,z=24)
    f.text(687,153+off,'時間処理',size=11)
    f.arrow([(624,219+off),(644,219+off),(644,200+off),(656,200+off)],lw=.9)
    f.arrow([(644,219+off),(644,263+off),(656,263+off)],lw=.9)
    branch_width=84 if f.language=='en' else 64
    for y,label in ((179,'持続'),(242,'瞬変')):
        f.rect(657,y+off,branch_width,42,'#fcfdfd',color,.7,rounded=True)
        f.text(657+branch_width/2,y+21+off,label,size=11.7)
    f.arrow([(660+branch_width,200+off),(767,200+off)],lw=.9)
    f.arrow([(660+branch_width,263+off),(767,263+off)],lw=.9)
    f.rect(769,178+off,111,106,'white',color,.85,rounded=True)
    f.text(824,198+off,'加重和',size=11.8)
    f.line([(779,215+off),(870,215+off)],color,.6,z=22)
    f.text(824,236+off,'非対称変換',size=11.8)
    f.text(824,263+off,'分岐内で K 共通',size=10.2)
    f.text(821,306+off,r'$T\times K\times2$',size=12)
    if broad:
        f.arrow([(881,525),(954,525),(954,658),(493,658),(493,778),(514,778)],BROAD,lw=1.0)
    else:
        f.arrow([(881,219),(899,219)],DIRECT,lw=1.0)
        f.node(915,219,r'$\Sigma_p$',r=14,color=DIRECT,size=11.5)
        f.arrow([(930,219),(962,219),(962,300),(1065,300)],DIRECT,lw=1.0)
        f.dot(962,300,DIRECT)
        f.arrow([(962,300),(962,516),(1065,516)],DIRECT,lw=1.0)
        f.text(949,186,r'$T\times K$',size=11.4)
    if not broad:
        f.line([(714,334),(714,378)],'#667d70',.8,dash=True,z=20)
        f.rect(650,345,128,21,'white','none',0,z=25)
        f.text(714,356,'基底・重み共有',size=10.8)


def ac_path(f):
    region(f,517,685,427,156,AC)
    f.text(534,709,'AC 経路',size=14.6,bold=True,ha='left')
    f.text(535,735,r'$T\times K\times2$',size=11.5,ha='left')
    stack(f,537,764,46,51,AC,2,n=2)
    f.text(561,828,r'$K=2$',size=10.5)
    f.text(694,710,'遅延・低域通過',size=11.8)
    f.arrow([(590,786),(607,786),(607,754),(623,754)],AC,lw=.95)
    f.arrow([(607,786),(607,810),(623,810)],AC,lw=.95)
    branch_extra=18 if f.language=='en' else 0
    for y,label in ((738,'持続'),(794,'瞬変')):
        f.rect(625,y,93+branch_extra,34,'#fcfaff',AC,.75,rounded=True)
        f.text(663 if f.language=='en' else 652,y+17,label,size=10.5)
        cycle(f,699+branch_extra,y+17,10,AC)
        f.arrow([(722+branch_extra,y+17),(759,y+17)],AC,lw=.9)
    f.rect(761,739,115,88,'white',AC,.8,rounded=True)
    f.text(818,762,'符号・利得',size=11.8)
    f.text(818,799,r'$T\times K\times2$',size=11.5)
    f.arrow([(879,784),(899,784)],AC,lw=1)
    f.node(915,784,r'$\Sigma_p$',r=14,color=AC,size=11.5)
    f.arrow([(930,784),(980,784),(980,373),(1065,373)],AC,lw=1.0)
    f.dot(980,589,AC)
    f.arrow([(980,589),(1065,589)],AC,lw=1)
    f.text(959,813,r'$T\times K$',size=11.5)
    f.line([(973,516),(987,516)],'white',4.5,z=33)
    f.line([(973,516),(987,516)],DIRECT,1.0,z=34)


def conductance_and_membrane(f):
    region(f,995,208,145,468,EDGE,True)
    f.text(1067,235,'コンダクタンス\n変換',size=12.4,bold=True)
    for k,y in ((1,275),(2,491)):
        for kind,yy,color in (('E',y,'#5b9a78'),('I',y+73,AC)):
            f.text(1032,yy+4,fr'$g_{{{kind},{k}}}(t)$',size=11.9)
            stack(f,1071,yy,42,48,color,3,n=2)
            f.arrow([(1117,yy+24),(1166,yy+24)],color,lw=1.0)
        f.text(1093,y+140,r'$T\times1$',size=10.8)
    region(f,1150,208,153,468,EDGE,True)
    f.text(1226,235,'膜電位\nダイナミクス',size=12.4,bold=True)
    for k,y in ((1,277),(2,493)):
        f.rect(1171,y,112,155,'#f3f9fd',DIRECT,.7,rounded=True)
        f.text(1227,y+26,fr'$V_{k}(t)$',size=15)
        cycle(f,1227,y+87,25,DIRECT)
        f.text(1191,y+110,r'$t-1$',size=10.1)
        f.text(1267,y+64,r'$t$',size=10.1)
        f.text(1227,y+138,r'$T\times1$',size=11.0)
    f.arrow([(1286,355),(1311,355),(1311,401),(1340,401)],lw=1)
    f.arrow([(1286,571),(1311,571),(1311,425),(1340,425)],lw=1)
    f.text(1325,382,r'$V_1$',size=10.6)
    f.text(1325,447,r'$V_2$',size=10.6)


def readout(f):
    region(f,1323,300,191,376,EDGE,True)
    f.text(1419,326,'RGC 読み出し',size=13.2,bold=True)
    f.rect(1348,373,139,126,'white','#7f7f87',.7,rounded=True)
    f.line([(1342,395),(1342,430)],lw=.8,z=30)
    f.arrow([(1342,413),(1348,413)],lw=.9)
    f.text(1417,393,'空間平均',size=11.8)
    f.text(1417,421,'基線・尺度調整',size=11.0)
    f.text(1417,450,'閾値・利得',size=11.8)
    f.text(1417,479,'順応・偏置',size=11.8)
    f.arrow([(1418,502),(1418,522)],lw=1)
    f.node(1418,541,r'$\Sigma$',r=17,size=16)
    p=f.rect(1340,591,157,63,'white','#7e778a',.7,rounded=True)
    p.set_linestyle((0,(3,2)))
    f.text(1418,609,'過去の実測発火',size=11.4)
    f.text(1418,634,'1 bin遅延・τ=30 ms',size=10.2)
    f.arrow([(1418,588),(1418,561)],dash=True,lw=.9)
    f.text(1439,568,'−',size=11)
    f.arrow([(1438,541),(1532,541)],lw=1)
    f.text(1506,520,r'$z(t)$',size=11.8)
    f.rect(1534,519,50,45,'white','#777c83',.7,rounded=True)
    f.text(1559,541,r'$\sigma$',size=16.5)
    f.arrow([(1587,541),(1610,541)],lw=1)
    f.text(1609,456,'発火確率',size=12.6,bold=True)
    f.text(1609,485,'単一 RGC',size=10.8)
    f.text(1635,541,r'$p(t)$',size=17)
    f.text(1634,578,r'$T\times1$',size=12.2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language',choices=('ja','en'),default='ja')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'qa').mkdir(exist_ok=True)
    fonts=(FONT_EN,FONT_EN_BOLD) if args.language=='en' else (FONT,FONT_BOLD)
    for font in fonts:font_manager.fontManager.addfont(font)
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans','axes.unicode_minus':False,'image.composite_image':False})
    d=frozen_data();f=Figure(args.language)
    inputs_and_h1(f,d);bc_path(f,d);bc_path(f,d,True);ac_path(f);conductance_and_membrane(f);readout(f)
    f.save()
    d['details'].update(reference_path=str(REFERENCE),reference_sha256=sha(REFERENCE),layout='Rebuild of 2026-09-17 reference with compact BC branches and lower AC group; no curve plots or spike bars; code-faithful E/I paths; white region backgrounds.',canvas=[W,H],fonts=list(fonts),language=args.language)
    (OUT/f'qa/{f.stem}_sources.json').write_text(json.dumps(d['details'],ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Exported {f.stem}: SVG / PDF / PNG without plots or spike bars.')


if __name__=='__main__':main()
