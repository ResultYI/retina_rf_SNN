# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.6.0"]
# ///
# Run from repository root with D:/anaconda/python.exe -B <this-file>.
# Audit-only reproduction: no training, optimizer steps, or checkpoint IO.
import json,sys,torch
sys.path.insert(0,'.')
from models.mechanistic_retina.contracts import ArchitectureMode,MechanisticRetinaConfig,PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.optimizer import build_phase1_optimizer
torch.set_num_threads(1); torch.manual_seed(831); torch.use_deterministic_algorithms(True)
axis=torch.arange(-6,7,dtype=torch.float64)*.04
cp=torch.cartesian_prod(axis,axis)
rp=torch.tensor([[0,0],[.04,0],[0,.04],[0,0],[.04,0],[0,.04]],dtype=torch.float64)
types=('midget','midget','midget','parasol','parasol','parasol'); pol=('ON','ON','OFF','ON','ON','OFF')
m=build_mechanistic_retina(MechanisticRetinaConfig(architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE),cp,rp,types,pol).eval()
t=torch.arange(40,dtype=torch.float64).view(1,40,1); c=torch.arange(len(cp),dtype=torch.float64).view(1,1,-1)
x=(.3*torch.sin(.23*t+.17*c)+.2*torch.cos(.11*t-.07*c)+.01*t).requires_grad_()
h=torch.zeros(1,40,6,dtype=torch.float64); h[:,2::7,:]=1
capture={}; identities=[]
def hh(mod,args,out):
    capture.update(X_H1=out.modulated_cones,H1_graph=out.graph_drive,H1_state=out.state,H1_contribution=out.surround)
def ah(mod,args,out):
    capture.update(AC_input=args[0],AC_state=torch.stack((out.local_state,out.transient_state),-1))
def bh(mod,args,out):
    identities.append((id(mod),id(mod.raw_weights),mod.raw_weights.data_ptr(),mod.raw_weights.untyped_storage().data_ptr()))
handles=[m.h1.register_forward_hook(hh),m.amacrine.register_forward_hook(ah),m.bipolar.register_forward_hook(bh)]
def run(clamps=frozenset()):
    capture.clear()
    o=m.forward_sequence(x,observed_counts=h,clamps=clamps)
    return o,{**capture,'BC_direct':o.bc_direct_presynaptic,'BC_broad':o.bc_broad_presynaptic,'direct_current':torch.stack((o.bc_sustained_current,o.bc_transient_current),-1),'AC_current':torch.stack((o.amacrine_local_current,o.amacrine_transient_current),-1),'total_current':o.total_current,'logits':o.logits}
def cmp(a,b):
    return {'equal':torch.equal(a,b),'max_abs_difference':float((a-b).abs().max().detach()),'right_exact_zero':bool((b==0).all())}
normal,base=run(); initial_ids=identities.copy()
clamps={'H1-off':{PathwayClamp.H1},'direct-BC-off':{PathwayClamp.DIRECT_BC_SUSTAINED,PathwayClamp.DIRECT_BC_TRANSIENT},'AC-off':{PathwayClamp.AMACRINE_LOCAL,PathwayClamp.AMACRINE_TRANSIENT}}
matrix={}
for name,cl in clamps.items():
    _,now=run(frozenset(cl)); matrix[name]={k:cmp(v,now[k]) for k,v in base.items()}
opt=build_phase1_optimizer(m,learning_rate=.001)
optids=[id(p) for group in opt.param_groups for p in group['params']]
named=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
grads=torch.autograd.grad(normal.logits.square().sum(),[p for _,p in named],allow_unused=True,retain_graph=True)
ownership=[dict(name=n,shape=list(p.shape),numel=p.numel(),object_id=id(p),data_ptr=p.data_ptr(),storage_ptr=p.untyped_storage().data_ptr(),state_dict_same_storage=m.state_dict()[n].data_ptr()==p.data_ptr(),optimizer_identity_count=optids.count(id(p)),autograd_connected=g is not None,gradient_max_abs=None if g is None else float(g.abs().max())) for (n,p),g in zip(named,grads)]
perturb={}
for name in ('bipolar.raw_weights','feature_bank.raw_tau','feature_bank.raw_delay','amacrine.raw_tau','amacrine.raw_delay','gates.raw_h1_amplitude'):
    p=dict(m.named_parameters())[name]; saved=p.detach().clone()
    with torch.no_grad(): p.view(-1)[0].add_(.2)
    _,now=run(); perturb[name]={k:cmp(base[k],now[k]) for k in ('X_H1','BC_direct','BC_broad','AC_state','AC_current','logits')}
    with torch.no_grad(): p.copy_(saved)
saved=m.feature_bank.path_spatial_basis.clone()
with torch.no_grad(): m.feature_bank.path_spatial_basis[:,2:].copy_(m.feature_bank.path_spatial_basis[:,:2])
_,equal=run(); perturb['equal_support']=cmp(equal['BC_direct'],equal['BC_broad'])
with torch.no_grad(): m.feature_bank.path_spatial_basis.copy_(saved)
scalar=base['AC_state'].square().sum()
gb=torch.autograd.grad(scalar,base['BC_broad'],retain_graph=True)[0]
gx=torch.autograd.grad(scalar,x,retain_graph=True)[0]
via=torch.autograd.grad(base['BC_broad'],x,grad_outputs=gb,retain_graph=True)[0]
def reaches(start,target,blocked):
    seen=set(); todo=[start]
    while todo:
        node=todo.pop()
        if node is None or node in seen or node is blocked: continue
        seen.add(node)
        if getattr(node,'variable',None) is target: return True
        todo.extend(v for v,_ in node.next_functions)
    return False
dependency=dict(dAC_dBCbroad_max=float(gb.abs().max()),dAC_dX_max=float(gx.abs().max()),gradient_through_broad_only=cmp(gx,via),X_reachable=reaches(scalar.grad_fn,x,None),X_reachable_after_BCbroad_cut=reaches(scalar.grad_fn,x,base['BC_broad'].grad_fn),AC_input_is_BCbroad=base['AC_input'] is base['BC_broad'],BC_call_identities=initial_ids,optimizer_extra=len(set(optids)-{id(p) for _,p in named}))
gd=torch.autograd.grad(base['BC_direct'][0,-1,0].sum(),base['X_H1'],retain_graph=True)[0]
ga=torch.autograd.grad(base['BC_broad'][0,-1,0].sum(),base['X_H1'],retain_graph=True)[0]
bm=m.feature_bank.bc_support[0].bool(); am=m.feature_bank.ac_support[0].bool()
spatial=dict(nested=bool((m.feature_bank.bc_support<=m.feature_bank.ac_support).all()),strict_extension_each_cell=bool((m.feature_bank.ac_support>m.feature_bank.bc_support).any(-1).all()),direct_outside_BC_mask_max=float(gd[...,~bm].abs().max()),broad_outside_AC_mask_max=float(ga[...,~am].abs().max()),direct_outside_cones=((gd.abs().sum((0,1))>0)&~bm).nonzero().flatten().tolist(),broad_outside_cones=((ga.abs().sum((0,1))>0)&~am).nonzero().flatten().tolist())
print(json.dumps(dict(fixture=dict(seed=831,dtype=str(x.dtype),torch=torch.__version__,X_shape=list(x.shape),history_shape=list(h.shape),cell_positions=rp.tolist(),cell_types=types,polarities=pol),runtime=matrix,parameters=ownership,perturbations=perturb,dependency=dependency,spatial=spatial),indent=2))

