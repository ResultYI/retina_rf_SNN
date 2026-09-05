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

named=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
def leaves(start):
    seen=set(); todo=[start]; found=set()
    while todo:
        node=todo.pop()
        if node is None or node in seen: continue
        seen.add(node)
        variable=getattr(node,'variable',None)
        if variable is not None: found.add(id(variable))
        todo.extend(v for v,_ in node.next_functions)
    return found
direct_leaves=leaves(base['BC_direct'].grad_fn); broad_leaves=leaves(base['BC_broad'].grad_fn)
shared=[{'name':n,'object_id':id(p),'in_direct_graph':id(p) in direct_leaves,'in_broad_graph':id(p) in broad_leaves,'data_ptr':p.data_ptr(),'state_key':n} for n,p in named if n.startswith(('feature_bank.','shared_subunits.','bipolar.'))]
degree=torch.bincount(m.shared_subunits.edge_index[0],minlength=6)
dead=[i for i,(a,b) in enumerate(m.shared_subunits.edge_index.T.tolist()) if degree[a]==1]
selfsaved=m.shared_subunits.raw_connections.detach().clone()
with torch.no_grad():
    for i in dead: m.shared_subunits.raw_connections[i].add_(1.)
_,selfchanged=run()
dead_effect=cmp(base['logits'],selfchanged['logits'])
with torch.no_grad(): m.shared_subunits.raw_connections.copy_(selfsaved)
from dataclasses import replace
variants={}
for variant,flags in [('aggregate',{'cell_specific_gains':True}),('pathway_mixture',{'cell_specific_pathway_mixture':True})]:
    vm=build_mechanistic_retina(replace(m.config,**flags),cp,rp,types,pol).eval()
    vo=vm.forward_sequence(x,observed_counts=h)
    vn=[(n,p) for n,p in vm.named_parameters() if p.requires_grad]
    vg=torch.autograd.grad(vo.logits.square().sum(),[p for _,p in vn],allow_unused=True)
    vi=[id(p) for group in build_phase1_optimizer(vm,learning_rate=.001).param_groups for p in group['params']]
    variants[variant]={'trainable_tensor_count':len(vn),'trainable_scalar_count':sum(p.numel() for _,p in vn),'all_connected':all(g is not None for g in vg),'all_optimizer_once':all(vi.count(id(p))==1 for _,p in vn),'additional_parameters':[{'name':n,'shape':list(p.shape),'object_id':id(p),'storage_ptr':p.data_ptr(),'state_dict_same_storage':vm.state_dict()[n].data_ptr()==p.data_ptr(),'optimizer_count':vi.count(id(p)),'gradient_max_abs':float(g.abs().max())} for (n,p),g in zip(vn,vg) if n.startswith('cell_gains.')]}
mini_cp=torch.tensor([[0.,0.],[.07,0.]],dtype=torch.float64)
mini=build_mechanistic_retina(m.config,mini_cp,mini_cp,('midget','midget'),('ON','ON')).eval()
mx=torch.sin(torch.arange(80,dtype=torch.float64).reshape(1,40,2)*.13).requires_grad_()
mo=mini.forward_sequence(mx,observed_counts=torch.zeros(1,40,2,dtype=torch.float64),clamps=frozenset({PathwayClamp.H1}))
mg=torch.autograd.grad(mo.bc_direct_presynaptic[0,-1,0].sum(),mx)[0]
print(json.dumps({'shared_parameter_graph_identity':shared,'dead_singleton_entries':dead,'singleton_edge_degree':degree.tolist(),'singleton_perturb_logit':dead_effect,'optional_gain_variants':variants,'two_cell_spatial_counterexample':{'BC_mask':mini.feature_bank.bc_support.tolist(),'AC_mask':mini.feature_bank.ac_support.tolist(),'mixing':mini.shared_subunits.connection_matrix().tolist(),'direct_broad_equal':cmp(mo.bc_direct_presynaptic,mo.bc_broad_presynaptic),'direct_cell0_outside_radius_gradient_max':float(mg[...,1].abs().max()),'outside_distance':.07,'BC_radius':.06}},indent=2))

